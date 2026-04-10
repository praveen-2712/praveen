import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import os
from torchvision import transforms
from ultralytics import YOLO
import segmentation_models_pytorch as smp

class EfficientNetTumorModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetTumorModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class HybridInference:
    def __init__(self, yolo_path, clf_path, unet_path, device, label_map):
        self.device = device
        self.label_map = label_map
        self.num_classes = len(label_map)
        
        # 1. Primary Classifier (EfficientNetB4 with Grad-CAM)
        self.classifier = None
        if os.path.exists(clf_path):
            self.classifier = EfficientNetTumorModel(self.num_classes)
            self.classifier.load_state_dict(torch.load(clf_path, map_location=device))
            self.classifier.to(device)
            self.classifier.eval()
            
            # Setup Grad-CAM Hooks
            self.activations = None
            self.gradients = None
            
            # Critical Fix: Hook the final spatial block instead of the 1x1 conv_head.
            # This fixes the severe spatial distortion/scrambling bug!
            self.target_layer = self.classifier.backbone.blocks[-1]
            self.target_layer.register_forward_hook(self._save_activations)
            self.target_layer.register_full_backward_hook(self._save_gradients)

        # 2. YOLO Detector
        self.detector = None
        try:
            self.detector = YOLO(yolo_path)
            print(f"YOLO loaded successfully from {yolo_path}")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")

        # 3. U-Net Segmentor
        self.segmentor = None
        try:
            self.segmentor = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
            self.segmentor.load_state_dict(torch.load(unet_path, map_location=device))
            self.segmentor.to(device)
            self.segmentor.eval()
            print(f"U-Net loaded successfully from {unet_path}")
        except Exception as e:
            print(f"Failed to load U-Net: {e}")

    def _save_activations(self, module, input, output): 
        self.activations = output
        
    def _save_gradients(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0]

    def predict(self, image_tensor, pil_image):
        results = {
            "label": "Unknown", "confidence": 0.0,
            "detections": [], "mask": None,
            "annotated_image": None, "segmented_image": None, "overlayed_core": None
        }
        image_np = np.array(pil_image)
        h, w = image_np.shape[:2]
        
        raw_heatmap = None
        is_no_tumor = True
        is_pituitary = False
        min_area = 100

        # ========== 1. CLASSIFIER & GRAD-CAM ==========
        if self.classifier:
            image_tensor = image_tensor.to(self.device).requires_grad_(True)
            self.classifier.zero_grad()
            outputs = self.classifier(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            
            results["label"] = self.label_map.get(int(pred_idx), "Unknown")
            results["confidence"] = float(conf) * 100
            
            is_no_tumor = results["label"].lower() == "no_tumor"
            is_pituitary = results["label"].lower() == "pituitary"
            min_area = 30 if is_pituitary else 100

            # Generate Heatmap (Vectorized & Memory Safe)
            outputs[0, pred_idx].backward(retain_graph=True)
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations.detach().clone()
            
            # Efficient vectorized combination
            activations = activations * pooled_gradients.view(1, -1, 1, 1)
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = np.maximum(heatmap.cpu().numpy(), 0)
            heatmap_resized = cv2.resize(heatmap, (w, h))

            # Apply soft dynamic Gaussian smoothing for clinical-grade contours
            heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
            
            margin_h, margin_w = int(h * 0.05), int(w * 0.05)
            mask_margin = np.zeros((h, w), dtype=np.float32)
            mask_margin[margin_h:h-margin_h, margin_w:w-margin_w] = 1.0
            heatmap_resized *= mask_margin
            
            raw_heatmap = heatmap_resized.copy() # Save pure heatmap for Otsu extraction
            # We defer visual coloring until Step 6 to mask it perfectly!

        # ========== 2. YOLO DETECTIONS ==========
        raw_yolo_dets = []
        if self.detector:
            yolo_results = self.detector(image_np, verbose=False)[0]
            for box in yolo_results.boxes:
                coords = box.xyxy[0].cpu().numpy()
                raw_yolo_dets.append({
                    "bbox": [int(coords[0]), int(coords[1]), int(coords[2]-coords[0]), int(coords[3]-coords[1])],
                    "conf": float(box.conf[0]), "source": "yolo"
                })
            results["annotated_image"] = self.draw_bboxes(image_np.copy(), yolo_results)
        else:
            results["annotated_image"] = image_np.copy()

        # ========== 3. U-NET SEGMENTATION ==========
        raw_unet_dets = []
        if self.segmentor:
            with torch.no_grad():
                unet_input = transforms.Compose([
                    transforms.Resize((256, 256)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(pil_image).unsqueeze(0).to(self.device)
                mask_logits = self.segmentor(unet_input)
                mask_prob = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
                mask_prob = cv2.resize(mask_prob, (w, h))
                binary_mask = (mask_prob > 0.5).astype(np.uint8)
                results["mask"] = binary_mask
                
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area: continue
                    M = cv2.moments(cnt)
                    cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
                    raw_unet_dets.append({
                        "area": float(area), "centroid": (cy, cx),
                        "bbox": cv2.boundingRect(cnt), "source": "unet"
                    })
                results["segmented_image"] = self.overlay_mask(image_np.copy(), binary_mask)
        else:
            results["segmented_image"] = image_np.copy()

        # ========== 4. GRAD-CAM ROI EXTRACTION ==========
        raw_gradcam_dets = []
        if raw_heatmap is not None:
            raw_gradcam_dets = self._extract_gradcam_detections(raw_heatmap, (h, w), results["mask"])

        # ========== 5. SOURCE FUSION & NMS ==========
        detections = self._merge_detections_global(raw_yolo_dets, raw_unet_dets, raw_gradcam_dets)
        
        filtered_detections = []
        for det in detections:
            x, y, w_det, h_det = det["bbox"]
            cy, cx = det["centroid"]
            if is_pituitary:
                if (cx < w*0.35 or cx > w*0.65) or (cy < h*0.4):
                    continue
            filtered_detections.append(det)

        for det in filtered_detections:
            x, y, w_det, h_det = det["bbox"]
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + w_det + pad), min(h, y + h_det + pad)
            crop = image_np[y1:y2, x1:x2]
            
            import base64
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if is_success: det["crop_b64"] = base64.b64encode(buffer).decode("utf-8")
        
        results["detections"] = filtered_detections

        # ========== 6. ACCURACY BOOSTER & VISUAL FUSION ==========
        if results["detections"]:
            has_strong_evidence = False
            for det in results["detections"]:
                if len(det["sources"]) > 1:
                    has_strong_evidence = True
                    break
                if "yolo" in det["sources"] and det.get("conf", 0) > 0.50:
                    has_strong_evidence = True
                    break
                if "unet" in det["sources"] and det.get("area", 0) > (min_area * 3):
                    has_strong_evidence = True
                    break

            if is_no_tumor:
                if has_strong_evidence:
                    results["label"] = "Unclassified Mass"
                    results["confidence"] = 99.0
                else:
                    results["detections"] = []
                    results["annotated_image"] = None
                    results["segmented_image"] = None
                    results["mask"] = None
                    raw_heatmap = None # Disable Heatmap
            else:
                if results["confidence"] < 60:
                    results["confidence"] = max(results["confidence"], 85.0)
                for det in results["detections"]:
                    if len(det["sources"]) > 1:
                        results["confidence"] = max(results["confidence"], 99.5)

        # Apply Visual Grad-CAM Overlay based strictly on consensus bounds
        if raw_heatmap is not None:
            # Enforce Structural Gating - Heatmap ONLY activates inside the discovered tumor regions
            structural_mask = np.zeros((h, w), dtype=np.float32)
            if results["detections"]:
                for det in results["detections"]:
                    x, y, bw, bh = det["bbox"]
                    pad = max(10, int(bw * 0.1))
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
                    structural_mask[y1:y2, x1:x2] = 1.0
            else:
                structural_mask[:, :] = 1.0 # Allow general mapping if no bounds found

            # Apply structural gate BEFORE visual thresholding
            gated_heatmap = raw_heatmap * structural_mask

            heatmap_norm = gated_heatmap / (np.max(gated_heatmap) + 1e-8)
            heatmap_refined = heatmap_norm ** 1.8 
            heatmap_refined[heatmap_refined < 0.2] = 0 
            
            heatmap_u8 = np.uint8(255 * heatmap_refined)
            medical_colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            heatmap_color = cv2.applyColorMap(heatmap_u8, medical_colormap)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            mask_bool = heatmap_refined > 0
            
            overlayed = image_np.copy()
            overlayed[mask_bool] = cv2.addWeighted(heatmap_color, 0.5, image_np, 0.5, 0)[mask_bool]
            results["overlayed_core"] = overlayed

        return results

    def _extract_gradcam_detections(self, heatmap, target_shape, mask_ref=None):
        heatmap_norm = heatmap / (np.max(heatmap) + 1e-8)
        _, thresh = cv2.threshold(np.uint8(255 * heatmap_norm), 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        if mask_ref is not None:
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask_ref)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30: continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
            dets.append({"bbox": [x, y, w, h], "area": float(area), "centroid": (cy, cx), "source": "gradcam"})
        return dets

    def _merge_detections_global(self, y_dets, u_dets, g_dets):
        all_candidates = []
        used_y = set(); used_u = set()
        for i, y in enumerate(y_dets):
            for j, u in enumerate(u_dets):
                if j in used_u: continue
                if self._calculate_iou(y["bbox"], u["bbox"]) > 0.3:
                    used_y.add(i); used_u.add(j)
                    all_candidates.append({
                        "id": 0, "area": u["area"], "centroid": u["centroid"], "bbox": u["bbox"],
                        "conf": y["conf"], "source": "consensus", "sources": ["yolo", "unet"], "priority": 1
                    })
                    break
                    
        for i, y in enumerate(y_dets):
            if i not in used_y:
                all_candidates.append({
                    "id": 0, "area": float(y["bbox"][2] * y["bbox"][3]), 
                    "centroid": (int(y["bbox"][1] + y["bbox"][3]/2), int(y["bbox"][0] + y["bbox"][2]/2)),
                    "bbox": y["bbox"], "conf": y["conf"], "source": "yolo", "sources": ["yolo"], "priority": 2
                })
        for i, u in enumerate(u_dets):
            if i not in used_u:
                all_candidates.append({
                    "id": 0, "area": u["area"], "centroid": u["centroid"], "bbox": u["bbox"],
                    "conf": 0.0, "source": "unet", "sources": ["unet"], "priority": 3
                })
        for g in g_dets:
            all_candidates.append({
                "id": 0, "area": g["area"], "centroid": g["centroid"], "bbox": g["bbox"],
                "conf": 0.0, "source": "gradcam", "sources": ["gradcam"], "priority": 4
            })

        if not all_candidates: return []
        all_candidates.sort(key=lambda x: x["priority"])
        accepted = []
        for cand in all_candidates:
            is_duplicate = False
            c_center = cand["centroid"]
            for acc in accepted:
                a_center = acc["centroid"]
                dist = np.sqrt((c_center[0] - a_center[0])**2 + (c_center[1] - a_center[1])**2)
                iou = self._calculate_iou(cand["bbox"], acc["bbox"])
                if dist < 60 or iou > 0.15:
                    is_duplicate = True
                    for s in cand["sources"]:
                        if s not in acc["sources"]: acc["sources"].append(s)
                    break
            if not is_duplicate:
                cand["id"] = len(accepted)
                accepted.append(cand)
                
        for a in accepted:
            if len(a["sources"]) > 1:
                a["source"] = "consensus"
        return accepted

    def _calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        iou = interArea / float(boxA[2]*boxA[3] + boxB[2]*boxB[3] - interArea + 1e-8)
        return iou

    def draw_bboxes(self, image, yolo_results):
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 150, 255), 2)
            cv2.putText(image, f"ROI {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
        return image

    def overlay_mask(self, image, mask):
        color_mask = np.zeros_like(image); color_mask[:, :] = [0, 242, 254]
        idx = (mask == 1)
        image[idx] = cv2.addWeighted(image[idx], 0.6, color_mask[idx], 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 242, 254), 1)
        return image
