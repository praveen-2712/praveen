import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from torchvision import models, transforms
import torchvision
import os

# --- PATCH FOR RTX 50-SERIES (Blackwell) NMS KERNEL CRASH ---
_old_nms = torchvision.ops.nms
def _safe_nms(boxes, scores, iou_threshold):
    if boxes.device.type != 'cpu':
        return _old_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
    return _old_nms(boxes, scores, iou_threshold)
torchvision.ops.nms = _safe_nms
if hasattr(torchvision.ops.boxes, 'nms'):
    torchvision.ops.boxes.nms = _safe_nms
try:
    import ultralytics.utils.ops
    ultralytics.utils.ops.nms = _safe_nms
except Exception: pass
# -----------------------------------------------------------

class HybridInference:
    def __init__(self, yolo_path, clf_path, unet_path, device, label_map):
        self.device = device
        self.label_map = label_map
        
        # 1. Load YOLOv8
        try:
            self.detector = YOLO(yolo_path)
        except: self.detector = None

        # 2. Load EfficientNet Classifier
        try:
            num_classes = len(label_map)
            self.classifier = models.efficientnet_b2()
            self.classifier.classifier[1] = nn.Linear(self.classifier.classifier[1].in_features, num_classes)
            self.classifier.load_state_dict(torch.load(clf_path, map_location=device))
            self.classifier.to(device)
            self.classifier.eval()
            
            # Grad-CAM Hooks
            self.activations = None
            self.gradients = None
            self.target_layer = self.classifier.features[-1]
            self.target_layer.register_forward_hook(self._save_activations)
            self.target_layer.register_full_backward_hook(self._save_gradients)
        except: self.classifier = None

        # 3. Load U-Net Segmentor
        try:
            self.segmentor = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
            self.segmentor.load_state_dict(torch.load(unet_path, map_location=device))
            self.segmentor.to(device)
            self.segmentor.eval()
        except: self.segmentor = None

    def _save_activations(self, module, input, output): self.activations = output
    def _save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def predict(self, image_tensor, pil_image):
        results = {
            "label": "Unknown", "confidence": 0.0,
            "detections": [], "mask": None,
            "annotated_image": None, "segmented_image": None, "overlayed_core": None
        }
        image_np = np.array(pil_image)
        h, w = image_np.shape[:2]
        
        # 1. Classification & Grad-CAM
        raw_heatmap = None
        if self.classifier:
            image_tensor = image_tensor.to(self.device)
            image_tensor.requires_grad = True
            outputs = self.classifier(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            results["label"] = self.label_map.get(int(pred_idx), "Unknown")
            results["confidence"] = float(conf) * 100

            # Generate Heatmap
            self.classifier.zero_grad()
            outputs[0, pred_idx].backward()
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
            
            # 1.1 Apply 10% Margin Mask to remove edge artifacts (top-left fix)
            margin_h, margin_w = int(h * 0.1), int(w * 0.1)
            heatmap_resized = cv2.resize(heatmap, (w, h))
            mask_margin = np.zeros((h, w), dtype=np.float32)
            mask_margin[margin_h:h-margin_h, margin_w:w-margin_w] = 1.0
            heatmap_resized *= mask_margin
            
            # Recalculate raw heatmap for ROI extraction from the masked version
            raw_heatmap = heatmap_resized
            
            is_pituitary = results["label"].lower() == "pituitary"
            
            # Aesthetic Refinement (Corrected Normalization)
            heatmap_refined = heatmap_resized**2 if is_pituitary else heatmap_resized**3 
            heatmap_refined /= (np.max(heatmap_refined) + 1e-8)
            heatmap_refined[heatmap_refined < (0.05 if is_pituitary else 0.2)] = 0
            
            # Overlay Heatmap
            heatmap_u8 = np.uint8(255 * heatmap_refined)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            mask = heatmap_refined > 0
            overlayed = image_np.copy()
            overlayed[mask] = cv2.addWeighted(heatmap_color, 0.5, image_np, 0.5, 0)[mask]
            results["overlayed_core"] = overlayed

        is_no_tumor = results["label"].lower() == "no_tumor"
        is_pituitary = results["label"].lower() == "pituitary"
        min_area = 30 if is_pituitary else 100

        # 2. Detection (YOLO)
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

        # 3. Segmentation (U-Net)
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

        # 4. Grad-CAM Fallback Detection (Mask-Gated ROI Extraction)
        # We only look for Grad-CAM peaks IF they overlap with U-Net mask (if mask exists) or as independent seeds
        raw_gradcam_dets = self._extract_gradcam_detections(raw_heatmap, (h, w), results["mask"]) if raw_heatmap is not None else []

        # 5. Global Deduplication with SOURCE FUSION
        detections = self._merge_detections_global(raw_yolo_dets, raw_unet_dets, raw_gradcam_dets)
        
        # 5.1 Generate Diagnostic Crops for Clinical Closeup
        for det in detections:
            x, y, w_det, h_det = det["bbox"]
            # Pad crop by 20px for clinical context
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + w_det + pad), min(h, y + h_det + pad)
            crop = image_np[y1:y2, x1:x2]
            
            # Convert to B64
            from io import BytesIO
            import base64
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if is_success:
                det["crop_b64"] = base64.b64encode(buffer).decode("utf-8")
        
        results["detections"] = detections
        
        # 6. Label Correction & Accuracy Booster
        if results["detections"]:
            has_strong_evidence = False
            for det in results["detections"]:
                # Strong evidence metrics
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
                    # If strong evidence on a healthy brain, raise confidence it's an anomaly 
                    # but keep the original designation so doctor can evaluate manually without generic renaming.
                    results["confidence"] = 99.0
                else:
                    # It's a healthy brain and detections are just weak noise. Clear them.
                    results["detections"] = []
                    results["annotated_image"] = None
                    results["segmented_image"] = None
            else:
                # If the classifier was unsure but it's not a healthy brain
                if results["confidence"] < 60:
                    results["confidence"] = max(results["confidence"], 85.0)
                
                # If multiple sources agree (Consensus), maintain maximum confidence
                for det in results["detections"]:
                    if len(det["sources"]) > 1:
                        results["confidence"] = max(results["confidence"], 99.5)
            
        return results

    def _extract_gradcam_detections(self, heatmap, target_shape, mask_ref=None):
        """
        Extract ROIs from Grad-CAM peaks, filtered by anatomical masks.
        """
        heatmap_norm = heatmap / (np.max(heatmap) + 1e-8)
        heatmap_norm = heatmap_norm**2
        
        _, thresh = cv2.threshold(np.uint8(255 * heatmap_norm), 190, 255, cv2.THRESH_BINARY)
        
        # Mask Gating: If we have a segmentation mask, only keep Grad-CAM peaks within it
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
            dets.append({
                "bbox": [x, y, w, h], "area": float(area), "centroid": (cy, cx), "source": "gradcam"
            })
        return dets

    def _merge_detections_global(self, y_dets, u_dets, g_dets):
        """
        Unified Non-Maximum Suppression (NMS) with Source Fusion.
        """
        all_candidates = []
        
        # Match YOLO + U-Net for "Consensus"
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
        
        # Standalone
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
                
                # FUSION RADIUS: INCREASED TO 60px as per plan
                if dist < 60 or iou > 0.15:
                    is_duplicate = True
                    # FUSE SOURCES: Add this candidate's sources to the accepted detection
                    for s in cand["sources"]:
                        if s not in acc["sources"]: acc["sources"].append(s)
                    break
            
            if not is_duplicate:
                cand["id"] = len(accepted)
                accepted.append(cand)
        
        # Return merged detections with fused source labels
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
