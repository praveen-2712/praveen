import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from preprocess import Preprocess
from train_classifier import DualBranchClassifier

# ── Clinical threshold loader ──────────────────────────────────────────────────
# Reads GLIOMA_THRESHOLD from weights/clinical_thresholds.json (written by
# calibrate.py). Falls back to 0.30 if the file is missing so the engine
# always starts cleanly even on a fresh clone.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THRESHOLD_PATH = os.path.join(_BASE_DIR, "weights", "clinical_thresholds.json")

def _load_thresholds():
    if os.path.exists(_THRESHOLD_PATH):
        try:
            with open(_THRESHOLD_PATH) as _f:
                _data = json.load(_f)
            _gt = float(_data.get("glioma_threshold", 0.30))
            _it = float(_data.get("indeterminate_threshold", 0.55))
            # Sanity-check: reject degenerate values (e.g. 1e-44 from broken calibration)
            if _gt < 1e-6 or _gt > 0.99:
                print(f"[Engine] WARNING: glioma_threshold={_gt:.2e} is out of range — using 0.30 fallback.")
                _gt = 0.30
            print(f"[Engine] Loaded clinical thresholds: glioma={_gt:.4f}, indeterminate={_it:.2f}")
            return _gt, _it
        except Exception as _e:
            print(f"[Engine] WARNING: Could not load clinical_thresholds.json ({_e}) — using defaults.")
    else:
        print(f"[Engine] WARNING: clinical_thresholds.json not found — using defaults (0.30 / 0.55).")
    return 0.30, 0.55

GLIOMA_THRESHOLD, INDETERMINATE_THRESHOLD = _load_thresholds()

# Load calibration temperature (applied to logits before softmax)
_CALIB_TEMP_PATH = os.path.join(_BASE_DIR, "weights", "calibration_temperature.pt")
CALIBRATION_TEMPERATURE = 1.0
try:
    if os.path.exists(_CALIB_TEMP_PATH):
        _calib_data = torch.load(_CALIB_TEMP_PATH, map_location="cpu", weights_only=False)
        _raw_temp = float(_calib_data.get("temperature", 1.0))
        CALIBRATION_TEMPERATURE = max(0.3, min(_raw_temp, 15.0))
        if CALIBRATION_TEMPERATURE != _raw_temp:
            logging.warning(
                f"[Engine] Calibration temperature {_raw_temp:.4f} was clamped to "
                f"{CALIBRATION_TEMPERATURE:.4f} (bounds: [0.3, 15.0])"
            )
        else:
            print(f"[Engine] Calibration temperature loaded: T = {CALIBRATION_TEMPERATURE:.4f}")
except Exception as _e:
    print(f"[Engine] WARNING: Could not load calibration temperature ({_e}) — using T=1.0.")

# ─────────────────────────────────────────────────────────────────────────────
#  Display-friendly label map
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "notumor":    "No Tumor Detected",
    "no_tumor":   "No Tumor Detected",
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "pituitary":  "Pituitary Tumor",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM  (Standard Grad-CAM targeting the final MBConv block)
#
#  Key fixes over previous implementation:
#   1. CAM forward pass is done OUTSIDE torch.no_grad() so gradients flow.
#   2. Hook captures the *output* tensor directly, not go[0] (which is
#      an input-gradient, not feature-gradient in full_backward_hook).
#   3. Proper percentile-based low-activation masking (≥10th pct threshold)
#      to remove noisy, diffuse background activations.
#   4. Larger Gaussian sigma (sigmaX=5) for medically smooth heatmaps.
#   5. Heatmap thresholded to zero below 20% of its peak to suppress clutter.
#   6. Brain mask gating applied AFTER smoothing (correct order).
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM++ targeting model.blocks[-1] (final EfficientNet-B4 MBConv block).
    Provides superior spatial localization over standard Grad-CAM by computing
    second and third-order derivatives to weight positive gradients.
    Falls back to Input-Gradient Saliency if gradients are empty/zero.
    """

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._handles    = []
        self._attach_hooks()

    def _attach_hooks(self):
        """
        Upgraded Hook Registration Engine:
        Automatically targets the deepest pointwise convolutional head (e.g. model.conv_head)
        for the absolute highest resolution explainability, falling back gracefully
        if model architectures are modified.
        """
        try:
            target = None
            layer_name = ""

            if hasattr(self.model, "branch_global"):
                model_to_hook = self.model.branch_global
            else:
                model_to_hook = self.model
                
            # Priority 1: Pointwise conv head (standard for EfficientNet V1/V2)
            if hasattr(model_to_hook, "conv_head") and model_to_hook.conv_head is not None:
                target = model_to_hook.conv_head
                layer_name = "conv_head"
            # Priority 2: Final sequential blocks container
            elif hasattr(model_to_hook, "blocks") and len(model_to_hook.blocks) > 0:
                target = model_to_hook.blocks[-1]
                layer_name = "blocks[-1]"
            # Priority 3: standard ResNet layer4
            elif hasattr(model_to_hook, "layer4") and len(model_to_hook.layer4) > 0:
                target = model_to_hook.layer4[-1]
                layer_name = "layer4[-1]"
            # Fallback: Find the very last active convolutional layer
            else:
                for module in reversed(list(self.model.modules())):
                    if isinstance(module, nn.Conv2d):
                        target = module
                        layer_name = f"Conv2d ({module.out_channels}ch)"
                        break

            if target is None:
                raise ValueError("Could not auto-detect any suitable target convolutional layer.")

            def _fwd_hook(module, input, output):
                self.activations = output.detach()

            def _bwd_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            self._handles.append(target.register_forward_hook(_fwd_hook))
            self._handles.append(
                target.register_full_backward_hook(_bwd_hook)
            )
            print(f"[Grad-CAM] Successfully targeted and hooked to layer: '{layer_name}'")
        except Exception as e:
            print(f"[Grad-CAM] Hook registration failed: {e}")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor, crop_tensor, class_idx):
        """
        Returns a normalised float32 heatmap in [0, 1] at the same spatial
        size as input_tensor.

        IMPORTANT: This method must be called WITHOUT wrapping in
        torch.no_grad(), because we need gradients to flow back through
        the network. The caller is responsible for this.
        """
        self.gradients   = None
        self.activations = None
        self.model.zero_grad()

        # Ensure the input requires grad so backward() works
        inp = input_tensor.detach().clone().requires_grad_(True)
        crop_inp = crop_tensor.detach().clone().requires_grad_(True)

        # Forward pass — hooks capture activations
        output = self.model(inp, crop_inp)
        score  = output[0, class_idx]

        # Backward pass — hooks capture gradients
        score.backward()

        if self.gradients is None or self.activations is None:
            print("[Grad-CAM] Hooks returned None — computing saliency fallback.")
            return self._saliency_fallback(inp)

        # ── Grad-CAM++ Weighting ───────────────────────────────────────
        grads = self.gradients
        acts  = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = acts.sum(dim=(2, 3), keepdim=True)

        eps = 1e-7
        denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = torch.where(denom != 0.0, grads_power_2 / denom, torch.zeros_like(grads))
        
        # Only positive gradients contribute to the spatial weights
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1,1,H',W')
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        hm = cam[0, 0].cpu().numpy()  # (H, W)

        # Check that the map has meaningful signal
        if hm.max() < 1e-7:
            print("[Grad-CAM] CAM is essentially zero — using saliency fallback.")
            return self._saliency_fallback(inp)

        # ── Post-processing ───────────────────────────────────────────────
        # 1. Gaussian smooth before stretching to remove random noise spikes
        import cv2 as _cv2
        hm = _cv2.GaussianBlur(hm, (0, 0), sigmaX=3)

        # 2. Tighter percentile stretch (5th–95th) — clips background dominance
        p5, p95 = np.percentile(hm, 5), np.percentile(hm, 95)
        hm = np.clip((hm - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)

        # 3. Aggressive suppression of low-activation background (< 40% of peak)
        hm[hm < 0.40] = 0.0

        # 4. Re-normalise so the hotspot always reaches 1.0
        if hm.max() > 1e-8:
            hm = hm / (hm.max() + 1e-8)

        return hm.astype(np.float32)

    @staticmethod
    def _saliency_fallback(inp):
        """Input-Gradient Saliency as a last resort."""
        if inp.grad is None:
            return np.zeros(inp.shape[2:], dtype=np.float32)
        sal = inp.grad[0].abs().mean(0).cpu().numpy()
        if sal.max() > 1e-8:
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Clinical Inference Engine  (YOLO-free edition)
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalInferenceEngine:
    """
    Neurologix — Grad-CAM-fixed Clinical Engine.

    Models active:
      - EfficientNet-B4   — 4-class classifier (+ MC-Dropout uncertainty)
      - YOLO11m           — Bounding box detection
      - U-Net (ResNet50)  — Pixel-precise tumour segmentation
    """

    def __init__(self, classifier_path, detector_path,
                 unet_path, device, label_map):
        self.device    = device
        self.label_map = label_map          # {int_idx: raw_key}
        self.preprocessor = Preprocess()

        # ── Classifier (DualBranchClassifier Upgraded) ────────────────────────
        self.classifier = DualBranchClassifier(num_classes=len(label_map))
        if os.path.exists(classifier_path):
            sd = torch.load(classifier_path, map_location=device, weights_only=False)
            sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
            self.classifier.load_state_dict(sd, strict=True)
        self.classifier.to(device)

        # BatchNorm in eval mode; Dropout layers left in train mode for MC-Dropout
        self.classifier.eval()
        self._set_dropout_train(self.classifier)

        # ── Grad-CAM explainer ────────────────────────────────────────────
        self.explainer = GradCAM(self.classifier)

        # ── U-Net segmentor (UnetPlusPlus — matches train_unet.py architecture) ────
        self.segmentor = None
        if os.path.exists(unet_path):
            self.segmentor = smp.UnetPlusPlus(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=1, classes=1).to(device)
            self.segmentor.load_state_dict(
                torch.load(unet_path, map_location=device, weights_only=False))
            self.segmentor.eval()

        # ── YOLO11m Detector ───────────────────────────────────────────────
        self.detector = None
        if os.path.exists(detector_path):
            from ultralytics import YOLO
            self.detector = YOLO(detector_path)

        self.T = CALIBRATION_TEMPERATURE
        self.GLIOMA_THRESHOLD = GLIOMA_THRESHOLD
        self.TTA_ENABLED = False

        print(f"[Engine] Device={device} | "
              f"Detector={'loaded' if self.detector else 'missing'} | "
              f"UNet={'loaded' if self.segmentor else 'missing'}")

    # ─────────────────── helpers ──────────────────────────────────────────────

    def _set_dropout_train(self, model):
        """Re-enable only Dropout layers for MC-Dropout sampling."""
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @staticmethod
    def _make_heatmap_overlay(canvas_rgb, heatmap_01, alpha=0.5):
        hm_u8 = np.uint8(255 * heatmap_01)
        cmap  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cmap  = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(canvas_rgb, alpha, cmap, alpha, 0)

    def _prepare_crop_tensor(self, raw_image, roi_crop_img, brain_mask):
        """
        Prepare branch_local (crop) tensor — MUST match dataset._get_local_crop()
        training distribution exactly.

        Training used:  RGB crop -> resize(380,380) -> A.Normalize(ImageNet)
        Previous (WRONG): gray -> 3x CLAHE -> normalize  <- model never saw this

        Fix: match training exactly — RGB crop, no CLAHE, standard ImageNet norm.
        """
        IMG_SIZE = 380
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # Determine the crop region (same priority as dataset._get_local_crop)
        if roi_crop_img is not None and roi_crop_img.size > 0:
            crop_base = roi_crop_img   # YOLO ROI crop — already RGB
        else:
            # Brain mask bounding box fallback
            y_idx, x_idx = np.where(brain_mask > 0)
            if len(y_idx) > 0 and len(x_idx) > 0:
                y1, y2 = y_idx.min(), y_idx.max()
                x1, x2 = x_idx.min(), x_idx.max()
                crop_base = raw_image[y1:y2, x1:x2]
            else:
                crop_base = raw_image

        # Ensure 3-channel RGB (raw_image is always RGB throughout the pipeline)
        if crop_base.ndim == 2:
            crop_base = cv2.cvtColor(crop_base, cv2.COLOR_GRAY2RGB)

        # Resize to training input size and normalize — identical to A.Normalize in training
        crop_resized = cv2.resize(crop_base, (IMG_SIZE, IMG_SIZE))
        crop_float   = crop_resized.astype(np.float32) / 255.0  # [0, 1]

        s_tensor = (torch.from_numpy(crop_float)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        .to(self.device))
        return (s_tensor - mean) / std

    # ─────────────────── main predict ─────────────────────────────────────────

    def predict(self, slice_stack, raw_image, mode="multi"):
        """
        Parameters
        ----------
        slice_stack : (1, 3, 224, 224) float tensor, already normalised.
        raw_image   : original full-resolution NumPy RGB array (H, W, 3).
        mode        : "binary" → Tumor / No Tumor
                      "multi"  → Glioma / Meningioma / Pituitary / No Tumor
        """
        H_orig, W_orig = raw_image.shape[:2]

        results = {
            "label": "No Tumor Detected", "confidence": 0.0,
            "status": "Success", "message": "",
            "heatmap_img": None, "detection_img": None,
            "segmentation_img": None, "roi_crop": None,
            "centroid": None, "mask_area": 0, "icv_pct": 0.0,
            "is_discordant": False, "used_yolo_override": False
        }

        # ── Brain mask (for anatomical gating) ───────────────────────────
        brain_mask   = self.preprocessor.get_brain_mask(raw_image)
        brain_pixels = int(np.sum(brain_mask > 0))

        # ── YOLO Detection (Moved UP for branch_local crop) ──────────────
        yolo_centroid = None
        roi_crop_img = None
        det_canvas = raw_image.copy()
        used_yolo_override = False
        det_conf = 0.0
        yolo_predicted_class = "notumor"
        yolo_bbox_diagonal = 0.0
        
        if self.detector:
            det_results = self.detector.predict(raw_image, conf=0.20, iou=0.4, verbose=False)
            if len(det_results) > 0 and len(det_results[0].boxes) > 0:
                best_box = det_results[0].boxes[0]
                box = best_box.xyxy[0].cpu().numpy().astype(int)
                det_conf = float(best_box.conf[0].cpu().item())
                
                yolo_cls_idx = int(best_box.cls[0].cpu().item())
                yolo_predicted_class = det_results[0].names[yolo_cls_idx].lower()
                
                w = float(box[2] - box[0])
                h = float(box[3] - box[1])
                yolo_bbox_diagonal = np.sqrt(w**2 + h**2)
                
                cv2.rectangle(det_canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw filled background behind label for readability
                _lbl = f"TUMOR  {det_conf*100:.0f}%"
                (_lw, _lh), _base = cv2.getTextSize(_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                _ly = max(box[1] - 4, _lh + 6)
                cv2.rectangle(det_canvas, (box[0], _ly - _lh - 6), (box[0] + _lw + 6, _ly + 2), (0, 0, 0), -1)
                cv2.putText(det_canvas, _lbl, (box[0] + 3, _ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
                pad = 15
                y1, y2 = max(0, box[1]-pad), min(H_orig, box[3]+pad)
                x1, x2 = max(0, box[0]-pad), min(W_orig, box[2]+pad)
                roi_crop_img = raw_image[y1:y2, x1:x2]

                yolo_centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                
        results["detection_img"] = det_canvas
        results["roi_crop"] = roi_crop_img

        # Prepare crop tensor for the dual-branch classifier
        crop_tensor = self._prepare_crop_tensor(raw_image, roi_crop_img, brain_mask)

        # PHASE 1 — Deterministic glioma check:
        # Call self.classifier.eval() to disable all dropout and freeze BatchNorm.
        glioma_idx = -1
        for idx_k, name in self.label_map.items():
            if name == "glioma":
                glioma_idx = idx_k
                break

        glioma_raw_p = 0.0
        glioma_flag = False
        if glioma_idx != -1:
            self.classifier.eval()
            with torch.no_grad():
                _raw_out = self.classifier(slice_stack, crop_tensor)
                _raw_p_unscaled = torch.softmax(_raw_out, dim=1).cpu().numpy()[0]
            glioma_raw_p = float(_raw_p_unscaled[glioma_idx])
            glioma_flag = (glioma_raw_p > self.GLIOMA_THRESHOLD)

        # PHASE 2 — MC-Dropout passes (dropout ON, BatchNorm frozen):
        # IMPORTANT: Do NOT call model.train() — that would also put BatchNorm into
        # training mode, causing batch-level noise on single-image inference (batch size=1).
        self.classifier.eval()
        for m in self.classifier.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                m.train()

        mc_probs = []
        with torch.no_grad():
            for _ in range(15):
                out = self.classifier(slice_stack, crop_tensor)
                p = torch.softmax(out / self.T, dim=1).cpu()
                mc_probs.append(p)

        # TTA passes (model in eval mode - dropout off)
        self.classifier.eval()
        tta_probs = []
        if self.TTA_ENABLED:
            # Rotational TTA disabled until model is retrained with
            # rotation augmentation — vertical flip and 90deg rotation are out-of-distribution
            # for standardized axial MRI and degrade Glioma recall below the 90% clinical gate.
            views = [
                (slice_stack, crop_tensor),
                (torch.flip(slice_stack, dims=[3]), torch.flip(crop_tensor, dims=[3]))
            ]
        else:
            # When self.TTA_ENABLED is False, run only a single forward pass (views = [None])
            views = [
                (slice_stack, crop_tensor)
            ]

        with torch.no_grad():
            for v_full, v_crop in views:
                out = self.classifier(v_full, v_crop)
                tta_probs.append(torch.softmax(out / self.T, dim=1).cpu())

        # PHASE 3 — Restore clean state:
        self.classifier.eval()

        all_probs = mc_probs + tta_probs
        mean_p = torch.stack(all_probs, dim=0).mean(dim=0).numpy()[0]

        if glioma_idx != -1 and glioma_flag:
            idx = glioma_idx
            confidence_pct = round(float(mean_p[glioma_idx]) * 100, 2)
        else:
            idx = int(np.argmax(mean_p))
            confidence_pct = round(float(np.max(mean_p)) * 100, 2)

        raw_label = self.label_map.get(idx, "unknown")

        display_label = DISPLAY_NAMES.get(
            raw_label.lower(), raw_label.replace("_", " ").title()
        )
        if mode == "binary":
            display_label = (
                "No Tumor Detected"
                if raw_label.lower() == "notumor"
                else "Tumor Detected"
            )

        results["label"]      = display_label
        results["confidence"] = confidence_pct

        is_tumor_class = raw_label.lower().replace("_", "") not in ["notumor"]

        # ── Grad-CAM Explainability ───────────────────────────────────────
        # CRITICAL: Grad-CAM forward+backward MUST run outside torch.no_grad().
        # We call it separately from the MC-Dropout loop above.
        heatmap = None
        heatmap_smooth = None
        try:
            # Grad-CAM pass — no torch.no_grad() wrapper here
            heatmap = self.explainer.generate(slice_stack, crop_tensor, idx)

            cam_h, cam_w = slice_stack.shape[2], slice_stack.shape[3]
            canvas = cv2.resize(raw_image, (cam_w, cam_h))

            # Medically advanced: Use edge-preserving Bilateral Filter to prevent activation bleeding into healthy tissue
            hm_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_smooth = cv2.bilateralFilter(
                hm_u8, d=9, sigmaColor=75, sigmaSpace=75
            ).astype(np.float32) / 255.0

            # Brain mask gating — zero outside intracranial cavity
            bm_resized = cv2.resize(
                brain_mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST
            )
            heatmap_smooth[bm_resized == 0] = 0.0

            # Only show heatmap overlay for positive (tumor) findings.
            # For "No Tumor", still generate the heatmap but do NOT overlay it —
            # return the plain canvas so there's no false-positive visual signal.
            if is_tumor_class:
                results["heatmap_img"] = self._make_heatmap_overlay(
                    canvas, heatmap_smooth, alpha=0.5
                )
            else:
                results["heatmap_img"] = canvas

        except Exception as e:
            print(f"[Engine] Grad-CAM error: {e}")
            cam_h, cam_w = slice_stack.shape[2], slice_stack.shape[3]
            canvas = cv2.resize(raw_image, (cam_w, cam_h))
            cv2.putText(canvas, "HEATMAP FAILED", (10, cam_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 1)
            results["heatmap_img"] = canvas

        # Stage 6: Consensus Arbitration (YOLO Override)
        if yolo_centroid is not None:
            YOLO_OVERRIDE_THRESHOLDS = {
                "glioma":     0.90,
                "meningioma": 0.80,
                "pituitary":  0.75,
                "notumor":    0.85,
            }
            # Look up threshold keyed on the YOLO-predicted class label. Default to 0.90 if not found.
            yolo_thresh = YOLO_OVERRIDE_THRESHOLDS.get(yolo_predicted_class, 0.90)
            if not is_tumor_class and det_conf >= yolo_thresh:
                results["label"] = "Unclassified Finding—Review Required"
                results["confidence"] = det_conf * 100
                results["used_yolo_override"] = True
                is_tumor_class = True  # Enable segmentation/heatmap for this override
            
        # "Requires Review" safety check
        if confidence_pct < (INDETERMINATE_THRESHOLD * 100) and not results["used_yolo_override"]:
            results["label"] = "Indeterminate Finding — Radiologist Review Recommended"
            # Note: we still process segmentation if it was originally tumor class
        
        if results["detection_img"] is None:
            results["detection_img"] = raw_image.copy()

        # ── U-Net Segmentation (tumour-positive cases only) ───────────────
        # DOMAIN-MISMATCH FIX: U-Net was trained on tumor ROI crops (BraTS
        # pipeline crops to mask bounding box before resizing). Feeding the
        # full brain image produces skull-boundary artifacts. We must feed
        # the same distribution: a cropped tumor ROI. We then project the
        # predicted mask back to the original full-image coordinates.
        seg_img = raw_image.copy()
        if self.segmentor and is_tumor_class:
            # ── Step 1: Determine ROI region in original image coordinates ──
            # Priority 1: YOLO bounding box (best — directly localised tumor)
            # Priority 2: Brain mask bounding box (fallback)
            # Priority 3: Full image (last resort, same as old broken behaviour)
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, W_orig, H_orig
            
            if roi_crop_img is not None and roi_crop_img.size > 0:
                # Recover original YOLO box coords from det_results
                if self.detector:
                    try:
                        _det = self.detector.predict(raw_image, conf=0.20, iou=0.4, verbose=False)
                        if len(_det) > 0 and len(_det[0].boxes) > 0:
                            _box = _det[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                            pad = 15
                            crop_x1 = max(0,      _box[0] - pad)
                            crop_y1 = max(0,      _box[1] - pad)
                            crop_x2 = min(W_orig, _box[2] + pad)
                            crop_y2 = min(H_orig, _box[3] + pad)
                    except Exception:
                        pass  # fall through to brain-bbox fallback
            
            # If YOLO didn't give us a good crop, use brain mask bounding box
            if crop_x2 - crop_x1 < 32 or crop_y2 - crop_y1 < 32:
                y_brain, x_brain = np.where(brain_mask > 0)
                if len(y_brain) > 0:
                    pad_b = 20
                    crop_x1 = max(0,      x_brain.min() - pad_b)
                    crop_y1 = max(0,      y_brain.min() - pad_b)
                    crop_x2 = min(W_orig, x_brain.max() + pad_b)
                    crop_y2 = min(H_orig, y_brain.max() + pad_b)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            # ── Step 2: Extract crop, normalise, run U-Net ────────────────
            with torch.no_grad():
                roi_for_unet = raw_image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Resize crop to 256x256 (U-Net input size)
                raw_unet  = cv2.resize(roi_for_unet, (256, 256))
                gray_unet = cv2.cvtColor(raw_unet, cv2.COLOR_RGB2GRAY) \
                            if raw_unet.ndim == 3 else raw_unet

                # Percentile normalisation — matches extract_brats_slices.py
                mask_px = gray_unet > 0
                if np.any(mask_px):
                    pixels  = gray_unet[mask_px]
                    p1, p99 = np.percentile(pixels, 1), np.percentile(pixels, 99)
                    norm_unet = np.clip(
                        (gray_unet - p1) / (p99 - p1 + 1e-8), 0, 1
                    ).astype(np.float32)
                else:
                    norm_unet = np.zeros_like(gray_unet, dtype=np.float32)

                unet_tsr = (torch.from_numpy(norm_unet)
                            .float().unsqueeze(0).unsqueeze(0)
                            .to(self.device))
                logits = self.segmentor(unet_tsr)
                p_map  = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # ── Step 3: Project mask back to original image coordinates ───
            # Resize the 256x256 probability map back to the crop region size
            p_map_crop = cv2.resize(p_map, (crop_w, crop_h))
            crop_mask  = (p_map_crop > 0.5).astype(np.uint8)

            # Place crop mask into a full-image-sized canvas
            mask_np = np.zeros((H_orig, W_orig), dtype=np.uint8)
            mask_np[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask

            # Anatomical gate — intracranial only
            mask_np = cv2.bitwise_and(
                mask_np, (brain_mask > 0).astype(np.uint8)
            )

            # Stage 7: Connected-component cleanup (15% Threshold)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_np, connectivity=8
            )
            if num_labels > 1:
                clean_mask = np.zeros_like(mask_np)
                max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                threshold = 0.15 * max_area

                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= threshold:
                        clean_mask[labels == i] = 1
                
                mask_np = clean_mask

            results["mask_area"] = int(np.sum(mask_np))
            results["icv_pct"]   = round(
                (results["mask_area"] / brain_pixels * 100)
                if brain_pixels > 0 else 0.0, 2
            )

            M = cv2.moments(mask_np)
            unet_centroid = None
            if M["m00"] > 0:
                unet_centroid = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                )
                results["centroid"] = unet_centroid

            # Stage 8: Cross-Modal Spatial Validation (Dynamic Discordance Threshold)
            if unet_centroid and yolo_centroid:
                dist = np.sqrt((unet_centroid[0] - yolo_centroid[0])**2 + 
                               (unet_centroid[1] - yolo_centroid[1])**2)
                # Compute dynamic threshold: max(80, 0.40 * yolo_bbox_diagonal)
                # where yolo_bbox_diagonal is the Euclidean diagonal of the YOLO bounding box
                dynamic_thresh = max(80.0, 0.40 * yolo_bbox_diagonal)
                if dist > dynamic_thresh:
                    results["is_discordant"] = True
                    results["confidence"] = 0.0  # Set confidence to zero as per paper

            # ── Grad-CAM Fallback (when U-Net mask is still too small) ────
            # If U-Net produces a tiny/fragmented mask (<0.5% ICV) — can
            # happen on very small tumors — fall back to Grad-CAM++ hotspot.
            used_gradcam_fallback = False
            
            if results["icv_pct"] < 0.5 and heatmap_smooth is not None:
                hm_resized = cv2.resize(heatmap_smooth, (W_orig, H_orig))
                cam_mask = (hm_resized > 0.6).astype(np.uint8)
                
                cam_area = int(np.sum(cam_mask))
                if cam_area > results["mask_area"]:
                    mask_np = cam_mask
                    results["mask_area"] = cam_area
                    results["icv_pct"]   = round((cam_area / brain_pixels * 100) if brain_pixels > 0 else 0.0, 2)
                    used_gradcam_fallback = True

            contours, _ = cv2.findContours(
                mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Draw contours: Yellow for U-Net, Magenta for Grad-CAM fallback
            contour_color = (255, 0, 255) if used_gradcam_fallback else (255, 255, 0)
            cv2.drawContours(seg_img, contours, -1, contour_color, 2)

            # ── Cross-Modal Bounding Box Fallback ─────────────────────────
            if results.get("roi_crop") is None and results["mask_area"] > 0:
                y_idx, x_idx = np.where(mask_np > 0)
                if len(x_idx) > 0 and len(y_idx) > 0:
                    x1, x2 = x_idx.min(), x_idx.max()
                    y1, y2 = y_idx.min(), y_idx.max()
                    
                    det_canvas = raw_image.copy()
                    box_color = (255, 0, 255) if used_gradcam_fallback else (0, 165, 255)
                    label_text = "GRAD-CAM BOUNDING BOX" if used_gradcam_fallback else "U-NET BOUNDING BOX"
                    
                    cv2.rectangle(det_canvas, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(det_canvas, label_text, (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    results["detection_img"] = det_canvas
                    
                    pad = 15
                    crop_y1_r, crop_y2_r = max(0, y1-pad), min(H_orig, y2+pad)
                    crop_x1_r, crop_x2_r = max(0, x1-pad), min(W_orig, x2+pad)
                    results["roi_crop"] = raw_image[crop_y1_r:crop_y2_r, crop_x1_r:crop_x2_r]

        results["segmentation_img"] = seg_img

        # ── HD ROI Crop (tumour-positive only) ────────────────────────────
        roi_crop = results.get("roi_crop")  # Might already be set by YOLO
        
        if is_tumor_class and roi_crop is None:
            cx, cy   = None, None
            pad_x, pad_y = 64, 64

            # Priority 1: U-Net centroid (pixel-precise)
            if self.segmentor and results.get("centroid"):
                cx, cy = results["centroid"]
                radius = int(np.sqrt(results.get("mask_area", 4000) / np.pi))
                pad_x  = max(64, int(radius * 1.5))
                pad_y  = max(64, int(radius * 1.5))

            # Priority 2: Grad-CAM peak (high-confidence only)
            elif confidence_pct > 70.0 and heatmap is not None:
                cam_h, cam_w = slice_stack.shape[2], slice_stack.shape[3]
                py, px  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                cx      = int(px * (W_orig / cam_w))
                cy      = int(py * (H_orig / cam_h))
                pad_x   = 128
                pad_y   = 128

            if cx is not None and cy is not None:
                y1 = max(0, cy - pad_y)
                y2 = min(H_orig, cy + pad_y)
                x1 = max(0, cx - pad_x)
                x2 = min(W_orig, cx + pad_x)
                roi_crop = raw_image[y1:y2, x1:x2]

        if roi_crop is not None and roi_crop.size > 0:
            roi_crop = cv2.resize(
                roi_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            results["roi_crop"] = roi_crop

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Demo Inference Engine  (runs when model weights are NOT available)
#  Returns realistic-looking outputs so the full web UI can be tested without
#  requiring trained weights. Clearly labelled as DEMO in all outputs.
# ─────────────────────────────────────────────────────────────────────────────

class DemoInferenceEngine:
    """
    Drop-in replacement for ClinicalInferenceEngine when weight files are absent.
    Produces visually rich but clearly-labelled DEMO outputs for UI testing.
    """

    DEMO_CLASSES = ["Glioma", "Meningioma", "No Tumor Detected", "Pituitary Tumor"]
    DEMO_COLORS  = {
        "Glioma":           (255, 80,  80),
        "Meningioma":       (255, 165,  0),
        "No Tumor Detected":(80,  220, 80),
        "Pituitary Tumor":  (80,  140, 255),
    }

    def __init__(self, device="cpu", label_map=None):
        self.device    = device
        self.label_map = label_map or {0: "glioma", 1: "meningioma",
                                        2: "notumor", 3: "pituitary"}
        self.preprocessor = Preprocess()
        print("[Engine] [DEMO MODE] No trained weights found. "
              "Outputs are SIMULATED for UI testing only.")

    def predict(self, slice_stack, raw_image, mode="multi"):
        import random
        H, W = raw_image.shape[:2]

        # Pick a deterministic-ish demo class based on image content
        seed = int(raw_image.mean() * 100) % 4
        demo_label = self.DEMO_CLASSES[seed]
        demo_conf  = round(60.0 + (raw_image.std() % 35), 2)

        if mode == "binary":
            display_label = "No Tumor Detected" if seed == 2 else "Tumor Detected"
        else:
            display_label = demo_label

        is_tumor = display_label not in ["No Tumor Detected"]

        # ── Fake heatmap overlay ──────────────────────────────────────────
        heatmap = np.zeros((H, W), dtype=np.float32)
        if is_tumor:
            cy, cx = H // 3, W // 2
            for y in range(H):
                for x in range(W):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heatmap[y, x] = max(0, 1 - dist / (min(H, W) * 0.35))
        hm_u8 = (heatmap * 255).astype(np.uint8)
        cmap  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cmap  = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        heatmap_img = cv2.addWeighted(raw_image, 0.5, cmap, 0.5, 0) if is_tumor else raw_image.copy()

        # Stamp DEMO watermark
        for img in [heatmap_img]:
            cv2.putText(img, "DEMO MODE — NOT REAL PREDICTION",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # ── Fake detection box ────────────────────────────────────────────
        detection_img = raw_image.copy()
        roi_crop      = None
        centroid      = None
        mask_area     = 0
        icv_pct       = 0.0
        if is_tumor:
            bx1, by1 = W // 3, H // 4
            bx2, by2 = 2 * W // 3, 2 * H // 3
            color = self.DEMO_COLORS.get(demo_label, (255, 255, 0))
            cv2.rectangle(detection_img, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(detection_img, f"DEMO: {demo_label} ({demo_conf:.0f}%)",
                        (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            roi_crop = cv2.resize(raw_image[by1:by2, bx1:bx2], (256, 256))
            centroid = ((bx1 + bx2) // 2, (by1 + by2) // 2)
            mask_area = (bx2 - bx1) * (by2 - by1)
            brain_px  = H * W
            icv_pct   = round(mask_area / brain_px * 100, 2)

        # ── Fake segmentation ─────────────────────────────────────────────
        segmentation_img = raw_image.copy()
        if is_tumor:
            seg_mask = np.zeros((H, W), dtype=np.uint8)
            cy_s, cx_s = H // 3, W // 2
            axes = (W // 8, H // 6)
            cv2.ellipse(seg_mask, (cx_s, cy_s), axes, 0, 0, 360, 255, -1)
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(segmentation_img, contours, -1, (255, 255, 0), 2)

        return {
            "label":             display_label,
            "confidence":        demo_conf,
            "status":            "Demo",
            "message":           "DEMO MODE — Install trained weights for real predictions.",
            "heatmap_img":       heatmap_img,
            "detection_img":     detection_img,
            "segmentation_img":  segmentation_img,
            "roi_crop":          roi_crop,
            "centroid":          centroid,
            "mask_area":         mask_area,
            "icv_pct":           icv_pct,
            "is_discordant":     False,
            "used_yolo_override":False,
        }

