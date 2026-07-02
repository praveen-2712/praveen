"""
dataset.py — Neurologix Pro V3
================================
2.5D MRI Dataset with YOLO-guided crops, class-specific augmentation,
and clinical preprocessing support.
"""

import os
import re
import json
import logging
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Canonical label → integer index mapping
LABEL_TO_IDX = {
    "glioma":     0,
    "meningioma": 1,
    "notumor":    2,
    "pituitary":  3,
}

def precompute_yolo_boxes(
    image_dir: str,
    yolo_checkpoint: str,
    output_json: str = "yolo_boxes.json",
    conf_threshold: float = 0.25,
) -> Dict[str, list]:
    """
    Pre-compute YOLO bounding boxes for all images in a directory before training.
    Run this ONCE before training to avoid GPU contention during data loading.
    Saves boxes to a JSON file for reproducibility.

    Usage:
        boxes = precompute_yolo_boxes("data/classification/Training", "weights/detector_yolo.pt")
        # Returns dict: {filename_stem: [x1, y1, x2, y2]}
    """
    from ultralytics import YOLO
    yolo    = YOLO(yolo_checkpoint)
    boxes   = {}
    missing = 0
    image_paths = []

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))

    logger.info(f"Pre-computing YOLO boxes for {len(image_paths)} images...")
    for path in image_paths:
        image    = cv2.imread(path)
        stem     = Path(path).stem
        results  = yolo(image, verbose=False, conf=conf_threshold)
        det_boxes = results[0].boxes
        if det_boxes is not None and len(det_boxes) > 0:
            best = det_boxes.conf.argmax().item()
            xyxy = det_boxes.xyxy[best].cpu().numpy().tolist()
            boxes[stem] = xyxy
        else:
            missing += 1

    fallback_rate = missing / len(image_paths) if image_paths else 0
    logger.info(f"YOLO boxes computed: {len(boxes)} detections, {missing} fallbacks ({fallback_rate:.1%} fallback rate)")
    with open(output_json, "w") as fp:
        json.dump(boxes, fp, indent=2)
    logger.info(f"Boxes saved to {output_json}")
    return boxes


def _build_glioma_transform(img_size):
    """
    Glioma-specific augmentation pipeline.
    Each transform is radiologically motivated:
    - ElasticTransform: simulates infiltrative tumor margin deformation (hallmark of glioma)
    - RandomGamma: T1 signal intensity variation across MRI protocols and scanner manufacturers
    - GaussianBlur: low-grade glioma (Grade II) has indistinct margins — blur simulates this
    - GridDistortion: simulates scanner-to-scanner geometric distortion
    - CoarseDropout: forces model not to rely on any single focal region (glioma is diffuse)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.ElasticTransform(alpha=120, sigma=6, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(15, 25),
            hole_width_range=(15, 25),
            fill=0, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _build_standard_transform(img_size):
    """Standard augmentation for non-glioma classes."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _natural_sort_key(filename: str):
    """
    Returns a sort key that orders strings by embedded integers numerically.

    Example:
        Alphabetic : Tr-gl_1, Tr-gl_10, Tr-gl_100, Tr-gl_2  (wrong)
        Natural    : Tr-gl_1, Tr-gl_2, Tr-gl_10, Tr-gl_100  (correct)
    """
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class MRI25DDataset(Dataset):
    """
    Neurologix Pro V3 — 2.5D Input Pipeline with YOLO-guided crops.

    Stacks (t-1, t, t+1) consecutive slices into RGB channels for spatial
    context. Handles series boundaries via boundary padding (slice duplication).

    UPGRADE — YOLO-guided branch_local crop:
        Accepts a precomputed `yolo_boxes` dict (stem -> [x1,y1,x2,y2]).
        If a YOLO box is found for an image, the crop fed to branch_local is
        the 15%-padded tumor ROI. This is the most impactful fix for Glioma recall:
        branch_local was previously blind (brain mask = full brain for glioma).

    UPGRADE — Class-specific augmentation:
        Glioma images receive stronger augmentation (ElasticTransform, GridDistortion,
        RandomGamma, GaussianBlur) to simulate diffuse infiltrative margins.
        All other classes receive standard augmentation.

    UPGRADE — single_image_mode:
        When True, loads only the center image and builds a 3-channel tensor
        by applying three different CLAHE clip limits (1.5, 2.0, 2.5) to
        the grayscale image.
    """

    def __init__(self, root_dir, transform=None, preprocess_logic=None, img_size=224,
                 single_image_mode=True, yolo_boxes: Optional[Dict[str, list]] = None,
                 local_pad_ratio: float = 0.15):
        self.root_dir         = root_dir
        self.transform        = transform
        self.preprocess_logic = preprocess_logic
        self.img_size         = img_size
        self.single_image_mode = single_image_mode
        self.yolo_boxes       = yolo_boxes or {}
        self.local_pad_ratio  = local_pad_ratio
        self.samples          = []

        # Pre-build class-specific augmentation transforms
        self.glioma_transform    = _build_glioma_transform(img_size)
        self.standard_transform  = _build_standard_transform(img_size)

        if os.path.isdir(root_dir):
            for class_name in sorted(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue

                # ── FIX: natural sort replaces plain sorted() ──────────────
                files = sorted(
                    [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                    key=_natural_sort_key,   # <-- was: sorted(...) with no key
                )
                # ───────────────────────────────────────────────────────────

                for i, f in enumerate(files):
                    self.samples.append({
                        "path":      os.path.join(class_dir, f),
                        "filename":  f,
                        "folder":    class_dir,
                        "index":     i,
                        "neighbors": files,
                        "label":     class_name.lower(),
                    })

    def __len__(self):
        return len(self.samples)

    def _get_local_crop(self, img_bgr: np.ndarray, filename: str) -> np.ndarray:
        """
        Get tumor ROI crop for branch_local.
        Priority:
            1. YOLO bounding box crop (actual tumor ROI + 15% padding for peritumoral edema)
            2. Brain mask contour crop (fallback — blind for glioma, but better than nothing)
        """
        H, W = img_bgr.shape[:2]
        stem = Path(filename).stem

        if stem in self.yolo_boxes or filename in self.yolo_boxes:
            box = self.yolo_boxes.get(stem) or self.yolo_boxes.get(filename)
            x1, y1, x2, y2 = [int(v) for v in box]
            pad_x = int((x2 - x1) * self.local_pad_ratio)
            pad_y = int((y2 - y1) * self.local_pad_ratio)
            crop = img_bgr[
                max(0, y1 - pad_y): min(H, y2 + pad_y),
                max(0, x1 - pad_x): min(W, x2 + pad_x),
            ]
            if crop.size > 0:
                return cv2.resize(crop, (self.img_size, self.img_size))

        # Fallback: brain mask contour
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            crop = img_bgr[y:y+h, x:x+w]
        else:
            crop = img_bgr
        return cv2.resize(crop, (self.img_size, self.img_size))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        folder = sample["folder"]
        files  = sample["neighbors"]
        i      = sample["index"]
        label_name = sample["label"]
        label_idx  = LABEL_TO_IDX.get(label_name, -1)

        # Load the center image as BGR for crop extraction
        img_path_center = os.path.join(folder, files[i])
        img_bgr = cv2.imread(img_path_center)
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # ── YOLO-guided branch_local crop ──────────────────────────────────────
        stack_crop_bgr = self._get_local_crop(img_bgr, files[i])
        # Convert to RGB (H, W, 3)
        stack_crop = cv2.cvtColor(stack_crop_bgr, cv2.COLOR_BGR2RGB)

        if self.single_image_mode:
            # ── Single-image mode: 3 CLAHE variants as channels ─────────────────
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_size, self.img_size))
            channels = []
            for clip_limit in [1.5, 2.0, 2.5]:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                channels.append(clahe.apply(img))
            stack = np.stack(channels, axis=-1)  # (H, W, 3)
            # Convert to RGB HWC
            stack = stack  # already in HWC format
        else:
            # ── Standard 2.5D mode: (t-1, t, t+1) volumetric stack ───────────
            prev_idx = max(0, i - 1)
            next_idx = min(len(files) - 1, i + 1)

            slices = []
            for nb in [prev_idx, i, next_idx]:
                slc = cv2.imread(os.path.join(folder, files[nb]), cv2.IMREAD_GRAYSCALE)
                if slc is None:
                    slc = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                slices.append(slc)

            if self.preprocess_logic:
                stack = self.preprocess_logic.prepare_stack(slices, img_size=self.img_size)
            else:
                resized = [cv2.resize(s, (self.img_size, self.img_size)) for s in slices]
                stack   = np.stack(resized, axis=-1)

        # ── Class-specific augmentation ────────────────────────────────────────
        # Use the transform passed via constructor (from train script) for the
        # global branch. For the local crop, apply class-specific augmentation.
        if self.transform:
            # Global branch: use the provided transform
            augmented = self.transform(image=stack)
            stack     = augmented['image']
            # Local branch: class-specific augmentation
            if label_idx == LABEL_TO_IDX.get("glioma", 0):
                aug_local = self.glioma_transform(image=stack_crop)
            else:
                aug_local = self.standard_transform(image=stack_crop)
            stack_crop = aug_local['image']
        else:
            stack      = (torch.from_numpy(stack).permute(2, 0, 1).float() / 255.0)
            stack_crop = (torch.from_numpy(stack_crop).permute(2, 0, 1).float() / 255.0)

        return {
            "image":      stack,
            "crop":       stack_crop,
            "label":      label_idx,
            "label_name": label_name,
            "filename":   sample["filename"],
        }
