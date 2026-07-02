"""
generate_pseudo_labels.py — Neurologix Pro V3
==============================================
Uses the trained Classifier and U-Net segmentor to generate bounding boxes
for the Kaggle Brain Tumor MRI dataset (Training/Glioma, Meningioma, Pituitary).

This enables YOLO to be trained on the 'normal' MRI domain even though that
dataset lacks ground-truth bounding box annotations.
"""

import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import timm

# ── Configuration ──────────────────────────────────────────────────────────────
KAGGE_TRAIN_DIR = "./data/classification/Training"
OUTPUT_DIR      = "./data/yolo_pseudo_labels"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights paths
CLASSIFIER_PATH = "./weights/tumor_classifier.pth"
UNET_PATH       = "./weights/unet_segmentor.pth"

# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models():
    print(f"[Pseudo] Loading models on {DEVICE}...")
    
    # 1. U-Net Segmentor (The primary source of box coordinates)
    unet = smp.Unet(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=1, classes=1).to(DEVICE)
    if os.path.exists(UNET_PATH):
        unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE, weights_only=False))
    unet.eval()
    
    # 2. Classifier (Used to confirm the presence of a tumor)
    # Note: We assume the image already has a tumor if it's in the Glioma/Meningioma/Pituitary folders
    return unet

# ── Processing ────────────────────────────────────────────────────────────────

def get_yolo_box(mask_01):
    """Convert binary mask to YOLO (cx, cy, w, h) format."""
    y_idx, x_idx = np.where(mask_01 > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return None
    
    h_img, w_img = mask_01.shape
    x1, x2 = x_idx.min(), x_idx.max()
    y1, y2 = y_idx.min(), y_idx.max()
    
    cx = ((x1 + x2) / 2.0) / w_img
    cy = ((y1 + y2) / 2.0) / h_img
    bw = (x2 - x1) / w_img
    bh = (y2 - y1) / h_img
    
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

def normalize_slice(gray):
    """Percentile normalization matching U-Net training."""
    mask_px = gray > 0
    if not np.any(mask_px):
        return np.zeros_like(gray, dtype=np.float32)
    pixels = gray[mask_px]
    p1, p99 = np.percentile(pixels, 1), np.percentile(pixels, 99)
    norm = np.clip((gray - p1) / (p99 - p1 + 1e-8), 0, 1)
    return norm.astype(np.float32)

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet = load_models()
    
    # Classes to process (excluding 'notumor')
    tumor_classes = ["glioma", "meningioma", "pituitary"]
    
    total_generated = 0
    
    for cls in tumor_classes:
        cls_dir = os.path.join(KAGGE_TRAIN_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"[WARN] Directory {cls_dir} not found. Skipping.")
            continue
            
        img_paths = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        print(f"[Pseudo] Processing {len(img_paths)} images for class: {cls}")
        
        for img_path in tqdm(img_paths, desc=f"Class {cls}"):
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue
            
            h_orig, w_orig = raw_img.shape[:2]
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            
            # 1. Normalize and prepare for U-Net
            norm = normalize_slice(gray)
            unet_input = cv2.resize(norm, (256, 256))
            tsr = torch.from_numpy(unet_input).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # 2. Predict mask
            with torch.no_grad():
                logits = unet(tsr)
                p_map = torch.sigmoid(logits).cpu().numpy()[0, 0]
            
            mask_01 = (cv2.resize(p_map, (w_orig, h_orig)) > 0.5).astype(np.uint8)
            
            # 3. Convert to YOLO box
            yolo_str = get_yolo_box(mask_01)
            
            if yolo_str:
                filename = os.path.basename(img_path)
                lbl_name = filename.rsplit(".", 1)[0] + ".txt"
                with open(os.path.join(OUTPUT_DIR, lbl_name), "w") as f:
                    f.write(yolo_str)
                total_generated += 1
                
    print(f"\n[Pseudo] Done! Generated {total_generated} labels in {OUTPUT_DIR}")
    print("[Pseudo] Next Step: Update scripts/generate_yolo_labels.py to include these labels.")

if __name__ == "__main__":
    generate()
