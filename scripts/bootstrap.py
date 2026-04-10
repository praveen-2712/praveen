import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn as nn
from utils.preprocess import load_and_preprocess_image
from utils.gradcam import GradCAM
import json
from tqdm import tqdm

# Config
DATA_DIR = "data/train"
BOOTSTRAP_DIR = "data/bootstrapped"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/tumor_classifier.pth"
LABEL_MAP_PATH = "models/label_map.json"

# Ensure dirs
os.makedirs(os.path.join(BOOTSTRAP_DIR, "yolo/labels"), exist_ok=True)
os.makedirs(os.path.join(BOOTSTRAP_DIR, "yolo/images"), exist_ok=True)
os.makedirs(os.path.join(BOOTSTRAP_DIR, "unet/masks"), exist_ok=True)
os.makedirs(os.path.join(BOOTSTRAP_DIR, "unet/images"), exist_ok=True)

def generate_labels():
    # Load model
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    model = models.efficientnet_b2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    target_layer = model.features[-1]
    gc = GradCAM(model, target_layer)
    
    classes = os.listdir(DATA_DIR)
    
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir) or cls == 'no_tumor': continue
        
        cls_idx = label_map[cls]
        print(f"Bootstrapping {cls}...")
        
        images = os.listdir(cls_dir)
        for img_name in tqdm(images):
            img_path = os.path.join(cls_dir, img_name)
            try:
                # Load & Preprocess
                image_tensor, pil_display = load_and_preprocess_image(img_path)
                image_tensor = image_tensor.to(DEVICE)
                
                # Grad-CAM peak detection
                heatmap = gc.generate_heatmap(image_tensor, cls_idx)
                
                # Threshold for ROI (Lowered for maximum yield)
                heatmap_u8 = np.uint8(255 * heatmap)
                _, thresh = cv2.threshold(heatmap_u8, 20, 255, cv2.THRESH_BINARY)
                
                # U-Net Mask generation
                mask_path = os.path.join(BOOTSTRAP_DIR, f"unet/masks/{cls}_{img_name}")
                cv2.imwrite(mask_path, thresh)
                
                # U-Net Image copy
                img_save_path = os.path.join(BOOTSTRAP_DIR, f"unet/images/{cls}_{img_name}")
                pil_display.save(img_save_path)
                
                # YOLO BBox generation
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                
                yolo_labels = []
                h_img, w_img = heatmap.shape
                for c in cnts:
                    if cv2.contourArea(c) < 50: continue
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = (x + w/2) / w_img
                    y_center = (y + h/2) / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img
                    yolo_labels.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Fallback: if no labels found but class is tumor, use center ROI
                if not yolo_labels:
                    x_center, y_center = 0.5, 0.5
                    w_norm, h_norm = 0.4, 0.4 # Reasonable center crop
                    yolo_labels.append(f"{cls_idx} {x_center} {y_center} {w_norm} {h_norm}")
                    # Re-generate mask as center box
                    thresh = np.zeros_like(heatmap_u8)
                    cv2.rectangle(thresh, (int(0.3*w_img), int(0.3*h_img)), (int(0.7*w_img), int(0.7*h_img)), 255, -1)

                label_path = os.path.join(BOOTSTRAP_DIR, f"yolo/labels/{cls}_{img_name.replace('.jpg', '.txt')}")
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_labels))
                
                # U-Net Mask generation
                mask_path = os.path.join(BOOTSTRAP_DIR, f"unet/masks/{cls}_{img_name}")
                cv2.imwrite(mask_path, thresh)

                # YOLO & U-Net Image copies
                pil_display.save(os.path.join(BOOTSTRAP_DIR, f"yolo/images/{cls}_{img_name}"))
                pil_display.save(os.path.join(BOOTSTRAP_DIR, f"unet/images/{cls}_{img_name}"))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    generate_labels()
    print("Bootstrapping complete. Multi-architecture data generated.")
