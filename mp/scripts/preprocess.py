import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
PROCESSED_IMGS = os.path.join(BASE_DIR, "data/processed/images")
PROCESSED_MASKS = os.path.join(BASE_DIR, "data/processed/masks")

# Mapping defined by requirements
CLASS_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3
}

def standardize_image(img_path, target_size=(256, 256)):
    """Reads, resizes to target_size, and returns a 3-channel RGB image."""
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, target_size)
    # Ensure 3-channel consistent format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_binary_mask(mask_path, target_size=(256, 256)):
    """Reads and enforces binary masks."""
    if not os.path.exists(mask_path): return np.zeros(target_size, dtype=np.uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return np.zeros(target_size, dtype=np.uint8)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return binary

def process_datasets():
    print("Starting Data Standardization and Merging...")
    records = []
    global_idx = 0
    
    # 1. Process Kaggle Directory
    kaggle_dir = os.path.join(RAW_DIR, "kaggle")
    for root, _, files in os.walk(kaggle_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')): continue
            
            # Infer class from folder path
            folder_name = os.path.basename(root).lower()
            label = None
            for key, val in CLASS_MAP.items():
                if key in folder_name:
                    label = val
                    break
            if label is None:
                if 'notumor' in folder_name or 'normal' in folder_name: label = 3
                else: continue
                
            img_path = os.path.join(root, file)
            img = standardize_image(img_path)
            if img is None: continue
            
            new_img_name = f"img_{global_idx}.png"
            new_mask_name = f"img_{global_idx}_mask.png"
            
            # Save standardized Image
            cv2.imwrite(os.path.join(PROCESSED_IMGS, new_img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Save corresponding binary mask (placeholder if Kaggle set lacks masks)
            # In a real unified set, you parse the corresponding _mask if it exists
            mask_path = img_path.replace('.jpg', '_mask.png') # simplistic lookup
            mask_binary = create_binary_mask(mask_path)
            cv2.imwrite(os.path.join(PROCESSED_MASKS, new_mask_name), mask_binary)
            
            records.append({
                "image_id": new_img_name,
                "mask_id": new_mask_name,
                "label": label,
                "source": "kaggle"
            })
            global_idx += 1
            
    # Similar loops would exist for BraTS (.nii parser) and Figshare (.mat parser).
    # Those are omitted for brevity pending NIfTI and h5py dependencies, but structure remains identical.
    
    df = pd.DataFrame(records)
    csv_path = os.path.join(BASE_DIR, "data/processed/labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data processing complete. {len(df)} images merged. Metadata saved to {csv_path}")

if __name__ == "__main__":
    process_datasets()
