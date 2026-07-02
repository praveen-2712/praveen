import os
import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

def normalize_flair(slice_2d):
    """Normalize FLAIR slice to [0, 255] uint8."""
    # Ignore absolute zeros
    mask = slice_2d > 0
    if not np.any(mask):
        return np.zeros_like(slice_2d, dtype=np.uint8)
        
    pixels = slice_2d[mask]
    p_min, p_max = np.percentile(pixels, 1), np.percentile(pixels, 99)
    
    norm = np.clip((slice_2d - p_min) / (p_max - p_min + 1e-8), 0, 1)
    return (norm * 255).astype(np.uint8)

def extract_slices(brats_dir, out_img_dir, out_mask_dir, target_size=(256, 256)):
    """Extract non-empty axial slices from BraTS 2020 NIfTI volumes."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    # BraTS directories typically contain subfolders for each patient
    # Search for all _flair.nii or _flair.nii.gz files
    flair_files = glob.glob(os.path.join(brats_dir, "**", "*_flair.nii*"), recursive=True)
    
    if not flair_files:
        print(f"[ERROR] No FLAIR volumes found in {brats_dir}")
        return
        
    print(f"Found {len(flair_files)} FLAIR volumes.")
    total_retained = 0
    
    for flair_path in tqdm(flair_files, desc="Extracting BraTS Slices"):
        # The segmentation mask is typically named *_seg.nii.gz corresponding to *_flair.nii.gz
        seg_path = flair_path.replace("_flair.", "_seg.")
        if not os.path.exists(seg_path):
            print(f"[WARN] Segmentation mask not found for {flair_path}")
            continue
            
        subject_id = os.path.basename(flair_path).split('_flair')[0]
        
        # Load NIfTI volumes
        try:
            flair_vol = nib.load(flair_path).get_fdata()
            seg_vol = nib.load(seg_path).get_fdata()
        except Exception as e:
            print(f"[ERROR] Failed to read {subject_id}: {e}")
            continue
            
        # BraTS is usually shape (240, 240, 155), axial slices are along axis 2
        num_slices = flair_vol.shape[2]
        
        for z in range(num_slices):
            mask_slice = seg_vol[:, :, z]
            
            # Collapse multi-class mask into binary tumor mask
            # BraTS labels: 1=necrotic, 2=edema, 4=enhancing (collapse all to 1)
            binary_mask = (mask_slice > 0).astype(np.uint8) * 255
            
            # Skip empty masks
            if not np.any(binary_mask):
                continue
                
            flair_slice = flair_vol[:, :, z]
            
            # Normalize image to 8-bit
            flair_img = normalize_flair(flair_slice)
            
            # Resize
            flair_resized = cv2.resize(flair_img, target_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Save
            img_filename = os.path.join(out_img_dir, f"{subject_id}_slice_{z:03d}.png")
            mask_filename = os.path.join(out_mask_dir, f"{subject_id}_slice_{z:03d}.png")
            
            cv2.imwrite(img_filename, flair_resized)
            cv2.imwrite(mask_filename, mask_resized)
            
            total_retained += 1
            
    print("-" * 50)
    print(f"Extraction complete! Total retained slices: {total_retained}")
    print("-" * 50)

if __name__ == "__main__":
    BRATS_DIR = "./data/brats/"
    OUT_IMG_DIR = "./data/brats_slices/images/"
    OUT_MASK_DIR = "./data/brats_slices/masks/"
    
    print("Starting BraTS slice extraction...")
    extract_slices(BRATS_DIR, OUT_IMG_DIR, OUT_MASK_DIR)
