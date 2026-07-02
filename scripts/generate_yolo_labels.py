import os
import cv2
import glob
import shutil
import random
import numpy as np
from tqdm import tqdm

# NOTE: After running this script, the YOLO model MUST be retrained for the
# mAP metric to be valid on held-out data. The previous dataset.yaml pointed
# val to images/train, meaning the reported 0.9222 mAP@0.5 was measured on
# training data and is NOT a valid generalisation metric.

def create_yolo_dataset(brats_mask_dir, brats_img_dir, class_notumor_dir, yolo_base_dir,
                        val_split=0.15, seed=42):
    """
    Generate YOLOv8 labels strictly from BraTS ground-truth segmentation masks,
    and mix in 'notumor' images from the classification dataset.

    FIX 2: Creates a proper 85/15 train/val split for tumor images so that
    the YOLO mAP metric is measured on genuinely held-out data.
    NotumOf images are kept in train only (YOLO val measures detection recall,
    not classification; negative samples belong in train for background diversity).
    """
    # Create all four split directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_base_dir, "labels", split), exist_ok=True)

    print("[YOLO Gen] Processing BraTS slices for Tumor class (85/15 train/val split)...")
    brats_masks = sorted(glob.glob(os.path.join(brats_mask_dir, "*.png")))

    # Collect valid tumor samples first
    valid_tumor_samples = []
    for mask_path in tqdm(brats_masks, desc="Scanning masks"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) == 0:
            continue  # Empty mask, skip

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        H, W = mask.shape
        cx = ((x_min + x_max) / 2.0) / W
        cy = ((y_min + y_max) / 2.0) / H
        bw = (x_max - x_min) / W
        bh = (y_max - y_min) / H

        if cx < 0 or cx > 1 or cy < 0 or cy > 1:
            print(f"[YOLO Gen] Out of bounds box on {mask_path}, skipping.")
            continue

        filename = os.path.basename(mask_path)
        img_src = os.path.join(brats_img_dir, filename)
        if not os.path.exists(img_src):
            continue

        yolo_str = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        valid_tumor_samples.append((img_src, filename, yolo_str))

    # Shuffle with fixed seed for reproducibility, then split 85/15
    rng = random.Random(seed)
    rng.shuffle(valid_tumor_samples)
    n_val = max(1, int(len(valid_tumor_samples) * val_split))
    val_samples   = valid_tumor_samples[:n_val]
    train_samples = valid_tumor_samples[n_val:]

    tumor_train_count = 0
    for img_src, filename, yolo_str in tqdm(train_samples, desc="Copying tumor train"):
        shutil.copy2(img_src, os.path.join(yolo_base_dir, "images", "train", filename))
        lbl_filename = filename.replace(".png", ".txt")
        with open(os.path.join(yolo_base_dir, "labels", "train", lbl_filename), "w") as f:
            f.write(yolo_str)
        tumor_train_count += 1

    tumor_val_count = 0
    for img_src, filename, yolo_str in tqdm(val_samples, desc="Copying tumor val"):
        shutil.copy2(img_src, os.path.join(yolo_base_dir, "images", "val", filename))
        lbl_filename = filename.replace(".png", ".txt")
        with open(os.path.join(yolo_base_dir, "labels", "val", lbl_filename), "w") as f:
            f.write(yolo_str)
        tumor_val_count += 1

    # NotumOf images go to train only — negative samples provide background
    # diversity for detector training but don't need val representation since
    # YOLO val measures detection recall, not classification accuracy.
    print("[YOLO Gen] Processing Classification images for NoTumor class (train only)...")
    notumor_images = (glob.glob(os.path.join(class_notumor_dir, "*.jpg")) +
                      glob.glob(os.path.join(class_notumor_dir, "*.png")))

    notumor_count = 0
    for img_path in tqdm(notumor_images, desc="Copying notumor train"):
        filename = "notumor_" + os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(yolo_base_dir, "images", "train", filename))
        lbl_filename = filename.rsplit(".", 1)[0] + ".txt"
        with open(os.path.join(yolo_base_dir, "labels", "train", lbl_filename), "w") as f:
            pass  # empty label = negative sample
        notumor_count += 1

    print(f"\n[YOLO Gen] Split summary:")
    print(f"  Tumor  train : {tumor_train_count}")
    print(f"  Tumor  val   : {tumor_val_count}  (held-out for mAP validation)")
    print(f"  NoTumor train: {notumor_count}")
    print(f"  Total  train : {tumor_train_count + notumor_count}")

    # Generate dataset.yaml — FIX: val now correctly points to images/val
    yaml_path = os.path.join(yolo_base_dir, "dataset.yaml")
    abs_path = os.path.abspath(yolo_base_dir).replace('\\', '/')
    yaml_content = (
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"# NOTE: val split is a genuine held-out 15% of BraTS tumor images\n"
        f"# (seed=42). Re-run this script and retrain YOLO if the dataset changes.\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['tumor']\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[YOLO Gen] Wrote {yaml_path}")
    print("[YOLO Gen] IMPORTANT: YOLO model must be retrained for mAP metrics to be valid.")


if __name__ == "__main__":
    create_yolo_dataset(
        brats_mask_dir="./data/brats_slices/masks/",
        brats_img_dir="./data/brats_slices/images/",
        class_notumor_dir="./data/classification/Training/notumor/",
        yolo_base_dir="./data/yolo/"
    )
