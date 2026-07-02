"""
train_unet.py — Neurologix Pro V3 (Optimized)
===============================================
Trains a U-Net segmentation model on BraTS 2020 extracted FLAIR slices.

Architecture : smp.Unet(encoder_name="resnet50", in_channels=1, classes=1)
Dataset      : BraTS extracted slices (from scripts/extract_brats_slices.py)
Loss         : 0.4 * DiceLoss + 0.3 * BCEWithLogitsLoss + 0.3 * FocalLoss
Metric       : Dice Score (saved on best validation Dice)
Output       : weights/unet_segmentor.pth

Improvements over V2:
  - Albumentations augmentation pipeline (flip, rotate, elastic, noise)
  - Mixed-precision training (AMP) for ~40% memory reduction
  - Focal loss component to handle tumor/background class imbalance
  - Gradient clipping to prevent exploding gradients
  - TrainConfig dataclass — all hyperparameters in one place
  - Proper early stopping with patience counter
  - Reproducibility seeds (numpy / torch / CUDA)
  - Post-epoch sanity warning if val Dice never improves
"""

import os
import glob
import random
import warnings
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # turn on only if input size is fixed


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    img_dir:    str   = "./data/brats_slices/images/"
    mask_dir:   str   = "./data/brats_slices/masks/"
    save_path:  str   = "./weights/unet_segmentor.pth"
    encoder:    str   = "resnet50"

    epochs:     int   = 50
    batch_size: int   = 16
    lr:         float = 3e-4
    val_split:  float = 0.15
    seed:       int   = 42

    # Loss weights  (must sum to 1.0)
    w_dice:     float = 0.40
    w_bce:      float = 0.30
    w_focal:    float = 0.30

    # Scheduler / early stopping
    lr_patience:    int   = 3
    lr_factor:      float = 0.5
    early_stop_patience: int = 10

    # Gradient clipping (0 = disabled)
    grad_clip:  float = 1.0


# ── Augmentation ───────────────────────────────────────────────────────────────

def build_transforms(split: str) -> A.Compose:
    """
    Training augmentations are chosen specifically for brain MRI:
      - Flips & rotation: tumors appear in varied orientations.
      - ElasticTransform: simulates natural brain shape deformation.
      - Brightness/Contrast: normalises scanner-specific intensity drift.
      - GaussNoise: regularises against acquisition noise.
    Validation receives only the normalisation step so metrics are comparable.
    """
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        ])
    return A.Compose([])   # val: identity (normalisation happens in __getitem__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class BraTSSliceDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, split: str = "train",
                 val_split: float = 0.15, seed: int = 42):

        img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: {len(img_paths)} vs {len(mask_paths)}. "
                "Run extract_brats_slices.py to re-generate paired slices."
            )

        rng = random.Random(seed)
        combined = list(zip(img_paths, mask_paths))
        rng.shuffle(combined)

        n_val = int(len(combined) * val_split)
        self.data      = combined[n_val:] if split == "train" else combined[:n_val]
        self.transform = build_transforms(split)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.data[idx]

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ── COARSE-TO-FINE PIPELINE ──────────────────────────────────────────
        # Simulate YOLO bounding box by finding the mask contour and padding it.
        # This forces the U-Net to learn high-resolution boundaries inside the ROI.
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Simulated YOLO margin (random during training, fixed during val)
            pad_x = int(w * random.uniform(0.1, 0.4)) if hasattr(self, 'transform') and len(self.transform.transforms) > 0 else int(w * 0.2)
            pad_y = int(h * random.uniform(0.1, 0.4)) if hasattr(self, 'transform') and len(self.transform.transforms) > 0 else int(h * 0.2)
            
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
            
            img  = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
            
        # Resize to fixed input size for U-Net
        img  = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        # Normalise to [0, 1]
        img  = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        # Albumentations expects HxW for grayscale; add a channel dim after
        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented["image"], augmented["mask"]

        img_tensor  = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor


# ── Loss ───────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Dice + BCE + Focal combination.

    Dice     → optimises region overlap directly; insensitive to class imbalance.
    BCE      → provides dense pixel-level gradient signal.
    Focal    → down-weights easy negatives (background pixels) so the model
               concentrates learning on hard-to-detect small tumour regions.
    """
    def __init__(self, w_dice: float = 0.4, w_bce: float = 0.3, w_focal: float = 0.3):
        super().__init__()
        assert abs(w_dice + w_bce + w_focal - 1.0) < 1e-6, "Loss weights must sum to 1."
        self.w_dice  = w_dice
        self.w_bce   = w_bce
        self.w_focal = w_focal
        self.dice  = smp.losses.DiceLoss(mode="binary", smooth=1e-6)
        self.bce   = nn.BCEWithLogitsLoss()
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.w_dice  * self.dice(logits, targets)
            + self.w_bce   * self.bce(logits, targets)
            + self.w_focal * self.focal(logits, targets)
        )


# ── Metrics ────────────────────────────────────────────────────────────────────

def dice_score(pred_logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > threshold).float()
        intersection = (pred * targets).sum(dim=(1, 2, 3))
        union        = pred.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice         = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, resume: bool = False):
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[UNet] Device: {device} | AMP: {use_amp}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[UNet] Enabled torch.backends.cudnn.benchmark for maximum convolution efficiency.")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = BraTSSliceDataset(cfg.img_dir, cfg.mask_dir, "train", cfg.val_split, cfg.seed)
    val_ds   = BraTSSliceDataset(cfg.img_dir, cfg.mask_dir, "val",   cfg.val_split, cfg.seed)

    num_workers = 2 if use_amp else 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_amp,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_amp,
                              persistent_workers=(num_workers > 0))

    print(f"[UNet] Train: {len(train_ds)} slices | Val: {len(val_ds)} slices")

    # ── Model ─────────────────────────────────────────────────────────────────
    # activation=None: raw logits are passed to CombinedLoss (which handles
    # sigmoid internally); dice_score applies sigmoid manually. This avoids
    # double-sigmoid and keeps numerical precision with AMP.
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    criterion = CombinedLoss(cfg.w_dice, cfg.w_bce, cfg.w_focal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.lr_patience, factor=cfg.lr_factor
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    last_ckpt_path = os.path.join(os.path.dirname(cfg.save_path), "unet_last_checkpoint.pth")

    best_dice     = 0.0
    no_improve    = 0          # early-stopping counter
    start_epoch   = 1

    if resume and os.path.exists(last_ckpt_path):
        print(f"[UNet] Resuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        no_improve = ckpt["no_improve"]
    elif resume:
        print(f"[UNet] Resume requested but {last_ckpt_path} not found. Starting fresh.")

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs} [Train]", leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

            with autocast('cuda', enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss / (pbar.n + 1):.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_dice_sum = 0.0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch:03d}/{cfg.epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for imgs, masks in vbar:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    logits = model(imgs)
                d = dice_score(logits, masks)
                val_dice_sum += d
                vbar.set_postfix(dice=f"{val_dice_sum / (vbar.n + 1):.4f}")

        avg_val_dice = val_dice_sum / len(val_loader)
        scheduler.step(avg_val_dice)

        # ── Checkpoint & early stopping ────────────────────────────────────────
        marker = ""
        if avg_val_dice > best_dice:
            best_dice  = avg_val_dice
            no_improve = 0
            marker     = " ← BEST"
            torch.save(model.state_dict(), cfg.save_path)
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | "
            f"LR: {current_lr:.2e}{marker}"
        )

        # Save last checkpoint for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
            "no_improve": no_improve,
        }, last_ckpt_path)

        if no_improve >= cfg.early_stop_patience:
            print(f"[UNet] Early stopping triggered after {cfg.early_stop_patience} epochs without improvement.")
            break

    # ── Sanity check ───────────────────────────────────────────────────────────
    if best_dice < 0.1:
        print(
            "[UNet] WARNING: Best Val Dice < 0.1. This usually means mask paths are "
            "misaligned with image paths or masks are all-zero. "
            "Re-run extract_brats_slices.py and verify pairing."
        )

    print(f"\n[UNet] Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"[UNet] Model saved to {cfg.save_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net segmentor")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.epochs = args.epochs

    if not os.path.exists(cfg.img_dir) or not os.path.exists(cfg.mask_dir):
        print(
            f"[ERROR] Data directories not found.\n"
            f"  Expected:\n    {cfg.img_dir}\n    {cfg.mask_dir}\n"
            f"  Run scripts/extract_brats_slices.py first."
        )
    else:
        train(cfg, resume=args.resume)
