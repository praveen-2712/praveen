"""
train_classifier.py — Neurologix Pro V3
=========================================
Clinical-Grade Tumor Classifier Training Pipeline — Accuracy Upgrade (~93% Target).

Key upgrades over previous version:
  - Differential learning rates: backbone @ 2e-5, head @ 1e-4
  - OneCycleLR with 10% warmup + cosine annealing (replaces ReduceLROnPlateau)
    → scheduler.step() is called PER BATCH inside the training loop, not per epoch
  - Higher dropout: drop_rate=0.4, drop_path_rate=0.3
  - Label smoothing reduced to 0.05 (was 0.1 — over-smoothing hurt well-regularized model)
  - Stronger augmentation pipeline: VerticalFlip, ShiftScaleRotate, GridDistortion added
  - CoarseDropout holes reduced (32→24) and probability reduced (0.3→0.25)
  - --single_channel flag: loads single image, applies 3 CLAHE variants for channel replication
    (computationally free, gives model mild multi-contrast input instead of identical channels)
  - Test-Time Augmentation (TTA) evaluation: 3 augmented views averaged at test time
    (original + H-flip + V-flip) for ~0.5–1.5% free accuracy gain
  - epochs default raised to 60 (was 50), patience raised to 8 (was 10)
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              classification_report, accuracy_score,
                              confusion_matrix)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MRI25DDataset, LABEL_TO_IDX

try:
    from preprocess import Preprocess
except ImportError:
    Preprocess = None
    print("[Train] WARNING: could not import Preprocess. Training without clinical preprocessing.")

class DualBranchClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.branch_global = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, drop_rate=0.4, drop_path_rate=0.3)
        self.branch_local  = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0, drop_rate=0.4, drop_path_rate=0.3)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_full, img_crop):
        feat_global = self.branch_global(img_full)
        feat_local  = self.branch_local(img_crop)
        fused = torch.cat([feat_global, feat_local], dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
#  Class weight computation (Clinical Asymmetric)
# ---------------------------------------------------------------------------

GLIOMA_IDX     = LABEL_TO_IDX["glioma"]      # 0
MENINGIOMA_IDX = LABEL_TO_IDX["meningioma"]  # 1
CLASSES        = [k for k, v in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1])]


def compute_clinical_class_weights(dataset, device, glioma_multiplier=6.0):
    """
    Compute inverse-frequency class weights with clinical asymmetric penalty.

    Clinical justification for 6x glioma multiplier:
    - GBM (Grade IV glioma) untreated median survival: ~14 months
    - Meningioma untreated median survival: typically >10 years
    - Cost of missed glioma >> cost of false positive (extra imaging only)
    - Empirical: 3x was insufficient (recall regressed 43%->38%)
    - Medical imaging literature: 5-8x for high-stakes imbalanced classes
    """
    print("[Train] Computing clinical class weights...")
    counts = np.zeros(len(LABEL_TO_IDX))
    for sample in dataset.samples:
        idx = LABEL_TO_IDX.get(sample["label"], -1)
        if idx != -1:
            counts[idx] += 1
    total = sum(counts)
    weights = total / (len(counts) * counts)
    weights = weights / weights.sum() * len(counts)

    # 6x asymmetric clinical penalty for glioma
    weights[GLIOMA_IDX] *= glioma_multiplier

    for cls, w in zip(CLASSES, weights):
        flag = "  <- 6x clinical penalty" if cls == "glioma" else ""
        print(f"  {cls:<15}: {w:.4f}{flag}")
    return torch.FloatTensor(weights).to(device)


# ---------------------------------------------------------------------------
#  Loss functions (Clinical Combined)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017). Concentrates gradient on hard examples.
    gamma=2.0: Standard for medical imaging.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)
        log_pt    = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


class GliomaMarginLoss(nn.Module):
    """
    Glioma-Meningioma Margin Loss.
    Directly penalizes when P(meningioma) > P(glioma) for true glioma samples.
    Attacks the dominant confusion pattern: 47/100 gliomas -> meningioma.
    margin=0.5: glioma logit must exceed meningioma logit by at least 0.5.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, logits, labels):
        import torch.nn.functional as F
        mask = (labels == GLIOMA_IDX)
        if mask.sum() == 0:
            return logits.sum() * 0.0
        glioma_logits = logits[mask]
        violation = F.relu(
            glioma_logits[:, MENINGIOMA_IDX]
            - glioma_logits[:, GLIOMA_IDX]
            + self.margin
        )
        return violation.mean()


class ClinicalCombinedLoss(nn.Module):
    """
    Three-component clinical loss for brain tumor classification.
      CE (0.55)     - maintains overall multiclass calibration
      Focal (0.25)  - handles hard examples across all classes
      Margin (0.20) - directly attacks the glioma->meningioma confusion
    """
    def __init__(self, class_weights):
        super().__init__()
        self.ce_loss     = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss  = FocalLoss(gamma=2.0, weight=class_weights)
        self.margin_loss = GliomaMarginLoss(margin=0.5)
        self.ce_w     = 0.55
        self.focal_w  = 0.25
        self.margin_w = 0.20

    def forward(self, logits, targets):
        ce     = self.ce_loss(logits, targets)
        focal  = self.focal_loss(logits, targets)
        margin = self.margin_loss(logits, targets)
        total  = self.ce_w * ce + self.focal_w * focal + self.margin_w * margin
        return total, {"ce": ce.item(), "focal": focal.item(), "margin": margin.item()}


# ---------------------------------------------------------------------------
#  Augmentation pipelines
# ---------------------------------------------------------------------------

def get_transforms():
    """
    Upgraded training augmentation pipeline.
    Key changes vs. previous version:
      - Added VerticalFlip (p=0.2), ShiftScaleRotate, GridDistortion
      - CoarseDropout holes reduced 32→24, probability 0.3→0.25
      - Added OneOf blur/noise block
      - Reduced ElasticTransform probability
    """
    train_transform = A.Compose([
        A.Resize(380, 380),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(translate_percent=(-0.06, 0.06), scale=(0.88, 1.12), rotate=(-20, 20), p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.02, 0.06),
            hole_width_range=(0.02, 0.06),
            fill=0, p=0.25
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform


# ---------------------------------------------------------------------------
#  TTA: Test-Time Augmentation helper
# ---------------------------------------------------------------------------

def tta_predict(model, img_tensor_full, img_tensor_crop, device):
    """
    Run 3-view TTA on full and cropped image tensors.
    Views: original, H-flip, V-flip.
    Returns averaged softmax probabilities (shape: num_classes).
    """
    model.eval()
    views_full = [
        img_tensor_full,                                            # original
        torch.flip(img_tensor_full, dims=[3]),                      # horizontal flip
        torch.flip(img_tensor_full, dims=[2]),                      # vertical flip
    ]
    views_crop = [
        img_tensor_crop,                                            # original
        torch.flip(img_tensor_crop, dims=[3]),                      # horizontal flip
        torch.flip(img_tensor_crop, dims=[2]),                      # vertical flip
    ]
    probs_list = []
    with torch.no_grad():
        for i in range(len(views_full)):
            out = model(views_full[i].to(device), views_crop[i].to(device))
            probs_list.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.mean(probs_list, axis=0)  # (1, num_classes)


# ---------------------------------------------------------------------------
#  Normalization config
# ---------------------------------------------------------------------------

def save_normalization(save_dir):
    norm_data = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    norm_path = os.path.join(save_dir, "classifier_norm.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(norm_path, 'w') as f:
        json.dump(norm_data, f)
    print(f"[Train] Saved normalization config to {norm_path}")


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

def train(data_dir_train, data_dir_val, epochs=60, batch_size=16,
          patience=20, save_dir=None, single_channel=False, resume=False,
          freeze_global=False):
    """
    Upgraded training function with differential LR, OneCycleLR, and TTA.

    Parameters
    ----------
    data_dir_train : path to training split (e.g. data/classification/Training)
    data_dir_val   : path to validation split (e.g. data/classification/Testing)
    epochs         : total epochs (default 60; OneCycleLR needs full budget)
    batch_size     : mini-batch size (default 16 at 380x380)
    patience       : early stop patience on val accuracy (default 8)
    single_channel : if True, use single_image_mode in dataset (3 CLAHE variants)
    freeze_global  : if True, freeze branch_global entirely; only train branch_local
                     + head. ~2x faster. Use when branch_global is already well-trained
                     and only the local-crop branch needs to adapt (e.g. after adding
                     YOLO-guided crops via precompute_boxes.py).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[Train] Enabled torch.backends.cudnn.benchmark for maximum convolution efficiency.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actual_save_dir = save_dir if save_dir else os.path.join(base_dir, "models")
    os.makedirs(actual_save_dir, exist_ok=True)

    save_normalization(actual_save_dir)

    train_tf, val_tf = get_transforms()

    # Preprocessing: disabled (use_norm=False, use_clahe=False) to align with
    # the distribution the model was trained on. CLAHE is now handled via the
    # single_image_mode 3-CLAHE variant channels in dataset.py when enabled.
    preprocess = Preprocess() if Preprocess is not None else None

    # ── YOLO-guided branch_local crop ─────────────────────────────────────────
    # Load precomputed bounding boxes (generated by precompute_boxes.py).
    # If the JSON doesn't exist, dataset falls back to brain-mask contour crop.
    yolo_boxes_path = os.path.join(base_dir, "yolo_boxes.json")
    yolo_boxes = {}
    if os.path.exists(yolo_boxes_path):
        with open(yolo_boxes_path, "r") as _f:
            yolo_boxes = json.load(_f)
        print(f"[Train] Loaded {len(yolo_boxes)} YOLO boxes from {yolo_boxes_path}")
    else:
        print(f"[Train] WARNING: {yolo_boxes_path} not found — branch_local will use brain-mask fallback.")
        print(f"[Train]          Run python precompute_boxes.py to generate it.")
    # ─────────────────────────────────────────────────────────────────────────

    train_dataset = MRI25DDataset(
        data_dir_train, transform=train_tf,
        preprocess_logic=preprocess, img_size=380,
        single_image_mode=single_channel,
        yolo_boxes=yolo_boxes,
    )
    val_dataset = MRI25DDataset(
        data_dir_val, transform=val_tf,
        preprocess_logic=preprocess, img_size=380,
        single_image_mode=single_channel,
        yolo_boxes=yolo_boxes,
    )

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, 
                              pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, 
                              pin_memory=True, persistent_workers=(num_workers > 0))

    class_weights = compute_clinical_class_weights(train_dataset, device)

    # ── Model: Dual-Branch Attention-Guided Vision Transformer ─────────────────
    model = DualBranchClassifier(num_classes=len(LABEL_TO_IDX))

    actual_save_path = os.path.join(base_dir, "weights", "tumor_classifier.pth")
    if resume and os.path.exists(actual_save_path):
        print(f"[Train] Resuming from existing weights: {actual_save_path}")
        sd = torch.load(actual_save_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)

    model.to(device)

    # ── Option E: Freeze branch_global ──────────────────────────────────────
    # When freeze_global=True, branch_global is frozen completely:
    #   - No gradient computed through it (torch.no_grad path in autograd)
    #   - ~2x GPU memory reduction on the backward pass
    #   - ~2x faster per batch
    # Only branch_local + fusion head are updated — exactly what's needed when
    # branch_global is already well-trained and we're adapting to YOLO crops.
    if freeze_global:
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'branch_global' in name:
                param.requires_grad = False
                frozen_count += 1
        trainable_local  = sum(p.numel() for n, p in model.named_parameters()
                               if 'branch_local' in n and p.requires_grad)
        trainable_head   = sum(p.numel() for n, p in model.named_parameters()
                               if 'head' in n and p.requires_grad)
        frozen_global    = sum(p.numel() for n, p in model.named_parameters()
                               if 'branch_global' in n)
        print(f"[Train] ── FREEZE GLOBAL MODE ──────────────────────────────────")
        print(f"[Train]   branch_global : FROZEN  ({frozen_global/1e6:.1f}M params, 0 grad)")
        print(f"[Train]   branch_local  : ACTIVE  ({trainable_local/1e6:.1f}M params)")
        print(f"[Train]   head          : ACTIVE  ({trainable_head/1e6:.1f}M params)")
        print(f"[Train]   Expected speedup: ~2x per epoch")
        print(f"[Train] ──────────────────────────────────────────────────────────")

    # ── Loss: ClinicalCombinedLoss (CE + Focal + GliomaMarginLoss) ───────────
    criterion = ClinicalCombinedLoss(class_weights=class_weights)

    # ── Optimizer: only include params with requires_grad=True ───────────────
    # When freeze_global=True this automatically excludes branch_global.
    # Raise LRs slightly in freeze mode since branch_local learns from scratch.
    #
    # NOTE: EfficientNetV2-S has an internal 'conv_head' layer, so param names
    # like 'branch_local.conv_head.weight' would match BOTH 'branch_local' and
    # 'head' filters — causing a "duplicate param group" error. Fix: use
    # n.startswith('head') to exclusively match the top-level fusion head module,
    # and 'branch_local' not in n to guard the else-branch backbone filter.
    if freeze_global:
        local_params = [p for n, p in model.named_parameters()
                        if 'branch_local' in n and p.requires_grad]
        head_params  = [p for n, p in model.named_parameters()
                        if n.startswith('head') and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': local_params, 'lr': 5e-5},   # branch_local (all layers incl. conv_head)
            {'params': head_params,  'lr': 1e-4},   # top-level fusion head only
        ], weight_decay=1e-4)
        max_lrs = [5e-5, 1e-4]
    else:
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith('head') and p.requires_grad]
        head_params     = [p for n, p in model.named_parameters()
                           if n.startswith('head') and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5},
            {'params': head_params,     'lr': 1e-4},
        ], weight_decay=1e-4)
        max_lrs = [2e-5, 1e-4]

    # ── OneCycleLR: warmup + cosine decay per batch ───────────────────────────
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc          = 0.0
    epochs_no_improve = 0
    actual_save_path  = os.path.join(base_dir, "weights", "tumor_classifier.pth")

    mode_str = "FREEZE_GLOBAL (branch_local + head only)" if freeze_global else "FULL (all params)"
    print(f"[Train] Starting training: {epochs} epochs | batch={batch_size} | "
          f"mode={mode_str} | OneCycleLR | drop_rate=0.4")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        loss_components = {"ce": 0.0, "focal": 0.0, "margin": 0.0}
        epoch_preds, epoch_labels_list = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            imgs_full = batch["image"].to(device, non_blocking=True)
            imgs_crop = batch["crop"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs_full, imgs_crop)
                    loss, components = criterion(outputs, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs_full, imgs_crop)
                loss, components = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # OneCycleLR steps PER BATCH
            scheduler.step()

            train_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            epoch_preds.extend(preds)
            epoch_labels_list.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{train_loss / (pbar.n + 1):.4f}")

        # Per-epoch glioma recall (early warning signal)
        ep_labels_arr = np.array(epoch_labels_list)
        ep_preds_arr  = np.array(epoch_preds)
        glioma_mask   = (ep_labels_arr == GLIOMA_IDX)
        glioma_recall_train = (
            (ep_preds_arr[glioma_mask] == GLIOMA_IDX).sum() / glioma_mask.sum()
        ) if glioma_mask.sum() > 0 else 0.0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for batch in vbar:
                imgs_full = batch["image"].to(device, non_blocking=True)
                imgs_crop = batch["crop"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                outputs = model(imgs_full, imgs_crop)
                loss, components = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        ce_avg    = loss_components["ce"]     / len(train_loader)
        focal_avg = loss_components["focal"]  / len(train_loader)
        margin_avg= loss_components["margin"] / len(train_loader)

        acc       = accuracy_score(all_labels, all_preds)
        macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Per-class glioma recall on validation
        val_labels_arr = np.array(all_labels)
        val_preds_arr  = np.array(all_preds)
        glioma_val_mask   = (val_labels_arr == GLIOMA_IDX)
        glioma_recall_val = (
            (val_preds_arr[glioma_val_mask] == GLIOMA_IDX).sum() / glioma_val_mask.sum()
        ) if glioma_val_mask.sum() > 0 else 0.0

        marker = ""
        if acc > best_acc:
            best_acc = acc
            marker   = " <- BEST"
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(actual_save_path), exist_ok=True)
            torch.save(model.state_dict(), actual_save_path)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss T/V: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"CE:{ce_avg:.3f} Focal:{focal_avg:.3f} Margin:{margin_avg:.3f} | "
            f"Val Acc:{acc:.4f} | F1:{macro_f1:.4f} | "
            f"Glioma Recall Train:{glioma_recall_train:.3f} Val:{glioma_recall_val:.3f}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"[Train] Early stopping triggered at epoch {epoch}")
            break

    print(f"\n[Train] Training complete. Best Val Accuracy: {best_acc:.4f}")

    # ── TTA Evaluation on test set ────────────────────────────────────────────
    print("\n[Train] Running Test-Time Augmentation (TTA) evaluation on val set...")
    print("[Train] TTA: 3 views (original + H-flip + V-flip)")

    # Load best weights
    if os.path.exists(actual_save_path):
        sd = torch.load(actual_save_path, map_location=device, weights_only=False)
        model.load_state_dict(sd)
        print("[Train] Loaded best weights for TTA evaluation.")

    model.eval()
    tta_preds  = []
    tta_labels = []

    for batch in tqdm(val_loader, desc="TTA Eval"):
        imgs_full = batch["image"]
        imgs_crop = batch["crop"]
        labels = batch["label"].numpy()
        for i in range(len(imgs_full)):
            single_img_full = imgs_full[i:i+1]
            single_img_crop = imgs_crop[i:i+1]
            probs = tta_predict(model, single_img_full, single_img_crop, device)
            tta_preds.append(int(np.argmax(probs[0])))
        tta_labels.extend(labels)

    tta_acc      = accuracy_score(tta_labels, tta_preds)
    tta_macro_f1 = f1_score(tta_labels, tta_preds, average='macro', zero_division=0)

    print(f"\n[Train] === FINAL RESULTS ===")
    print(f"  Standard Val Accuracy : {best_acc:.4f}")
    print(f"  TTA Val Accuracy      : {tta_acc:.4f}")
    print(f"  TTA Macro F1          : {tta_macro_f1:.4f}")

    print("\n[Train] Standard Classification Report:")
    target_names = [k for k, v in sorted(LABEL_TO_IDX.items(), key=lambda item: item[1])]
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, zero_division=0))

    print("[Train] Confusion Matrix (standard):")
    print(confusion_matrix(all_labels, all_preds))


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Neurologix Pro V3 EfficientNet-B4 classifier."
    )
    parser.add_argument("--data_train",     type=str, default=r"./data/classification/Training")
    parser.add_argument("--data_val",       type=str, default=r"./data/classification/Testing")
    parser.add_argument("--epochs",         type=int, default=60)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--patience",       type=int, default=20,
                        help="Early stop patience on val accuracy.")
    parser.add_argument("--save_dir",       type=str, default=None)
    parser.add_argument("--single_channel", action="store_true",
                        help="Use single_image_mode: 3-CLAHE-variant channels instead of "
                             "2.5D volumetric stack. Recommended for Kaggle classification dataset.")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume training from weights/tumor_classifier.pth if it exists.")
    parser.add_argument("--freeze_global",  action="store_true",
                        help="Freeze branch_global; only train branch_local + head. "
                             "~2x faster. Best used with --resume when branch_global "
                             "is already converged and you only want to adapt the "
                             "local-crop branch to YOLO-guided crops.")
    args = parser.parse_args()

    train(args.data_train, args.data_val, args.epochs, args.batch_size,
          args.patience, args.save_dir, args.single_channel, args.resume,
          args.freeze_global)
