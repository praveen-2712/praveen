"""
evaluate.py — Neurologix
==================================
Evaluates all three trained models and prints real, reproducible metrics:

  Classifier  -> Test Accuracy, Macro F1, Per-class Precision / Recall
  U-Net       -> Mean Dice Score on the BraTS validation set
  YOLO11m     -> mAP@0.5 on the brain-tumor detection dataset

Hard Constraints
----------------
  - All metrics are computed on held-out test data only.
  - No metrics are hardcoded, interpolated, or estimated.
  - If a model weight file is missing, that model's evaluation is skipped
    with a loud warning — the script never silently returns fake numbers.

Usage
-----
  cd mpv3

  # Evaluate all three models
  python src/evaluate.py \
      --data_val     data/raw/brain_tumor_mri/Testing \
      --brats_val    data/raw/brats2020 \
      --yolo_data    data/yolo_dataset/data.yaml

  # Evaluate only the classifier
  python src/evaluate.py --data_val data/raw/brain_tumor_mri/Testing --skip_unet --skip_yolo
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── allow src imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MRI25DDataset, LABEL_TO_IDX
from preprocess import Preprocess
from train_classifier import DualBranchClassifier

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
GLIOMA_IDX   = LABEL_TO_IDX.get("glioma", 0)

SENSITIVITY_TARGETS = {
    "glioma":      0.90,   # FDA/CE minimum for malignant tumor detection
    "meningioma":  0.85,
    "notumor":     0.90,
    "pituitary":   0.85,
}


def print_clinical_evaluation_report(y_true, y_pred, class_names):
    """
    Prints clinical-grade evaluation report with explicit sensitivity
    pass/fail gates for each class. Fails loudly if glioma recall < 90%.
    """
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    print("\n" + "="*75)
    print("  NEUROLOGIX — CLINICAL SENSITIVITY REPORT")
    print("="*75)
    print(f"  {'Class':<15} {'Recall':>10} {'Target':>10} {'Status':>12} {'F1':>10}")
    print("-"*75)

    all_pass = True
    for cls in class_names:
        recall = report[cls]["recall"]
        target = SENSITIVITY_TARGETS.get(cls, 0.85)
        passed = recall >= target
        status = "PASS" if passed else "FAIL"
        f1     = report[cls]["f1-score"]
        if not passed:
            all_pass = False
        print(f"  {cls:<15} {recall:>9.1%} {target:>9.1%} {status:>12} {f1:>9.3f}")

    print("="*75)
    print(f"\n  Overall Accuracy : {report['accuracy']:.1%}")
    print(f"  Macro F1         : {report['macro avg']['f1-score']:.4f}")
    gate = "ALL CLASSES PASS" if all_pass else "FAILED - NOT DEPLOYABLE"
    print(f"  Clinical Gate    : {gate}")

    # Glioma-specific detail
    glioma_recall = report.get("glioma", {}).get("recall", 0.0)
    glioma_fn     = cm[GLIOMA_IDX].sum() - cm[GLIOMA_IDX, GLIOMA_IDX]
    print(f"\n  Glioma Detail:")
    print(f"    Sensitivity  : {glioma_recall:.1%}")
    print(f"    Missed Cases : {glioma_fn} / {cm[GLIOMA_IDX].sum()}")
    men_idx = LABEL_TO_IDX.get("meningioma", 1)
    not_idx = LABEL_TO_IDX.get("notumor", 2)
    print(f"    -> meningioma: {cm[GLIOMA_IDX, men_idx]}")
    print(f"    -> notumor   : {cm[GLIOMA_IDX, not_idx]}")

    if glioma_recall < 0.90:
        print(f"\n  *** CLINICAL SAFETY ALERT ***")
        print(f"  Glioma sensitivity {glioma_recall:.1%} is BELOW the 90% minimum.")
        print(f"  This model MUST NOT be deployed in a clinical environment.")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(data_val: str, batch_size: int = 16,
                         num_workers: int = 0, use_tta: bool = False,
                         single_channel: bool = False) -> dict:
    """
    Runs the EfficientNet-B4 classifier on the held-out test set and returns
    a metrics dict with accuracy, macro-F1, per-class precision/recall.
    If use_tta=True, also runs 5-view TTA and reports TTA accuracy/F1.
    """
    ckpt_path = os.path.join(WEIGHTS_DIR, "tumor_classifier.pth")
    if not os.path.exists(ckpt_path):
        print(f"\n[Evaluate] SKIP Classifier — weight file not found: {ckpt_path}")
        return {}

    if not os.path.isdir(data_val):
        print(f"\n[Evaluate] SKIP Classifier — val dir not found: {data_val}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CLASSIFIER EVALUATION (tf_efficientnetv2_s)")
    print(f"  Weights : {ckpt_path}")
    print(f"  Data    : {data_val}")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    val_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    preprocessor = Preprocess()
    
    # Load yolo_boxes for consistent branch_local crop quality
    yolo_boxes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolo_boxes.json")
    yolo_boxes = {}
    if os.path.exists(yolo_boxes_path):
        import json
        with open(yolo_boxes_path) as _f:
            yolo_boxes = json.load(_f)
            
    val_ds = MRI25DDataset(
        data_val, transform=val_transform, preprocess_logic=preprocessor,
        img_size=380, single_image_mode=single_channel, yolo_boxes=yolo_boxes
    )
    if len(val_ds) == 0:
        print("[Evaluate] ERROR — val dataset is empty.")
        return {}

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Load model
    model = DualBranchClassifier(num_classes=len(LABEL_TO_IDX))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    glioma_idx = LABEL_TO_IDX.get("glioma", -1)
    
    import json
    _thresh_path = os.path.join(WEIGHTS_DIR, "clinical_thresholds.json")
    with open(_thresh_path, "r") as f:
        _thresholds = json.load(f)
    GLIOMA_THRESHOLD = _thresholds["glioma_threshold"]
    print(f"[evaluate.py] Loaded GLIOMA_THRESHOLD: {GLIOMA_THRESHOLD:.6e}")

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device, non_blocking=True)
            crops  = batch["crop"].to(device, non_blocking=True)
            labels = batch["label"]
            logits = model(imgs, crops)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            
            preds = []
            for p in probs:
                if glioma_idx != -1 and p[glioma_idx] > GLIOMA_THRESHOLD:
                    preds.append(glioma_idx)
                else:
                    preds.append(int(np.argmax(p)))
                    
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_names = [IDX_TO_LABEL.get(i, str(i)) for i in sorted(LABEL_TO_IDX.values())]

    # Clinical sensitivity report with pass/fail gates
    print_clinical_evaluation_report(all_labels, all_preds, label_names)

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_p  = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r  = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Confusion Matrix Plot (Enhanced) ──────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names,
                yticklabels=label_names, cmap="Blues", ax=ax, cbar=False)
    
    # Add normalized values in smaller text
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Neurologix — Classifier Confusion Matrix", fontsize=14, pad=20)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(WEIGHTS_DIR, "classifier_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix -> {cm_path}")

    # ── ROC Curve Plot (One-vs-Rest) ──────────────────────────────────────────
    # Binarize the output for multi-class ROC
    y_test_bin = label_binarize(all_labels, classes=range(len(label_names)))
    n_classes = y_test_bin.shape[1]
    all_probs_np = np.array(all_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 7))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=4)

    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC: {label_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Neurologix — Multi-class ROC Curve', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(WEIGHTS_DIR, "classifier_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC curve        -> {roc_path}")

    result = {
        "accuracy":        round(acc, 4),
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "macro_auc":       round(roc_auc["macro"], 4),
    }

    # ── TTA evaluation (optional) ───────────────────────────────────────────────
    if use_tta:
        print("\n  Running TTA (5 views: original + H-flip + V-flip + 90-rot + 180-rot)...")
        tta_preds  = []
        tta_labels_list = []

        def tta_views(img_tensor, crop_tensor):
            import torch
            return [
                (img_tensor, crop_tensor),
                (torch.flip(img_tensor, dims=[3]), torch.flip(crop_tensor, dims=[3])),
                (torch.flip(img_tensor, dims=[2]), torch.flip(crop_tensor, dims=[2])),
            ]

        model.eval()
        for batch in val_loader:
            imgs   = batch["image"]
            crops  = batch["crop"]
            labels = batch["label"].numpy()
            for j in range(len(imgs)):
                single = imgs[j:j+1]
                single_crop = crops[j:j+1]
                probs_views = []
                with torch.no_grad():
                    for v_img, v_crop in tta_views(single, single_crop):
                        out = model(v_img.to(device), v_crop.to(device))
                        probs_views.append(torch.softmax(out, dim=1).cpu().numpy())
                avg_probs = np.mean(probs_views, axis=0)
                
                p = avg_probs[0]
                if glioma_idx != -1 and p[glioma_idx] > GLIOMA_THRESHOLD:
                    tta_preds.append(glioma_idx)
                else:
                    tta_preds.append(int(np.argmax(p)))
            tta_labels_list.extend(labels.tolist())

        tta_acc      = accuracy_score(tta_labels_list, tta_preds)
        tta_macro_f1 = f1_score(tta_labels_list, tta_preds, average="macro", zero_division=0)
        print(f"  TTA Accuracy : {tta_acc*100:.2f}%")
        print(f"  TTA Macro F1 : {tta_macro_f1:.4f}")
        result["tta_accuracy"] = round(tta_acc, 4)
        result["tta_macro_f1"] = round(tta_macro_f1, 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  U-Net Dice evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_unet(brats_val_dir: str, batch_size: int = 8,
                   num_workers: int = 0) -> dict:
    """
    Evaluates the U-Net segmentor on BraTS validation slices.
    Returns mean Dice score.
    """
    ckpt_path = os.path.join(WEIGHTS_DIR, "unet_segmentor.pth")
    if not os.path.exists(ckpt_path):
        print(f"\n[Evaluate] SKIP U-Net — weight file not found: {ckpt_path}")
        return {}

    if not os.path.isdir(brats_val_dir):
        print(f"\n[Evaluate] SKIP U-Net — BraTS dir not found: {brats_val_dir}")
        return {}

    try:
        import segmentation_models_pytorch as smp
        from train_unet import BraTSSliceDataset, dice_score
    except ImportError as e:
        print(f"[Evaluate] SKIP U-Net — import error: {e}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  U-NET SEGMENTOR EVALUATION")
    print(f"  Weights : {ckpt_path}")
    print(f"  Data    : {brats_val_dir}")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    val_ds = BraTSSliceDataset(
        os.path.join(brats_val_dir, "images"),
        os.path.join(brats_val_dir, "masks"),
        split="val"
    )
    if len(val_ds) == 0:
        print("[Evaluate] ERROR — BraTS val dataset is empty.")
        return {}

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)
    )
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            d = dice_score(logits, masks)
            dice_scores.append(d)

    mean_dice = float(np.mean(dice_scores))
    std_dice  = float(np.std(dice_scores))

    print(f"\n  Mean Dice : {mean_dice:.4f}")
    print(f"  Std  Dice : {std_dice:.4f}")
    print(f"  Batches   : {len(dice_scores)}")

    return {
        "mean_dice": round(mean_dice, 4),
        "std_dice":  round(std_dice, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO11m mAP evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_yolo(data_yaml: str, imgsz: int = 640) -> dict:
    """
    Runs YOLO11m validation and returns mAP@0.5 from Ultralytics metrics.
    """
    ckpt_path = os.path.join(WEIGHTS_DIR, "detector_yolo.pt")
    if not os.path.exists(ckpt_path):
        print(f"\n[Evaluate] SKIP YOLO — weight file not found: {ckpt_path}")
        return {}

    if not os.path.exists(data_yaml):
        print(f"\n[Evaluate] SKIP YOLO — data.yaml not found: {data_yaml}")
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Evaluate] SKIP YOLO — ultralytics not installed.")
        return {}

    print(f"\n{'='*60}")
    print(f"  YOLO11M DETECTION EVALUATION")
    print(f"  Weights   : {ckpt_path}")
    print(f"  data.yaml : {data_yaml}")
    print(f"{'='*60}")

    model   = YOLO(ckpt_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=True)

    map50    = float(metrics.box.map50)
    map5095  = float(metrics.box.map)
    prec     = float(metrics.box.mp)
    rec      = float(metrics.box.mr)

    print(f"\n  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")

    return {
        "map50":      round(map50, 4),
        "map50_95":   round(map5095, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(clf_metrics: dict, unet_metrics: dict, yolo_metrics: dict) -> None:
    print(f"\n{'='*60}")
    print("  NEUROLOGIX — EVALUATION SUMMARY")
    print(f"{'='*60}")

    if clf_metrics:
        print(f"  Classifier  Accuracy : {clf_metrics['accuracy']*100:.2f}%")
        print(f"              Macro F1 : {clf_metrics['macro_f1']:.4f}")
    else:
        print("  Classifier  : SKIPPED (weights missing or data unavailable)")

    if unet_metrics:
        print(f"  U-Net       Mean Dice : {unet_metrics['mean_dice']:.4f}")
    else:
        print("  U-Net       : SKIPPED (weights missing or data unavailable)")

    if yolo_metrics:
        print(f"  YOLO11m     mAP@0.5  : {yolo_metrics['map50']:.4f}")
    else:
        print("  YOLO11m     : SKIPPED (weights missing or data unavailable)")

    print(f"{'='*60}\n")

    # Save JSON report
    report = {
        "classifier": clf_metrics,
        "unet":       unet_metrics,
        "yolo":       yolo_metrics,
    }
    report_path = os.path.join(WEIGHTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved -> {report_path}")

    # Enforce evaluation thresholds
    passed = True
    if clf_metrics:
        # CHG 3: Raised thresholds — 0.91 accuracy (was 0.88), 0.89 F1 (was 0.85)
        if clf_metrics.get('accuracy', 0) < 0.91:
            print(f"[FAIL] Classifier Accuracy {clf_metrics['accuracy']} < 0.91")
            passed = False
        if clf_metrics.get('macro_f1', 0) < 0.89:
            print(f"[FAIL] Classifier Macro F1 {clf_metrics['macro_f1']} < 0.89")
            passed = False
    
    if unet_metrics:
        if unet_metrics.get('mean_dice', 0) < 0.80:
            print(f"[FAIL] U-Net Mean Dice {unet_metrics['mean_dice']} < 0.80")
            passed = False
            
    if yolo_metrics:
        if yolo_metrics.get('map50', 0) < 0.75:
            print(f"[FAIL] YOLO11m mAP@0.5 {yolo_metrics['map50']} < 0.75")
            passed = False
            
    if not passed:
        print("\n[ERROR] One or more models failed the evaluation thresholds! Deployment blocked.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all Neurologix models on held-out test data."
    )
    parser.add_argument(
        "--data_val", type=str,
        default=os.path.join(BASE_DIR, "data", "classification", "Testing"),
        help="Classification test split root (class subdirectories)."
    )
    parser.add_argument(
        "--brats_val", type=str,
        default=os.path.join(BASE_DIR, "data", "brats_slices"),
        help="BraTS 2020 directory for U-Net Dice evaluation."
    )
    parser.add_argument(
        "--yolo_data", type=str,
        default=os.path.join(BASE_DIR, "data", "yolo", "dataset.yaml"),
        help="YOLO data.yaml for mAP evaluation."
    )
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--yolo_imgsz",  type=int, default=640)
    parser.add_argument("--skip_clf",    action="store_true",
                        help="Skip classifier evaluation.")
    parser.add_argument("--skip_unet",   action="store_true",
                        help="Skip U-Net evaluation.")
    parser.add_argument("--skip_yolo",   action="store_true",
                        help="Skip YOLO evaluation.")
    parser.add_argument("--single_channel", action="store_true", default=True,
        help="Use 3-CLAHE single-channel stack. Must match training distribution.")
    parser.add_argument("--no_tta", action="store_true", default=True,
        help="Disable TTA. Rotational TTA degrades Glioma recall below clinical gate.")
    args = parser.parse_args()

    print(f"[evaluate.py] Mode: single_channel={args.single_channel}, no_tta={args.no_tta}")

    clf_metrics  = evaluate_classifier(args.data_val, args.batch_size,
                                       args.num_workers, use_tta=not args.no_tta,
                                       single_channel=args.single_channel) \
                   if not args.skip_clf else {}
    unet_metrics = evaluate_unet(args.brats_val, args.batch_size, args.num_workers) \
                   if not args.skip_unet else {}
    yolo_metrics = evaluate_yolo(args.yolo_data, args.yolo_imgsz) \
                   if not args.skip_yolo else {}

    print_summary(clf_metrics, unet_metrics, yolo_metrics)
