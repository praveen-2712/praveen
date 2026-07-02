"""
calibrate.py — Neurologix Pro V3
===================================
Post-hoc temperature scaling calibration.

Run AFTER retraining, BEFORE deployment.
Fits a single temperature parameter T on the validation set such that
the model's softmax probabilities are well-calibrated.

Per FDA AI guidance: probability outputs used for clinical decision-making
must be calibrated and validated on a held-out dataset.

Usage:
    cd Neurologix
    venv\\Scripts\\python.exe src\\calibrate.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MRI25DDataset, LABEL_TO_IDX
from train_classifier import DualBranchClassifier

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
GLIOMA_IDX  = LABEL_TO_IDX["glioma"]
CLASSES     = [k for k, v in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1])]


class TemperatureScaler:
    """
    Post-hoc temperature scaling calibration.
    Fits a single scalar T on logits to minimize NLL on validation set.
    T > 1: model was overconfident → softens probabilities
    T < 1: model was underconfident → sharpens probabilities
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        def nll(T):
            scaled    = torch.tensor(logits / T, dtype=torch.float32)
            log_probs = F.log_softmax(scaled, dim=1).numpy()
            return -log_probs[np.arange(len(labels)), labels].mean()

        # Widened to (0.1, 30.0) — prevents silent convergence to old 10.0 wall.
        # T>10 typically signals a dataset distribution mismatch, not true overconfidence.
        result = minimize_scalar(nll, bounds=(0.1, 30.0), method="bounded")
        self.temperature = float(result.x)

        print(f"\n[Calibrate] Optimal temperature: T = {self.temperature:.4f}")
        if self.temperature >= 29.5:
            print(f"  *** WARNING: T hit upper bound (30.0). Calibration may be degenerate.")
            print(f"  *** This usually means train/val distribution mismatch. Check your val set.")
            print(f"  *** Defaulting to T=1.0 (no scaling) is safer in this case.")
        elif self.temperature > 1.0:
            print(f"  Model was OVERCONFIDENT -> probabilities softened")
        else:
            print(f"  Model was UNDERCONFIDENT -> probabilities sharpened")
        print(f"  NLL before: {nll(1.0):.4f}  |  NLL after: {nll(self.temperature):.4f}")
        return self.temperature


    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = torch.tensor(logits / self.temperature, dtype=torch.float32)
        return F.softmax(scaled, dim=1).numpy()

    def save(self, path: str):
        torch.save({"temperature": self.temperature}, path)
        print(f"[Calibrate] Temperature saved to {path}")

    @classmethod
    def load(cls, path: str):
        scaler = cls()
        data   = torch.load(path, map_location="cpu")
        scaler.temperature = float(data["temperature"])
        return scaler


def tune_glioma_threshold(
    glioma_probs: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """
    Empirically tune GLIOMA_THRESHOLD on the validation ROC curve.
    Finds the highest threshold T such that glioma sensitivity >= target.
    This is the clinically correct way to set the threshold — NOT a guess.
    """
    binary_labels = (labels == GLIOMA_IDX).astype(int)
    fpr, tpr, thresholds = roc_curve(binary_labels, glioma_probs)
    roc_auc = auc(fpr, tpr)

    valid_thresholds = thresholds[tpr >= target_sensitivity]

    print(f"\n[Calibrate] Glioma Threshold Calibration (target sensitivity={target_sensitivity:.0%}):")
    if len(valid_thresholds) == 0:
        print(f"  WARNING: Cannot achieve {target_sensitivity:.0%} sensitivity at any threshold.")
        print(f"  Model needs retraining. Using 0.20 as emergency fallback threshold.")
        return 0.20

    # sklearn roc_curve returns thresholds in DECREASING order, tpr in INCREASING order.
    # valid_thresholds[0]  = highest threshold where tpr >= target  (what we want)
    # valid_thresholds[-1] = lowest threshold (≈0) where tpr = 1.0 (always trivially true)
    optimal_T = float(valid_thresholds[0])
    idx_match  = np.where(thresholds >= optimal_T)[0]
    if len(idx_match) == 0:
        idx_match = [len(tpr) - 1]
    best_idx   = idx_match[-1]
    achieved_sensitivity = float(tpr[best_idx])
    achieved_specificity = float(1 - fpr[best_idx])

    print(f"  Optimal threshold  : {optimal_T:.4f}")
    print(f"  Sensitivity achieved: {achieved_sensitivity:.1%} (target: {target_sensitivity:.0%})")
    print(f"  Specificity achieved: {achieved_specificity:.1%}")
    print(f"  AUC (Glioma vs Rest): {roc_auc:.4f}")
    return optimal_T


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str = None,
):
    """
    Plot calibration reliability diagrams for all classes.
    A well-calibrated model has curves close to the diagonal.
    Required for FDA/CE clinical AI validation documentation.
    """
    if output_path is None:
        output_path = os.path.join(WEIGHTS_DIR, "reliability_diagram.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (cls, ax) in enumerate(zip(CLASSES, axes)):
        binary_labels = (labels == idx).astype(int)
        cls_probs     = probs[:, idx]

        if binary_labels.sum() == 0:
            ax.set_title(f"{cls} — no samples")
            continue

        fraction_pos, mean_pred = calibration_curve(binary_labels, cls_probs, n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(mean_pred, fraction_pos, "o-", color="steelblue", label=f"{cls} (model)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Reliability Diagram: {cls.capitalize()}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Neurologix Pro V3 — Calibration Reliability Diagrams", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Calibrate] Reliability diagram saved to {output_path}")


def run_calibration(
    data_val: str = None,
    target_sensitivity: float = 0.90,
):
    if data_val is None:
        data_val = os.path.join(BASE_DIR, "data", "classification", "Testing")

    ckpt_path = os.path.join(WEIGHTS_DIR, "tumor_classifier.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Calibrate] ERROR: weights not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Calibrate] Device: {device}")

    val_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Load yolo_boxes for consistent branch_local crop quality
    yolo_boxes_path = os.path.join(BASE_DIR, "yolo_boxes.json")
    yolo_boxes = {}
    if os.path.exists(yolo_boxes_path):
        with open(yolo_boxes_path) as _f:
            yolo_boxes = json.load(_f)
        print(f"[Calibrate] Loaded {len(yolo_boxes)} YOLO boxes.")

    val_ds = MRI25DDataset(data_val, transform=val_transform, img_size=380,
                           single_image_mode=True, yolo_boxes=yolo_boxes)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DualBranchClassifier(num_classes=len(LABEL_TO_IDX))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            crops  = batch["crop"].to(device)
            labels = batch["label"]
            logits = model(imgs, crops)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)

    # 1. Fit temperature
    scaler = TemperatureScaler()
    scaler.fit(all_logits, all_labels)

    calib_path = os.path.join(WEIGHTS_DIR, "calibration_temperature.pt")
    scaler.save(calib_path)

    # 2. Get calibrated probabilities (for reliability diagram only)
    calib_probs = scaler.calibrate(all_logits)

    # 3. Tune glioma threshold on UNCALIBRATED (T=1) probabilities
    # CRITICAL: At high temperatures (T≥5), all post-temp probs collapse to ~1/N,
    # making the ROC threshold degenerate. The threshold MUST be calibrated in
    # the same probability space used at inference time.
    # Since inference_engine averages 19 logit passes then applies T,
    # and calibrate uses single-pass logits, we calibrate the threshold on
    # T=1 (uncalibrated) probabilities — this is a stable, meaningful space.
    raw_probs = torch.softmax(torch.tensor(all_logits, dtype=torch.float32), dim=1).numpy()
    optimal_threshold = tune_glioma_threshold(
        glioma_probs=raw_probs[:, GLIOMA_IDX],
        labels=all_labels,
        target_sensitivity=target_sensitivity,
    )

    # 4. Save threshold
    threshold_data = {
        "glioma_threshold":       optimal_threshold,
        "indeterminate_threshold": 0.55,
        "calibration_temperature": scaler.temperature,
    }
    threshold_path = os.path.join(WEIGHTS_DIR, "clinical_thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\n[Calibrate] Clinical thresholds saved to {threshold_path}")
    print(f"  GLIOMA_THRESHOLD (T=1 space)   = {optimal_threshold:.4f}")
    print(f"  INDETERMINATE_THRESHOLD        = 0.55")
    print(f"  CALIBRATION_TEMPERATURE        = {scaler.temperature:.4f}")
    print(f"  NOTE: Glioma threshold is calibrated on T=1 probabilities.")
    print(f"        Inference engine compares T=1 single-pass prob to this threshold.")
    print(f"        (19-pass logit average uses temperature for final softmax display only)")

    # 5. Reliability diagram (on calibrated probs)
    plot_reliability_diagram(calib_probs, all_labels)

    return optimal_threshold, scaler.temperature


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate Neurologix Pro V3 classifier probabilities.")
    parser.add_argument("--data_val", type=str, default=None)
    parser.add_argument("--target_sensitivity", type=float, default=0.90)
    args = parser.parse_args()
    run_calibration(args.data_val, args.target_sensitivity)
