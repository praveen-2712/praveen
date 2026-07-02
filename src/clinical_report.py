"""
clinical_report.py — Neurologix Pro V3
========================================
Generates a clinical model card for regulatory documentation.
Required for FDA SaMD submission and IEC 62304 Class C documentation.

Usage:
    cd Neurologix
    venv\\Scripts\\python.exe src\\clinical_report.py
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MRI25DDataset, LABEL_TO_IDX
from train_classifier import DualBranchClassifier
from preprocess import Preprocess

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
GLIOMA_IDX  = LABEL_TO_IDX["glioma"]
CLASSES     = [k for k, v in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1])]

SENSITIVITY_TARGETS = {
    "glioma":      0.90,
    "meningioma":  0.85,
    "notumor":     0.90,
    "pituitary":   0.85,
}


def print_clinical_evaluation_report(y_true, y_pred):
    """
    Prints a clinical-grade evaluation report with explicit sensitivity
    pass/fail gates for each class.
    """
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)

    print("\n" + "="*75)
    print("  NEUROLOGIX PRO V3 — CLINICAL SENSITIVITY REPORT")
    print("="*75)
    print(f"  {'Class':<15} {'Recall':>10} {'Target':>10} {'Status':>12} {'F1':>10}")
    print("-"*75)

    all_pass = True
    for cls in CLASSES:
        recall  = report[cls]["recall"]
        target  = SENSITIVITY_TARGETS[cls]
        passed  = recall >= target
        status  = "PASS" if passed else "FAIL"
        f1      = report[cls]["f1-score"]
        if not passed:
            all_pass = False
        print(f"  {cls:<15} {recall:>9.1%} {target:>9.1%} {status:>12} {f1:>9.3f}")

    print("="*75)
    print(f"\n  Overall Accuracy  : {report['accuracy']:.1%}")
    print(f"  Macro F1          : {report['macro avg']['f1-score']:.4f}")
    print(f"  Macro Precision   : {report['macro avg']['precision']:.4f}")
    print(f"  Macro Recall      : {report['macro avg']['recall']:.4f}")

    gate_str = "ALL CLASSES PASS" if all_pass else "FAILED — NOT DEPLOYABLE"
    print(f"\n  Clinical Gate     : {gate_str}")

    # Glioma detail
    glioma_recall = report["glioma"]["recall"]
    glioma_fn     = cm[GLIOMA_IDX].sum() - cm[GLIOMA_IDX, GLIOMA_IDX]
    glioma_total  = cm[GLIOMA_IDX].sum()
    men_idx       = LABEL_TO_IDX["meningioma"]
    not_idx       = LABEL_TO_IDX["notumor"]

    print(f"\n  === Glioma Detail ===")
    print(f"  Sensitivity      : {glioma_recall:.1%}")
    print(f"  Missed cases     : {glioma_fn} / {glioma_total}")
    print(f"  -> meningioma    : {cm[GLIOMA_IDX, men_idx]}")
    print(f"  -> notumor       : {cm[GLIOMA_IDX, not_idx]}")

    if glioma_recall < 0.90:
        print(f"\n  *** CLINICAL SAFETY ALERT ***")
        print(f"  Glioma sensitivity {glioma_recall:.1%} is BELOW the 90% minimum.")
        print(f"  This model MUST NOT be deployed in a clinical environment.")
        print(f"  Continue retraining or adjust GLIOMA_THRESHOLD.")

    return all_pass, {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        **{cls: {"recall": report[cls]["recall"],
                 "precision": report[cls]["precision"],
                 "f1": report[cls]["f1-score"]}
           for cls in CLASSES},
        "confusion_matrix": cm.tolist(),
    }


def generate_model_card(eval_results: dict, output_path: str = None):
    """
    Generate a clinical model card for regulatory documentation.
    Based on FDA AI/ML Action Plan and EU AI Act requirements.
    """
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "MODEL_CARD.md")

    threshold_path = os.path.join(WEIGHTS_DIR, "clinical_thresholds.json")
    thresholds = {}
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            thresholds = json.load(f)

    glioma_thresh = thresholds.get("glioma_threshold", 0.30)
    indet_thresh  = thresholds.get("indeterminate_threshold", 0.55)
    temp          = thresholds.get("calibration_temperature", 1.0)

    def cls_row(cls, target):
        r   = eval_results.get(cls, {}).get("recall", 0.0)
        sp  = "PASS" if r >= target else "FAIL"
        return f"| {cls.capitalize():<12} | {r:.1%} | {target:.0%} | {sp} |"

    card = f"""# Neurologix Pro V3 — Clinical Model Card

## Intended Use
Decision support tool for brain tumor MRI classification.  
**NOT a standalone diagnostic device. Requires radiologist oversight.**  
Classes: Glioma, Meningioma, No Tumor, Pituitary Adenoma

## Performance Summary

| Class        | Sensitivity | Target | Met |
|---|---|---|---|
{cls_row("glioma", 0.90)}
{cls_row("meningioma", 0.85)}
{cls_row("notumor", 0.90)}
{cls_row("pituitary", 0.85)}

**Overall Accuracy**: {eval_results.get("accuracy", 0):.1%}  
**Macro F1**: {eval_results.get("macro_f1", 0):.4f}

## Clinical Decision Thresholds
- **Glioma detection threshold** : P(glioma) > {glioma_thresh:.5e}
- **Indeterminate threshold**    : max(P) < {indet_thresh:.2f} → radiologist referral
- **Calibration temperature**    : T = {temp:.4f}

## Safety Features
- MC-Dropout (15 passes): epistemic uncertainty quantification
- TTA (2 passes: original + H-flip): aleatoric uncertainty reduction
- Indeterminate output: required by IEC 62304 Class C
- Radiologist referral flag: compliant with FDA AI guidance
- GliomaMarginLoss: prevents Glioma↔Meningioma confusion at training time
- 6× Glioma class weight: clinical asymmetric penalty for missed diagnoses

## Known Limitations
- Trained on T1-weighted MRI only. T2/FLAIR not supported.
- Single 2D slice input. Volumetric analysis not performed.
- Not validated on paediatric populations.
- Not validated on post-operative or treated tumor imaging.

## Regulatory Framework
- Standard: IEC 62304 (Software as a Medical Device)
- Guidance: FDA AI/ML-Based SaMD Action Plan
- Decision support only — final diagnosis by licensed radiologist
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[ClinicalReport] Model card written to {output_path}")


def run(data_val: str = None):
    if data_val is None:
        data_val = os.path.join(BASE_DIR, "data", "classification", "Testing")

    ckpt_path = os.path.join(WEIGHTS_DIR, "tumor_classifier.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ClinicalReport] ERROR: weights not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load calibrated glioma threshold from clinical_thresholds.json
    threshold_path = os.path.join(WEIGHTS_DIR, "clinical_thresholds.json")
    glioma_threshold = 0.30  # safe default
    if os.path.exists(threshold_path):
        with open(threshold_path) as _tf:
            _td = json.load(_tf)
        glioma_threshold = float(_td.get("glioma_threshold", 0.30))
    print(f"[ClinicalReport] Using glioma_threshold = {glioma_threshold:.4f}")

    val_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    preprocessor = Preprocess()

    # Load yolo_boxes for consistent branch_local crop quality
    yolo_boxes_path = os.path.join(BASE_DIR, "yolo_boxes.json")
    yolo_boxes = {}
    if os.path.exists(yolo_boxes_path):
        with open(yolo_boxes_path) as _f:
            yolo_boxes = json.load(_f)

    val_ds     = MRI25DDataset(data_val, transform=val_transform, preprocess_logic=preprocessor,
                               img_size=380, single_image_mode=True, yolo_boxes=yolo_boxes)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DualBranchClassifier(num_classes=len(LABEL_TO_IDX))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    all_preds  = []
    all_labels = []
    all_probs  = []
    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            crops  = batch["crop"].to(device)
            labels = batch["label"]
            logits = model(imgs, crops)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())

            preds = []
            for p in probs:
                if p[GLIOMA_IDX] > glioma_threshold:
                    preds.append(GLIOMA_IDX)
                else:
                    preds.append(int(np.argmax(p)))
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    passed, eval_results = print_clinical_evaluation_report(all_labels, all_preds)
    generate_model_card(eval_results)

    report_path = os.path.join(WEIGHTS_DIR, "clinical_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[ClinicalReport] Full results saved to {report_path}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_val", type=str, default=None)
    args = parser.parse_args()
    run(args.data_val)
