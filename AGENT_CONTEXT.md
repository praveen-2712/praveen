# Neurologix — Agent Context File
> **Last updated:** 2026-06-08 (Metrics sync & audit report generation) | **Project root:** `c:\Users\Praveen\Documents\antigravity\Neurologix\`
> Paste this file at the start of any new agent session to restore full context.

---

## 🎯 Project Goal
Build **Neurologix** — a clinical-grade, strictly-validated brain tumor MRI diagnostic system.

**Core mandate:** Every output (confidence scores, bounding boxes, segmentation masks, centroids) must come from **live model inference only** — zero hardcoded values, zero UI heuristics.

---

## 🏗️ Architecture Overview

### ML Pipeline (3-model ensemble)
| Model | Role | File | Weights |
|---|---|---|---|
| EfficientNet-B4 | 4-class classifier (Glioma / Meningioma / No Tumor / Pituitary) | `src/train_classifier.py` | `weights/tumor_classifier.pth` |
| YOLOv11m | Bounding box detection on tumor region | `src/train_yolo.py` | `weights/detector_yolo.pt` |
| U-Net++ (ResNet50) | Pixel-precise segmentation mask | `src/train_unet.py` | `weights/unet_segmentor.pth` |

**Note:** The OOD (Out-of-Distribution) gate was removed in May 2026 to streamline inference.

### Inference Pipeline
- Entry: `web/app.py` (Flask web server, port 5000)
- Inference orchestrator: `src/inference_engine.py`
- Preprocessing: `src/preprocess.py` (Scanner normalization and CLAHE stack generation)
- Consensus Arbitration: Handles deterministic Glioma gate, MC-Dropout sampling (15 runs), YOLO safety overrides, and dynamic spatial discordance validation.

### Web UI
- Upload page: `web/templates/index.html` → `web/templates/predict.html`
- Results page: `web/templates/result.html` (3-column pipeline view)
- Styles: `web/static/styles.css`

---

## 📁 Directory Structure
```
Neurologix/
├── data/
│   ├── brats_slices/            # Extracted: 10,364 image/mask pairs ✅
│   ├── classification/          # Kaggle brain tumor MRI dataset (7k training, 1.3k test)
│   └── yolo/                    # Generated YOLO dataset (11k+ images/labels)
├── src/                         # All training + inference code
├── web/                         # Flask app + templates
├── weights/                     # Trained weights (Classifier, YOLO, U-Net)
├── Neurologix_Pro_Research_Paper.txt
├── PROJECT_MANIFEST_FOR_REVIEW.txt
└── requirements.txt
```

---

## ✅ Current Task Status

### Completed ✅
- [x] **Branding Update** — Project renamed to "Neurologix". "Pro V3" removed from all UI and docs.
- [x] **OOD Gate Removal** — ConvAutoEncoder removed from pipeline, `app.py`, and `inference_engine.py`.
- [x] **Classifier Hardening** — Achieved **97.69% Accuracy** and **0.9770 Macro F1** under standard test conditions. Weights: `weights/tumor_classifier.pth`.
- [x] **YOLO Detector Hardening** — Retrained YOLOv11m on valid 85/15 split. Achieved **0.9399 mAP@0.5**. Weights: `weights/detector_yolo.pt`.
- [x] **U-Net Validated** — Best Mean Dice: **0.8826** (UnetPlusPlus, ResNet50). Weights: `weights/unet_segmentor.pth`.
- [x] **UI Restoration** — YOLO detection card restored to `result.html` with 3-column layout.
- [x] **Grad-CAM Fixed** — Hooked to `conv_head` layer for high-resolution heatmaps.
- [x] **Research Paper Sync** — Updated to reflect the final 3-stage architecture and latest metrics.
- [x] **Full Evaluation** — All 3 models validated on held-out test sets. Report saved to `weights/evaluation_report.json`.
- [x] **UI Bug Fixes** — Gauge `UNKNOWN` bug fixed, Grad-CAM sharpened, YOLO bbox label readable.
- [x] **Training Status Bug** — Fixed stale `training_status.json` + CUDA detection in `/train/status` endpoint always uses live torch detection (not cached JSON).
- [x] **Stats Accuracy** — Landing page metrics synced to real `evaluation_report.json` values (97.69% acc, 0.940 mAP, 0.883 Dice).
- [x] **Inference Pipeline Upgrades** — Added temperature calibration clamp bounds (`[0.3, 15.0]`), consensus safety override, dynamic spatial discordance (`max(80, 0.40 * diagonal)`), and thread-safe serial inference worker queue.
- [x] **LLM Context Audit File** — Generated `neurologix_project_audit_report.md` containing all verbatim source code and operational specifications.

---

## 🔑 Validation Metrics (Latest — from `weights/evaluation_report.json`)
| Metric | Result | Status |
|---|---|---|
| Classifier Accuracy | 97.69% | ✅ Exceeds 93% target |
| Classifier Macro F1 | 0.9770 | ✅ Exceeds 0.89 target |
| Classifier Macro AUC | 0.9955 | ✅ Excellent |
| YOLO mAP@0.5 | 0.9399 | ✅ Exceeds 0.75 target |
| YOLO mAP@0.5:0.95 | 0.7846 | ✅ |
| U-Net Dice Score | 0.8826 | ✅ Exceeds 0.80 target |

---

## 🚀 Commands to Resume Work
Run from `C:\Users\Praveen\Documents\antigravity\Neurologix\`:

```powershell
# Launch web server
venv\Scripts\python.exe web/app.py

# Run full evaluation suite (Single-channel, No TTA to maintain Glioma recall gate)
venv\Scripts\python.exe src/evaluate.py --single_channel --no_tta
```

---

## 📐 Design Decisions Made
- **Ensemble Priority**: ROI Localization (YOLO) > Segmentation Centroid (U-Net) > Grad-CAM peak.
- **Normalisation**: Scanner normalization (HistMatch) is available in `preprocess.py` but disabled by default to match the training distribution of the high-accuracy classifier.
- **Explainability**: Heatmaps are suppressed for "No Tumor" cases to prevent false-positive visual feedback.
- **Training Status**: `/train/status` endpoint always queries `torch.cuda.is_available()` live — the JSON device field is informational only and overridden at runtime.
- **TTA Policy**: Rotational TTA is disabled because it degrades glioma recall below the 90% gate.

---

## 🗂️ Prior Conversation References
| Conversation | Key Work Done |
|---|---|
| `5ff6bfe2` | **Audit Report & Context Update**: Generated full verbatim code audit file (`neurologix_project_audit_report.md`) for Claude ingestion, synced metrics to final validated scores (97.69% Acc, 0.977 F1), added dynamic discordance and temperature clamp checks. |
| `f6320817` | **Final Training + Fixes**: U-Net retrained (0.8826 Dice), architecture bug fixed, UI gauge/heatmap/bbox fixed. |
| `6ade3ddb` | **Final Polish**: Removed OOD gate, fixed YOLO UI, updated branding to 'Neurologix'. |
| `7072b226` | **YOLO Hardening**: Fixed scheduler bug, reached 0.9621 mAP. |
| `371a4e01` | **Classifier Hardening**: Accuracy upgrade to 95% via TTA and differential LR. |
| `83082c37` | **Documentation**: Drafted Research Paper, fixed Grad-CAM hooks. |
