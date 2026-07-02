# 🧠 Neurologix — Clinical Brain Tumor MRI Diagnostic System

> **Research Tool Only — Not a Medical Device. Requires radiologist oversight.**

Neurologix is a clinical-grade, 3-model ensemble pipeline for brain tumor MRI analysis. It performs simultaneous **classification**, **detection**, and **segmentation** on a single MRI scan, producing a structured radiology-style report.

![Overall Accuracy](https://img.shields.io/badge/Accuracy-97.69%25-brightgreen) ![Macro F1](https://img.shields.io/badge/Macro%20F1-0.977-brightgreen) ![YOLO mAP](https://img.shields.io/badge/YOLO%20mAP%400.5-0.940-blue) ![U--Net Dice](https://img.shields.io/badge/Dice%20Score-0.883-blue)

---

## ✨ Key Features

- **4-class tumor classification** — Glioma, Meningioma, No Tumor, Pituitary Adenoma
- **YOLO bounding box detection** — localizes the tumor region
- **U-Net++ segmentation** — pixel-precise tumor mask with ICV % estimate
- **Grad-CAM heatmaps** — visual explainability for every prediction
- **MC-Dropout uncertainty** — 15 stochastic passes for epistemic confidence
- **Demo Mode** — works out of the box even without weights (uses synthetic outputs)
- **Web UI** — Flask-based dashboard with multi-image upload, history, and training monitor

---

## 🚀 Quick Start (No Training Required)

### 1. Clone the repo (with model weights via Git LFS)
```bash
git clone https://github.com/YOUR_USERNAME/Neurologix.git
cd Neurologix

# Pull the large model weight files (requires Git LFS)
git lfs pull
```

> **Install Git LFS first if you haven't:**
> ```bash
> # Windows (winget)
> winget install GitHub.GitLFS
> # macOS
> brew install git-lfs
> # Linux
> sudo apt install git-lfs
> ```
> Then run `git lfs install` once.

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web app
```bash
python web/app.py
```

Open **http://localhost:5000** in your browser and upload a brain MRI scan.

> **No GPU?** The app automatically falls back to CPU inference. It will be slower but fully functional.

---

## 🏗️ Architecture

### 3-Model Ensemble Pipeline

| Model | Role | Backbone | Weights File | Metrics |
|---|---|---|---|---|
| EfficientNet-B4 | 4-class Classifier | timm EfficientNet | `weights/tumor_classifier.pth` | Acc 97.69%, F1 0.977 |
| YOLOv11m | Bounding Box Detector | Ultralytics YOLO | `weights/detector_yolo.pt` | mAP@0.5 0.940 |
| U-Net++ (ResNet50) | Segmentation | segmentation-models-pytorch | `weights/unet_segmentor.pth` | Dice 0.883 |

### Inference Flow
```
MRI Upload → CLAHE Preprocessing → [Classifier | YOLO Detector | U-Net Segmentor]
           → Consensus Arbitration → Grad-CAM Heatmap → Clinical Report
```

Key safety features:
- **Glioma deterministic gate** — any P(glioma) > 2.4e-5 triggers glioma alert
- **YOLO safety override** — detection without classification flags unclassified finding
- **Spatial discordance check** — flags cases where classifier region ≠ YOLO/U-Net centroid
- **Indeterminate output** — max(P) < 0.55 → mandatory radiologist referral

---

## 📁 Directory Structure

```
Neurologix/
├── src/
│   ├── inference_engine.py     # Core 3-model ensemble orchestrator
│   ├── train_classifier.py     # EfficientNet-B4 classifier training
│   ├── train_yolo.py           # YOLOv11m detector training
│   ├── train_unet.py           # U-Net++ segmentor training
│   ├── evaluate.py             # Full evaluation suite
│   ├── calibrate.py            # Temperature scaling calibration
│   ├── dataset.py              # Dataset loaders
│   ├── preprocess.py           # CLAHE preprocessing + scanner normalization
│   └── clinical_report.py      # Structured report generator
├── web/
│   ├── app.py                  # Flask web server (port 5000)
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS assets
├── weights/                    # Trained model weights (Git LFS)
│   ├── tumor_classifier.pth    # ~168 MB — EfficientNet-B4 classifier
│   ├── detector_yolo.pt        # ~120 MB — YOLOv11m detector
│   ├── unet_segmentor.pth      # ~196 MB — U-Net++ segmentor
│   ├── unet_last_checkpoint.pth# ~588 MB — Training checkpoint (resume training)
│   ├── label_map.json
│   ├── clinical_thresholds.json
│   ├── calibration_temperature.pt
│   └── evaluation_report.json
├── models/
│   └── classifier_norm.json    # Dataset normalization stats
├── data/                       # NOT included — see Training section
├── train_all.py                # One-shot full pipeline training script
├── requirements.txt
└── MODEL_CARD.md               # Model card with safety documentation
```

---

## 🎛️ Demo Mode vs. Clinical Mode

The app automatically detects which mode to use at startup:

| Mode | Condition | Behaviour |
|---|---|---|
| **Clinical Mode** | `weights/tumor_classifier.pth` > 1 MB | Full 3-model inference on every upload |
| **Demo Mode** | Weights missing or are LFS pointers | Synthetic outputs shown — UI fully functional |

A banner in the UI clearly indicates which mode is active.

---

## 🔬 Re-training from Scratch (Optional)

If you want to retrain the models yourself instead of using the provided weights:

### 1. Prepare datasets
```bash
# Kaggle brain tumor MRI classification dataset
python setup_kaggle_data.py

# BraTS 2020 slices for U-Net segmentation
# (requires ~10k .nii.gz files in data/brats/)
python src/preprocess.py
```

### 2. Train all three models
```bash
python train_all.py
# or step-by-step:
python src/train_classifier.py
python src/train_yolo.py
python src/train_unet.py
```

### 3. Calibrate and evaluate
```bash
python src/calibrate.py
python src/evaluate.py --single_channel --no_tta
```

---

## 📊 Validation Metrics

| Metric | Result | Target | Status |
|---|---|---|---|
| Classifier Accuracy | **97.69%** | 93% | ✅ |
| Classifier Macro F1 | **0.9770** | 0.89 | ✅ |
| Classifier Macro AUC | **0.9955** | — | ✅ |
| YOLO mAP@0.5 | **0.9399** | 0.75 | ✅ |
| YOLO mAP@0.5:0.95 | **0.7846** | — | ✅ |
| U-Net Dice Score | **0.8826** | 0.80 | ✅ |

Per-class sensitivity (classifier):

| Class | Sensitivity | Target | Status |
|---|---|---|---|
| Glioma | 91.0% | 90% | ✅ |
| Meningioma | 99.2% | 85% | ✅ |
| No Tumor | 100.0% | 90% | ✅ |
| Pituitary | 98.7% | 85% | ✅ |

---

## ⚠️ Disclaimer

This system is a **research prototype** and is **NOT approved as a medical device**. It must not be used for clinical diagnosis without oversight from a licensed radiologist. All outputs are for decision-support only.

- Standard: IEC 62304 (Software as a Medical Device)
- Guidance: FDA AI/ML-Based SaMD Action Plan

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
