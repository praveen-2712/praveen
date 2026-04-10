# Neurologix AI: Brain Tumor Detection System

A research-grade full-stack web application designed for analyzing Brain MRI scans.

## Features
- **Hybrid AI Pipeline**: Architecture prepared for YOLOv8 (Detection), U-Net (Segmentation), and EfficientNet-B4 (Classification).
- **Robust Fallback**: Advanced simulated image processing ensures the pipeline runs without requiring gigabytes of model weights for demonstration.
- **Frontend**: Dynamic React-based architecture via CDN featuring a modern UI built with Tailwind CSS.
- **Explainability**: Heatmap overlays dynamically mapped over specific tumor crop regions.

## How to Run

Because this is built fully integrated without Node.js dependencies:

1. **Install Python Dependencies**:
   Open a terminal in this directory (`mp`) and run:
   ```bash
   py -m pip install -r requirements.txt
   ```
   *(If `py` doesn't work, try `python` or `python3`)*

2. **Start the Flask Server**:
   ```bash
   py app.py
   ```

3. **Open the Application**:
   Navigate to `http://localhost:5000` in your web browser. Upload any Brain MRI image (.jpg or .png) to see the full visualization suite.
