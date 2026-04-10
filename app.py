from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image
import subprocess
import threading
from utils.gpu_config import get_nvidia_device

# Import Torch Utils
from utils.preprocess import load_and_preprocess_image
from utils.inference import HybridInference
from utils.analyzer import generate_report

app = Flask(__name__)

# Configuration
CONFIG = {
    "classifier": "models/tumor_classifier.pth",
    "yolo": "runs/detect/neurologix_yolo_bt_cpu4/weights/best.pt",
    "unet": "models/unet_segmentor.pth",
    "label_map": "models/label_map.json"
}

# Global variables
engine = None
idx_to_label = {}
device = get_nvidia_device()

def load_engine():
    global engine, idx_to_label
    
    # Load Label Map
    if os.path.exists(CONFIG["label_map"]):
        with open(CONFIG["label_map"], "r") as f:
            label_map = json.load(f)
        idx_to_label = {int(v): k for k, v in label_map.items()}
    else:
        idx_to_label = {0: "Unknown"}

    # Initialize Hybrid Engine
    engine = HybridInference(
        yolo_path=CONFIG["yolo"],
        clf_path=CONFIG["classifier"],
        unet_path=CONFIG["unet"],
        device=device,
        label_map=idx_to_label
    )

load_engine()

def pil_to_base64(pil_image):
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mode = request.form.get("mode", "multi")
        return redirect(url_for("predict", mode=mode))
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    mode = request.args.get("mode", "multi")
    if request.method == "POST":
        if "image" not in request.files: return "No file", 400
        file = request.files["image"]
        if engine is None: return "Engine not initialized", 503

        try:
            image_tensor, pil_display = load_and_preprocess_image(file)
            
            # Hybrid Inference
            results = engine.predict(image_tensor, pil_display)
            
            # Generate Clinical Report
            report = generate_report(results["label"], round(results["confidence"], 2), results["detections"], np.array(pil_display).shape)

            return render_template(
                "result.html",
                mode=mode, 
                label=results["label"], 
                confidence=round(results["confidence"], 2),
                original_image=pil_to_base64(results["overlayed_core"]) if results["overlayed_core"] is not None else pil_to_base64(pil_display),
                detection_image=pil_to_base64(results["annotated_image"]) if results["annotated_image"] is not None else pil_to_base64(pil_display),
                segmentation_image=pil_to_base64(results["segmented_image"]) if results["segmented_image"] is not None else pil_to_base64(pil_display),
                report=report
            )
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error: {e}", 500

    return render_template("predict.html", mode=mode)

if __name__ == "__main__":
    app.run(debug=True)
