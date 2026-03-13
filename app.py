from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

# Import utils - assuming these will be created
from utils.preprocess import load_and_preprocess_image
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)

# Configuration
MODEL_PATH = "models/tumor_classifier.h5"
LABEL_MAP_PATH = "models/label_map.json"

# Global variables for model and labels
model = None
idx_to_label = {}

def load_model_and_labels():
    global model, idx_to_label
    
    # Load Label Map
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}
        print(f"Loaded label map: {idx_to_label}")
    else:
        print(f"Warning: Label map not found at {LABEL_MAP_PATH}")
        # Default fallback if needed, or just empty
        idx_to_label = {0: "No Model Loaded"}

    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

# Load on startup
load_model_and_labels()

def pil_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

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
        if "image" not in request.files:
            return "No file part", 400

        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        if model is None:
            return "Model not loaded. Please train the model first.", 503

        try:
            img_array, pil_image = load_and_preprocess_image(file)

            # Run prediction
            preds = model.predict(img_array)[0]
            predicted_idx = int(np.argmax(preds))
            confidence = float(preds[predicted_idx])
            
            # Get label safely
            label = idx_to_label.get(predicted_idx, "Unknown")

            # Binary mode logic
            binary_label = None
            if mode == "binary":
                # Assuming 'no_tumor' is the label for no tumor
                if label.lower() == "no_tumor":
                    binary_label = "No Tumor"
                else:
                    binary_label = "Tumor"

            # Grad CAM
            # Note: 'top_conv' is a placeholder. EfficientNetB0 usually has 'top_activation' or similar.
            # We will need to verify the layer name after training.
            # For now, we'll wrap in try-except to avoid crashing if layer name is wrong
            heatmap_b64 = None
            try:
                last_conv_layer_name = "out_relu"  # MobileNetV2 last activation layer
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                overlayed = overlay_heatmap(heatmap, pil_image)
                heatmap_b64 = pil_to_base64(Image.fromarray(overlayed))
            except Exception as e:
                print(f"GradCAM error: {e}")
                # Fallback: just show original image twice or handle in template
                heatmap_b64 = pil_to_base64(pil_image)

            original_b64 = pil_to_base64(pil_image)

            return render_template(
                "result.html",
                mode=mode,
                label=label,
                binary_label=binary_label,
                confidence=round(confidence * 100, 2),
                original_image=original_b64,
                heatmap_image=heatmap_b64
            )
        except Exception as e:
            print(f"Prediction error: {e}")
            return f"An error occurred during prediction: {e}", 500

    return render_template("predict.html", mode=mode)

if __name__ == "__main__":
    app.run(debug=True)
