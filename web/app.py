import sys
import os
import logging
import random
import datetime
import torch
import numpy as np
import cv2
import json
import base64
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference_engine import ClinicalInferenceEngine, DemoInferenceEngine

try:
    from pyngrok import ngrok as _ngrok
except ImportError:
    _ngrok = None

app = Flask(__name__)

# BUG-14 FIX: allowed upload extensions whitelist
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

CONFIG = {
    "classifier": "../weights/tumor_classifier.pth",
    "detector":   "../weights/detector_yolo.pt",
    "unet":       "../weights/unet_segmentor.pth",
    "label_map":  "../weights/label_map.json",
    "device":     "cuda" if torch.cuda.is_available() else "cpu"
}

engine          = None
classifier_norm = None
idx_to_label    = {}
HISTORY_CACHE   = []
TRAINING_PROCESS = None
TRAINING_STATUS_FILE = os.path.join(os.path.dirname(__file__), "training_status.json")


def load_engine():
    global engine, idx_to_label, classifier_norm, _worker
    base = os.path.dirname(__file__)
    
    lm_p   = os.path.join(base, CONFIG["label_map"])
    norm_p = os.path.join(base, "../models/classifier_norm.json")

    if os.path.exists(lm_p):
        with open(lm_p) as f:
            raw = json.load(f)
            idx_to_label = {int(v): k for k, v in raw.items()}
    else:
        idx_to_label = {0: "glioma", 1: "meningioma",
                        2: "notumor", 3: "pituitary"}

    if os.path.exists(norm_p):
        with open(norm_p) as f:
            classifier_norm = json.load(f)
    else:
        # FIX 4: Fall back to ImageNet defaults — never leave classifier_norm None.
        classifier_norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        print(
            "[App] WARNING: models/classifier_norm.json not found. "
            "Falling back to ImageNet defaults. "
            "Run python src/train_classifier.py to regenerate it."
        )

    # ── Engine Selection: Real weights → ClinicalInferenceEngine
    #                      No weights / LFS pointers → DemoInferenceEngine ────
    # Git LFS pointer files are only ~133 bytes. We require >1MB to be a real
    # model checkpoint — this prevents _pickle.UnpicklingError on LFS stubs.
    # Each weight is validated independently so partial installs are handled.
    classifier_path = os.path.join(base, CONFIG["classifier"])
    unet_path       = os.path.join(base, CONFIG["unet"])
    detector_path   = os.path.join(base, CONFIG["detector"])

    def _is_real_weight(path):
        return os.path.exists(path) and os.path.getsize(path) > 1_000_000

    classifier_real = _is_real_weight(classifier_path)
    unet_real       = _is_real_weight(unet_path)
    # YOLO optional — classifier alone is sufficient for clinical mode
    weights_are_real = classifier_real

    if weights_are_real:
        print("[App] Trained weights detected — loading ClinicalInferenceEngine...")
        print(f"[App]   Classifier : OK ({os.path.getsize(classifier_path)/1e6:.1f} MB)")
        print(f"[App]   U-Net      : {'OK' if unet_real else 'LFS/Missing — segmentor disabled'}")
        engine = ClinicalInferenceEngine(
            classifier_path=classifier_path,
            detector_path  =detector_path,
            unet_path      =unet_path if unet_real else "",
            device         =CONFIG["device"],
            label_map      =idx_to_label
        )
    else:
        if os.path.exists(classifier_path):
            print("[App] [WARN] Weight files are Git LFS pointers (not downloaded).")
        else:
            print("[App] [WARN] No trained weights found.")
        print("[App]    Starting in DEMO MODE -- upload any brain MRI to test the UI.")
        print("[App]    To enable real inference, download weights via Git LFS or train from scratch.")
        engine = DemoInferenceEngine(
            device    =CONFIG["device"],
            label_map =idx_to_label
        )

    # Start the worker when the engine is initialized
    _worker = _InferenceWorker(engine)
    _worker.start()



def to_b64(img_np):
    """Convert a NumPy RGB image to a Base64-encoded PNG string."""
    if img_np is None:
        return ""
    try:
        h, w = img_np.shape[:2]
        if max(h, w) > 640:
            scale  = 640 / max(h, w)
            img_np = cv2.resize(img_np,
                                (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(img_np.astype(np.uint8))
        buf     = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[B64] Encoding error: {e}")
        return ""


def _make_case_meta():
    """BUG-12 FIX: generate case ID and timestamp in Python, not in Jinja."""
    case_id   = f"NLE-{random.randint(1000, 9999)}"
    timestamp = datetime.datetime.now().strftime("%H:%M · %d %b %Y")
    return case_id, timestamp


import queue, threading
_inference_queue = queue.Queue(maxsize=10)

class _InferenceWorker(threading.Thread):
    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
    def run(self):
        while True:
            task = _inference_queue.get()
            if task is None:
                break
            s_tensor, raw_np, mode, event, holder = task
            try:
                holder["result"] = self.engine.predict(s_tensor, raw_np, mode=mode)
            except Exception as e:
                holder["error"] = str(e)
            finally:
                event.set()
                _inference_queue.task_done()
@app.route("/", methods=["GET", "POST"])
def index():
    is_demo = isinstance(engine, DemoInferenceEngine)
    if request.method == "POST":
        return redirect(url_for("predict",
                                mode=request.form.get("mode", "multi")))
    return render_template("index.html", is_demo=is_demo)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    mode = request.args.get("mode", "multi")
    is_demo = isinstance(engine, DemoInferenceEngine)

    if request.method == "POST":
        # ── Multi-Image Decode ──────────────────────────────────────────────
        images = request.files.getlist("image")
        if not images or (len(images) == 1 and images[0].filename == ''):
            return "No images uploaded.", 400

        all_results = []
        for file in images:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue

            try:
                img_pil = Image.open(file).convert("RGB")
            except:
                continue

            raw_np  = np.array(img_pil)
            gray_np = np.array(img_pil.convert("L"))

            # Preprocess
            IMG_SIZE = 380
            gray_resized = cv2.resize(gray_np, (IMG_SIZE, IMG_SIZE))
            channels = []
            for clip_limit in [1.5, 2.0, 2.5]:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                channels.append(clahe.apply(gray_resized))
            stack = np.stack(channels, axis=-1)

            stack_np = stack
            assert stack_np.dtype == np.uint8 or stack_np.dtype == np.float32, \
                f"Unexpected stack dtype: {stack_np.dtype}"
            if stack_np.dtype == np.uint8:
                s_tensor = torch.from_numpy(stack_np).permute(2,0,1).float() / 255.0
            else:
                s_tensor = torch.from_numpy(stack_np).permute(2,0,1).float()

            _norm_path = os.path.join(os.path.dirname(__file__), "..", "models", "classifier_norm.json")
            if os.path.exists(_norm_path):
                with open(_norm_path) as _f:
                    _norm = json.load(_f)
                MEAN = torch.tensor(_norm["mean"]).view(3,1,1)
                STD  = torch.tensor(_norm["std"]).view(3,1,1)
                print(f"[app.py] Loaded normalization stats from {_norm_path}")
            else:
                MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

            device = CONFIG["device"]
            s_tensor = (s_tensor.unsqueeze(0).to(device) - MEAN.to(device)) / STD.to(device)

            _event = threading.Event()
            _holder = {}
            try:
                _inference_queue.put_nowait((s_tensor, raw_np, mode, _event, _holder))
            except queue.Full:
                return jsonify({"error": "Server busy. Try again shortly."}), 503
            
            _event.wait(timeout=30)
            if "error" in _holder:
                return jsonify({"error": _holder["error"]}), 500
            res = _holder["result"]
            case_id, timestamp = _make_case_meta()

            # ── Assemble clinical report ───────────────────────────────────────
            conf_pct = round(res["confidence"], 2)
            raw_label_text = res["label"]   # display label, e.g. "Glioma", "Tumor Detected"
            # For narrative text, get the clean display name without uppercasing
            narrative_label = raw_label_text  # already display-ready
            is_tumor = raw_label_text not in ("No Tumor Detected",)
            has_mass = res.get("mask_area", 0) > 0
            icv = res.get("icv_pct", 0.0)
            centroid = res.get("centroid", "N/A")

            explanation_parts = []

            if res["is_discordant"]:
                explanation_parts.append("CRITICAL ALERT: Discordant Modalities Detected. The spatial localized tumor centroid differs significantly from the classified region. Automated results are considered non-diagnostic; mandatory expert review required.")
            elif res["used_yolo_override"]:
                explanation_parts.append("Radiological Alert: The system detected a potential pathology sector via YOLO-Consensus that was initially omitted by the primary classifier. This case is flagged as 'Unclassified Finding' for specialist audit.")
            
            if is_tumor:
                if conf_pct >= 80.0:
                    explanation_parts.append(f"Radiological Review: MRI scan demonstrates a distinct pathological mass with radiomic features highly characteristic of a {narrative_label}.")
                    if has_mass:
                        explanation_parts.append(f"The lesion occupies approximately {icv}% of the intracranial volume, with defined margins centered at {centroid}.")
                    explanation_parts.append(f"The diagnostic confidence is very high ({conf_pct}%). Immediate clinical correlation and specialist consultation are recommended.")
                
                elif conf_pct >= 60.0:
                    explanation_parts.append(f"Radiological Review: Analysis of the MRI scan reveals an abnormal region suggestive of a {narrative_label}.")
                    if has_mass:
                        explanation_parts.append(f"A localized mass occupying roughly {icv}% of the intracranial volume is noted.")
                    explanation_parts.append(f"However, the diagnostic confidence is moderate ({conf_pct}%). Differential diagnosis must be considered.")
                
                else:
                    explanation_parts.append(f"Radiological Review: Potential pathological features identified, but with low confidence ({conf_pct}%). Result is clinically unreliable.")
            else:
                if conf_pct >= 80.0:
                    explanation_parts.append(f"Radiological Review: Evaluation of the MRI slice reveals no definitive pathological masses. Parenchyma appears unremarkable ({conf_pct}% confidence).")
                else:
                    explanation_parts.append(f"Radiological Review: Analysis leans towards a negative finding, but with low diagnostic confidence ({conf_pct}%). Manual review required.")

            report = {
                "explanation": " ".join(explanation_parts),
                "clinical_relevance": "RESEARCH TOOL ONLY — Not a Medical Device.",
                "primary_roi": None,
            }

            if res["roi_crop"] is not None:
                report["primary_roi"] = {
                    "crop_b64": to_b64(res["roi_crop"]),
                    "location": "Localised Pathology Sector",
                    "area_px":  f"{res['icv_pct']:.2f}% ICV" if res["icv_pct"] > 0 else f"{res['mask_area']} px",
                    "centroid": f"({res['centroid'][0]}, {res['centroid'][1]})" if res["centroid"] else "N/A",
                    "sources":  ["Ensemble Pipeline"],
                }

            all_results.append({
                "label": res["label"].upper(),
                "confidence": conf_pct,
                "original_image": to_b64(res["heatmap_img"]),
                "detection_image": to_b64(res["detection_img"]),
                "segmentation_image": to_b64(res["segmentation_img"]),
                "roi_crop": to_b64(res["roi_crop"]),
                "case_id": case_id,
                "timestamp": timestamp,
                "report": report,
                "is_discordant": res["is_discordant"],
                "used_yolo_override": res["used_yolo_override"]
            })

        # Save to history cache
        global HISTORY_CACHE
        for r in all_results:
            if not any(item['case_id'] == r['case_id'] for item in HISTORY_CACHE):
                HISTORY_CACHE.insert(0, r)
        HISTORY_CACHE = HISTORY_CACHE[:20]

        r = all_results[0]
        return render_template(
            "result.html",
            mode               =mode,
            label              =r["label"],
            confidence         =r["confidence"],
            original_image     =r["original_image"],
            detection_image    =r["detection_image"],
            segmentation_image =r["segmentation_image"],
            roi_crop           =r["roi_crop"],
            case_id            =r["case_id"],
            timestamp          =r["timestamp"],
            is_rejected        =False,
            rejection_reason   ="",
            report             =r["report"],
            is_discordant      =r["is_discordant"],
            used_yolo_override =r["used_yolo_override"],
            is_demo            =is_demo,
            results            =all_results
        )

    return render_template("predict.html", mode=mode, is_demo=is_demo)


@app.route("/history")
def history():
    is_demo = isinstance(engine, DemoInferenceEngine)
    return render_template("history.html", history=HISTORY_CACHE, is_demo=is_demo)


@app.route("/history/view/<case_id>")
def view_history(case_id):
    is_demo = isinstance(engine, DemoInferenceEngine)
    record = None
    for r in HISTORY_CACHE:
        if r["case_id"] == case_id:
            record = r
            break
            
    if record is None:
        return "Report not found or expired from history.", 404
        
    return render_template(
        "result.html",
        mode               ="multi",
        label              =record["label"],
        confidence         =record["confidence"],
        original_image     =record["original_image"],
        detection_image    =record["detection_image"],
        segmentation_image =record["segmentation_image"],
        roi_crop           =record["roi_crop"],
        case_id            =record["case_id"],
        timestamp          =record["timestamp"],
        is_rejected        =False,
        rejection_reason   ="",
        report             =record["report"],
        is_discordant      =record["is_discordant"],
        used_yolo_override =record["used_yolo_override"],
        is_demo            =is_demo,
        results            =[record]
    )


def get_cuda_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU Only (No CUDA Available)"


@app.route("/train")
def train():
    is_demo = isinstance(engine, DemoInferenceEngine)
    device_name = get_cuda_device_name()
    return render_template("train.html", is_demo=is_demo, device_name=device_name)


@app.route("/train/status")
def train_status():
    global TRAINING_PROCESS
    
    is_running = False
    if TRAINING_PROCESS is not None:
        poll = TRAINING_PROCESS.poll()
        if poll is None:
            is_running = True
        else:
            TRAINING_PROCESS = None
            
    status_data = {
        "device": get_cuda_device_name(),
        "status": "running" if is_running else "idle",
        "current_model": "Pipeline Inactive" if not is_running else "Ensemble pipeline running...",
        "stage": 1 if is_running else 0,
        "progress": 0.0,
        "eta": "N/A",
        "epoch": 0,
        "total_epochs": 0,
        "loss": 0.0,
        "val_acc": 0.0,
        "elapsed": "00:00"
    }
    
    if os.path.exists(TRAINING_STATUS_FILE):
        try:
            with open(TRAINING_STATUS_FILE, "r") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    status_data[k] = v
                status_data["status"] = "running" if is_running else "completed" if saved.get("status") == "completed" else "idle"
        except Exception as e:
            print(f"[Train] Error reading status file: {e}")
    
    # Always override device with live detection — never trust stale JSON value
    status_data["device"] = get_cuda_device_name()
            
    return jsonify(status_data)


@app.route("/train/start", methods=["POST"])
def train_start():
    global TRAINING_PROCESS
    
    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        return jsonify({"message": "Training pipeline is already running in the background!"}), 400
        
    initial_status = {
        "device": get_cuda_device_name(),
        "status": "running",
        "current_model": "Ensemble Pipeline Launching...",
        "stage": 1,
        "progress": 0.0,
        "eta": "Calculating...",
        "epoch": 0,
        "total_epochs": 30,
        "loss": 0.0,
        "val_acc": 0.0,
        "elapsed": "00:00"
    }
    
    try:
        with open(TRAINING_STATUS_FILE, "w") as f:
            json.dump(initial_status, f)
    except Exception as e:
        print(f"[Train] Error writing initial status file: {e}")

    base = os.path.dirname(__file__)
    script_path = os.path.join(base, "..", "train_all.py")
    python_exe = sys.executable or "python"
    
    try:
        TRAINING_PROCESS = subprocess.Popen([python_exe, script_path], cwd=os.path.join(base, ".."))
        return jsonify({"message": "Training pipeline spawned successfully in background! Monitoring active."})
    except Exception as e:
        return jsonify({"message": f"Failed to spawn training script: {str(e)}"}), 500


@app.route("/train/stop", methods=["POST"])
def train_stop():
    global TRAINING_PROCESS
    
    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        try:
            TRAINING_PROCESS.terminate()
            TRAINING_PROCESS.wait(timeout=5)
            TRAINING_PROCESS = None
            
            if os.path.exists(TRAINING_STATUS_FILE):
                try:
                    with open(TRAINING_STATUS_FILE, "r") as f:
                        data = json.load(f)
                    data["status"] = "idle"
                    data["current_model"] = "Training Aborted by User"
                    with open(TRAINING_STATUS_FILE, "w") as f:
                        json.dump(data, f)
                except:
                    pass
                    
            return jsonify({"message": "Training pipeline terminated successfully."})
        except Exception as e:
            return jsonify({"message": f"Error terminating process: {str(e)}"}), 500
    else:
        return jsonify({"message": "Training pipeline is not running."}), 400



@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Verifies that all model weight files exist on disk and returns their
    file sizes. Does NOT reload models (engine is loaded at startup).
    Returns HTTP 200 if all files are present, HTTP 503 if any are missing.
    """
    base        = os.path.dirname(__file__)
    weight_keys = {
        "classifier":   CONFIG["classifier"],
        "segmentor":    CONFIG["unet"],
        "norm_config":  "../models/classifier_norm.json",
    }

    status  = {}
    healthy = True

    for model_name, rel_path in weight_keys.items():
        abs_path = os.path.join(base, rel_path)
        exists   = os.path.isfile(abs_path)
        status[model_name] = {
            "path":    abs_path,
            "present": exists,
            "size_mb": (
                round(os.path.getsize(abs_path) / (1024 ** 2), 2)
                if exists else None
            ),
        }
        if not exists:
            healthy = False

    engine_loaded = engine is not None
    if not engine_loaded:
        healthy = False

    response = {
        "status":        "healthy" if healthy else "degraded",
        "engine_loaded": engine_loaded,
        "device":        CONFIG["device"],
        "models":        status,
        "disclaimer":    (
            "This system is for research purposes only and does not constitute "
            "a medical device or replace clinical radiological diagnosis."
        ),
    }
    return jsonify(response), (200 if healthy else 503)


if __name__ == "__main__":
    load_engine()

    # ── NGROK PUBLIC TUNNEL (optional) ───────────────────────────────────────
    # Set the NGROK_AUTHTOKEN environment variable to enable public tunneling.
    # Do NOT hardcode your token here — rotate any leaked tokens at ngrok.com.
    # Example: set NGROK_AUTHTOKEN=your_token_here  (Windows)
    if _ngrok is not None:
        ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "")
        if ngrok_token:
            try:
                _ngrok.set_auth_token(ngrok_token)
                public_url = _ngrok.connect(5000).public_url
                print(f"\n[NGROK] Public URL: {public_url}")
                print("[NGROK] Share this link with others to access your dashboard!\n")
            except Exception as e:
                print(f"\n[NGROK] Could not start tunnel: {e}")
                print("[NGROK] App accessible at http://127.0.0.1:5000\n")
        else:
            print("[NGROK] No NGROK_AUTHTOKEN set — skipping tunnel.")

    # host='0.0.0.0' allows access from other devices on the same network
    app.run(host='0.0.0.0', debug=False, port=5000)
