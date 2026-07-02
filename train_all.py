import os
import sys
import time
import json
import subprocess
import re

STATUS_FILE = os.path.join(os.path.dirname(__file__), "web", "training_status.json")

def update_status(status_dict):
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
    except Exception as e:
        print(f"[Master Train] Error updating status file: {e}")

def get_cuda_device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except:
        pass
    return "CPU Only (No CUDA Available)"

def run_step(command, stage_num, model_name, total_epochs, status_template):
    print(f"\n==================================================")
    print(f"   Spawning Stage {stage_num}: {model_name}")
    print(f"   Command: {' '.join(command)}")
    print(f"==================================================")

    # Setup status template
    status_template["stage"] = stage_num
    status_template["current_model"] = model_name
    status_template["total_epochs"] = total_epochs
    status_template["epoch"] = 0
    status_template["loss"] = 0.0
    status_template["val_acc"] = 0.0
    status_template["progress"] = ((stage_num - 1) / 3.0) * 100.0
    update_status(status_template)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    stage_start = time.time()

    # Regexes to capture training logs
    # e.g., "Epoch 02/60 | Loss T/V: 0.3541/0.4215 | Val Acc: 0.8420"
    classifier_epoch_re = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Loss T/V:\s+([\d\.]+)/[\d\.]+\s+\|\s+Val Acc:\s+([\d\.]+)")
    classifier_loss_re = re.compile(r"loss=([\d\.]+)")
    
    # e.g., "  ↳ [P2] Epoch 3/75  mAP@0.5: 0.7230"
    yolo_epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)\s+mAP@0.5:\s+([\d\.]+)")
    yolo_loss_re = re.compile(r"loss:\s+([\d\.]+)")

    # e.g., "Epoch 003/050 | Loss: 0.2450 | Val Dice: 0.6540"
    unet_epoch_re = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Val Dice:\s+([\d\.]+)")

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        if not line:
            continue
            
        print(line.strip()) # Stream to console

        # Parse Classifier Output
        c_match = classifier_epoch_re.search(line)
        if c_match:
            ep = int(c_match.group(1))
            val_metric = float(c_match.group(3))
            
            status_template["epoch"] = ep
            status_template["val_acc"] = val_metric
            
            # Update elapsed & ETA
            elapsed_sec = time.time() - stage_start
            status_template["elapsed"] = f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}"
            
            time_per_ep = elapsed_sec / ep
            remaining_ep = total_epochs - ep + (30 if stage_num == 1 else 20 if stage_num == 2 else 0) # approximation for remaining stages
            eta_sec = time_per_ep * remaining_ep
            status_template["eta"] = f"{int(eta_sec // 60)}m" if eta_sec < 3600 else f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            
            status_template["progress"] = (((stage_num - 1) / 3.0) + (ep / total_epochs / 3.0)) * 100.0
            update_status(status_template)
            continue
            
        c_loss = classifier_loss_re.search(line)
        if c_loss:
            status_template["loss"] = float(c_loss.group(1))
            update_status(status_template)
            continue

        # Parse YOLO Output
        y_match = yolo_epoch_re.search(line)
        if y_match:
            ep = int(y_match.group(1))
            val_metric = float(y_match.group(3))
            
            status_template["epoch"] = ep
            status_template["val_acc"] = val_metric
            
            elapsed_sec = time.time() - stage_start
            status_template["elapsed"] = f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}"
            
            time_per_ep = elapsed_sec / ep
            remaining_ep = total_epochs - ep + (20 if stage_num == 2 else 0)
            eta_sec = time_per_ep * remaining_ep
            status_template["eta"] = f"{int(eta_sec // 60)}m" if eta_sec < 3600 else f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            
            status_template["progress"] = (((stage_num - 1) / 3.0) + (ep / total_epochs / 3.0)) * 100.0
            update_status(status_template)
            continue
            
        y_loss = yolo_loss_re.search(line)
        if y_loss:
            try:
                status_template["loss"] = float(y_loss.group(1).split('|')[0])
                update_status(status_template)
            except:
                pass
            continue

        # Parse U-Net Output
        u_match = unet_epoch_re.search(line)
        if u_match:
            ep = int(u_match.group(1))
            loss_val = float(u_match.group(2))
            val_metric = float(u_match.group(3))
            
            status_template["epoch"] = ep
            status_template["loss"] = loss_val
            status_template["val_acc"] = val_metric
            
            elapsed_sec = time.time() - stage_start
            status_template["elapsed"] = f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}"
            
            time_per_ep = elapsed_sec / ep
            remaining_ep = total_epochs - ep
            eta_sec = time_per_ep * remaining_ep
            status_template["eta"] = f"{int(eta_sec // 60)}m" if eta_sec < 3600 else f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            
            status_template["progress"] = (((stage_num - 1) / 3.0) + (ep / total_epochs / 3.0)) * 100.0
            update_status(status_template)
            continue

    rc = process.wait()
    if rc != 0:
        print(f"[Master Train] ERROR: Stage {stage_num} failed with return code {rc}")
        status_template["status"] = "idle"
        status_template["current_model"] = f"Error in Stage {stage_num}"
        update_status(status_template)
        sys.exit(rc)

    print(f"[Master Train] Stage {stage_num} completed successfully.")

def main():
    device = get_cuda_device_name()
    print("=" * 60)
    print("      Neurologix Master Ensembled Training Pipeline")
    print("=" * 60)
    print(f"Target CUDA Device: {device}\n")

    # Initial status object
    status = {
        "device": device,
        "status": "running",
        "current_model": "Stage 0: Preparing Data...",
        "stage": 0,
        "progress": 0.0,
        "eta": "Calculating...",
        "epoch": 0,
        "total_epochs": 1,
        "loss": 0.0,
        "val_acc": 0.0,
        "elapsed": "00:00"
    }
    update_status(status)

    python_exe = sys.executable or "python"

    # Step 1: Ingest/Extract BraTS slices (requires data downloaded at ./data/brats/)
    print("\n--- Running BraTS Slice Extraction ---")
    extract_rc = subprocess.run([python_exe, "scripts/extract_brats_slices.py"], capture_output=True, text=True)
    if extract_rc.returncode != 0:
        print("[ERROR] Failed to extract BraTS slices. Please verify that the BraTS dataset exists in ./data/brats/")
        print("Error output:\n", extract_rc.stderr)
        status["status"] = "idle"
        status["current_model"] = "Data Extraction Failed"
        update_status(status)
        sys.exit(1)
    print("[Success] BraTS slice extraction completed.")

    # Step 2: Generate YOLO labels
    print("\n--- Generating YOLO Bounding Box Labels ---")
    yolo_gen_rc = subprocess.run([python_exe, "scripts/generate_yolo_labels.py"], capture_output=True, text=True)
    if yolo_gen_rc.returncode != 0:
        print("[ERROR] Failed to generate YOLO labels.")
        print("Error output:\n", yolo_gen_rc.stderr)
        status["status"] = "idle"
        status["current_model"] = "YOLO Label Gen Failed"
        update_status(status)
        sys.exit(1)
    print("[Success] YOLO labels generated.")

    # Step 3: Run Stage 1 Training (YOLO Detector)
    # We use YOLO11m. Runs Phase 1 (25 epochs) + Phase 2 (75 epochs max)
    # We pass total_epochs as 60 since early stopping typically hits around there.
    run_step(
        [python_exe, "src/train_yolo.py"],
        stage_num=1,
        model_name="Stage 1: YOLO11m Detector",
        total_epochs=60,
        status_template=status
    )

    # Precompute YOLO boxes before classifier training
    print("\n--- Running Bounding Box Precomputation ---")
    precompute_rc = subprocess.run([python_exe, "precompute_boxes.py"], capture_output=True, text=True)
    if precompute_rc.returncode != 0:
        print("[WARN] precompute_boxes.py failed — classifier will use brain-mask fallback.")
        print("Precompute error output:\n", precompute_rc.stderr)
    else:
        print("[Success] YOLO bounding boxes precomputed.")

    # Step 4: Run Stage 2 Training (U-Net++ Segmentor)
    # 20 epochs is highly optimized with FocalLoss + Mixed Precision
    run_step(
        [python_exe, "src/train_unet.py", "--epochs", "20"],
        stage_num=2,
        model_name="Stage 2: U-Net++ Segmentor",
        total_epochs=20,
        status_template=status
    )

    # Step 5: Run Stage 3 Training (Classifier)
    # 30 epochs is highly optimized with OneCycleLR
    run_step(
        [python_exe, "src/train_classifier.py", "--epochs", "30", "--single_channel"],
        stage_num=3,
        model_name="Stage 3: EfficientNetV2-S Classifier",
        total_epochs=30,
        status_template=status
    )

    # Complete!
    status["status"] = "completed"
    status["current_model"] = "Training Completed Successfully!"
    status["progress"] = 100.0
    status["eta"] = "Done"
    status["epoch"] = 0
    status["total_epochs"] = 0
    status["loss"] = 0.0
    status["val_acc"] = 1.0
    status["elapsed"] = "Done"
    update_status(status)
    
    print("\n" + "=" * 60)
    print("   Ensembled Training Completed Successfully!")
    print("   All ensembled checkpoints have been updated in ./weights/")
    print("==================================================")

if __name__ == "__main__":
    main()
