import os
import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"STARTING COMPONENT TRAINING: {script_name}")
    print(f"{'='*60}\n")
    
    python_exe = sys.executable
    process = subprocess.Popen(
        [python_exe, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ""):
        clean_line = line.strip().encode('ascii', 'ignore').decode('ascii')
        print(f"[{script_name}] {clean_line}")
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"\nCRITICAL FAILURE: {script_name} exited with code {process.returncode}")
        return False
    
    print(f"\nSUCCESS: {script_name} completed.")
    return True

def full_train():
    start_time = time.time()
    
    components = [
        # "scripts/train_yolo.py",
        # "scripts/train_unet.py",
        "scripts/train.py"
    ]
    
    print("Initializing Clinical Ensemble Training Suite (120 Total Epochs)...")
    
    for script in components:
        if not os.path.exists(script):
            print(f"Error: {script} not found.")
            sys.exit(1)
            
        success = run_script(script)
        if not success:
            print("Ensemble training aborted due to component failure.")
            sys.exit(1)
            
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TRAINING COMPLETE")
    print(f"Total Duration: {total_time:.2f} hours")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    full_train()
