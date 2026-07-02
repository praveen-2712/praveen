import os
import sys
import zipfile
import shutil
import subprocess


def setup_kaggle():
    """
    Auto-downloads and extracts the brain tumor datasets from Kaggle.
    Requires Kaggle credentials configured in environment variables or ~/.kaggle/kaggle.json.
    """
    print("=" * 60)
    print("        Neurologix Dataset Setup Automation Tool")
    print("=" * 60)

    # 1. Verify Kaggle Credentials
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_file = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    
    if not (has_env or has_file):
        print("[KAGGLE CREDENTIAL ERROR]")
        print("To download the training datasets, you MUST configure Kaggle API credentials.")
        print("Please do one of the following:")
        print("  1. Place your 'kaggle.json' API key file at C:\\Users\\Praveen\\.kaggle\\kaggle.json")
        print("  2. Set environment variables before running this tool:")
        print("     $env:KAGGLE_USERNAME = \"yourusername\"")
        print("     $env:KAGGLE_KEY = \"your_api_key_here\"")
        print("-" * 60)
        print("Exiting setup.")
        sys.exit(1)

    try:
        import kaggle
    except ImportError:
        print("[App] Installing kaggle python API library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    os.makedirs("./data", exist_ok=True)

    # 2. Download and Set Up Classification Dataset (Massive Aggregation)
    class_dir = "./data/classification"
    temp_dir = "./data/temp_extract"
    
    datasets = [
        "sartajbhuvaji/brain-tumor-classification-mri",
        "masoudnickparvar/brain-tumor-mri-dataset",
        "tombackert/brain-tumor-mri-data",
        "denizkavi1/brain-tumor"
    ]
    
    LABEL_MAP = {
        "glioma_tumor": "glioma",
        "glioma": "glioma",
        "meningioma_tumor": "meningioma",
        "meningioma": "meningioma",
        "no_tumor": "notumor",
        "notumor": "notumor",
        "normal": "notumor",
        "pituitary_tumor": "pituitary",
        "pituitary": "pituitary",
        "pituitary_adenoma": "pituitary",
        "1": "meningioma",
        "2": "glioma",
        "3": "pituitary"
    }
    
    print(f"\n--- 1/2 Aggregating Multiple Classification Datasets (>12k images) ---")
    
    # We will always run the aggregation logic if the dataset isn't fully compiled yet
    # Or we can just build it if it doesn't exist
    if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
        os.makedirs(class_dir, exist_ok=True)
        os.makedirs(os.path.join(class_dir, "Training"), exist_ok=True)
        os.makedirs(os.path.join(class_dir, "Testing"), exist_ok=True)
        
        for class_name in set(LABEL_MAP.values()):
            os.makedirs(os.path.join(class_dir, "Training", class_name), exist_ok=True)
            os.makedirs(os.path.join(class_dir, "Testing", class_name), exist_ok=True)

        for ds in datasets:
            ds_name = ds.split("/")[1]
            extract_path = os.path.join(temp_dir, ds_name)
            if not os.path.exists(extract_path):
                print(f"[Download] Fetching {ds} ...")
                try:
                    kaggle.api.dataset_download_files(ds, path=extract_path, unzip=True)
                except Exception as e:
                    print(f"[Warning] Failed to download {ds}: {e}")
                    continue
            
            # Walk and merge
            print(f"[Merge] Processing {ds_name}...")
            copied_count = 0
            for root, _, files in os.walk(extract_path):
                parent_dir = os.path.basename(root).lower().strip()
                # Check if this directory is one of our classes
                if parent_dir in LABEL_MAP:
                    canon_class = LABEL_MAP[parent_dir]
                    
                    # Determine split
                    lower_root = root.lower()
                    if "test" in lower_root or "val" in lower_root:
                        split = "Testing"
                    else:
                        split = "Training"
                        
                    dest_dir = os.path.join(class_dir, split, canon_class)
                    
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src_file = os.path.join(root, f)
                            # Create a unique filename to avoid collisions
                            import uuid
                            unique_id = uuid.uuid4().hex[:6]
                            dest_file = os.path.join(dest_dir, f"{ds_name}_{unique_id}_{f}")
                            shutil.copy2(src_file, dest_file)
                            copied_count += 1
            print(f"        -> Merged {copied_count} images.")
            
        print("[Cleanup] Removing temporary extractions...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[Warning] Cleanup failed: {e}")
            
        print("[Success] Massive classification dataset aggregated successfully.")
    else:
        print("\n[Skip] Classification dataset already aggregated.")

    # 3. Download and Set Up Segmentation (BraTS) Dataset
    brats_dir = "./data/brats"
    
    if not os.path.exists(brats_dir):
        print("\n--- 2/2 Downloading awsaf49/brats20-dataset-training-validation ---")
        print("Note: This is a large clinical dataset (~3.5 GB). Downloading may take a few minutes...")
        
        try:
            kaggle.api.dataset_download_files(
                "awsaf49/brats20-dataset-training-validation",
                path=brats_dir,
                unzip=True
            )
            print("[Success] BraTS dataset downloaded and extracted.")
        except Exception as e:
            print(f"[Error] Failed to download BraTS dataset: {e}")
            sys.exit(1)
            
        # Standard folder inside is 'BraTS2020_TrainingData' or similar.
        # Let's verify and map it so extract_brats_slices.py can find it!
        # If it extracted directly into ./data/brats/, it should have Patient folders.
        print("[Success] BraTS dataset configured correctly.")
    else:
        print("\n[Skip] BraTS dataset already configured.")

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("Run the master training launcher: python train_all.py")
    print("=" * 60)

if __name__ == "__main__":
    setup_kaggle()
