import os
import zipfile
import subprocess

def setup_dirs():
    dirs = [
        "data/raw/kaggle",
        "data/raw/brats",
        "data/raw/figshare",
        "data/processed/images",
        "data/processed/masks"
    ]
    for d in dirs:
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), d), exist_ok=True)

def download_kaggle():
    print("Downloading Kaggle Dataset...")
    # Using a common Kaggle brain tumor dataset as an example
    dataset_name = "sartajbhuvaji/brain-tumor-classification-mri"
    kaggle_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/raw/kaggle")
    
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", kaggle_dir], check=True)
        zip_path = os.path.join(kaggle_dir, "brain-tumor-classification-mri.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(kaggle_dir)
            os.remove(zip_path)
        print("Kaggle dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Kaggle download failed (Requires kaggle.json configured): {e}")

def manual_instructions():
    print("\n--- MANUAL DOWNLOAD INSTRUCTIONS ---")
    print("1. BraTS Dataset: Brain Tumor Segmentation challenges require registration.")
    print("   Go to: http://braintumorsegmentation.org/ or use Synapse to download.")
    print("   Extract the NIfTI masks/images into: data/raw/brats/")
    print("\n2. Figshare Dataset: Contains meningioma, glioma, and pituitary tumors.")
    print("   Go to: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
    print("   Extract the .mat files into: data/raw/figshare/")
    print("------------------------------------\n")

if __name__ == "__main__":
    setup_dirs()
    download_kaggle()
    manual_instructions()
