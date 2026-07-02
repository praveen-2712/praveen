import os
import cv2
import numpy as np
import random
import shutil

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(os.path.dirname(base_dir), "mpv2", "data", "train", "no_tumor")
TARGET_ASSETS = os.path.join(base_dir, "assets")
GOLDEN_NAME = "golden_reference.jpg"

def calculate_histogram(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # 256 bins for 0-255 pixels
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def find_golden():
    files = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(files) < 50:
        samples = files
    else:
        samples = random.sample(files, 50)

    histograms = []
    valid_samples = []
    for s in samples:
        h = calculate_histogram(s)
        if h is not None:
            histograms.append(h)
            valid_samples.append(s)

    if not histograms:
        print("No valid histograms found.")
        return

    # Calculate Mean Histogram
    mean_hist = np.mean(histograms, axis=0)

    # Find file with lowest Bhattacharyya distance to mean
    min_dist = float('inf')
    golden_path = None

    for i, h in enumerate(histograms):
        # Bhattacharyya distance in OpenCV: cv2.compareHist returns 0 for identical, 1 for completely different
        # It's actually: d(H1,H2) = sqrt(1 - sum(sqrt(H1*H2)))
        dist = cv2.compareHist(mean_hist, h, cv2.HISTCMP_BHATTACHARYYA)
        if dist < min_dist:
            min_dist = dist
            golden_path = valid_samples[i]

    if golden_path:
        print(f"Golden Reference Identified: {golden_path}")
        print(f"Bhattacharyya Distance: {min_dist}")
        
        target_path = os.path.join(TARGET_ASSETS, GOLDEN_NAME)
        shutil.copy(golden_path, target_path)
        print(f"Copied to {target_path}")
        
        # Also save the path in a config file for inference engine
        with open(os.path.join(TARGET_ASSETS, "GOLDEN_REFERENCE.txt"), "w") as f:
            f.write(golden_path)
    else:
        print("Failed to identify golden reference.")

if __name__ == "__main__":
    if not os.path.exists(TARGET_ASSETS):
        os.makedirs(TARGET_ASSETS)
    find_golden()
