import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import precompute_yolo_boxes

if __name__ == "__main__":
    train_dir = "data/classification/Training"
    yolo_ckpt = "models/yolo_runs/tumor_detector/weights/best.pt"
    out_json  = "yolo_boxes.json"
    
    if not os.path.exists(yolo_ckpt):
        print(f"Error: YOLO checkpoint not found at {yolo_ckpt}")
        print("Please ensure YOLO detection model is trained first.")
        sys.exit(1)
        
    print(f"Pre-computing YOLO boxes for {train_dir}...")
    # Dropping confidence threshold to 0.05 to see if detections exist but are just low confidence
    precompute_yolo_boxes(train_dir, yolo_ckpt, out_json, conf_threshold=0.05)
    print("Precomputation complete!")
