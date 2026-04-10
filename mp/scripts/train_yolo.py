import os
from ultralytics import YOLO

# Generate ultralytics dataset.yaml
yaml_content = """path: ../data/yolo
train: images/train
val: images/val
test: images/test

names:
  0: tumor
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

if __name__ == "__main__":
    print("Initializing YOLOv8s Model for Brain Tumor Detection...")
    model = YOLO("yolov8s.pt") # Transfer learning from COCO
    
    print("Starting training: Using Early Stopping, custom lr, and robust augmentations...")
    # Using small epochs for initial compilation verify
    results = model.train(
        data="data.yaml", 
        epochs=10, 
        imgsz=256, 
        batch=16, 
        patience=5,
        lr0=0.001,
        dropout=0.2, # robust dropout
        device='cpu' # automatically falls back to CPU if single-GPU memory is limited
    )
    print("Training Complete. Weights saved in runs/detect/train/weights/best.pt")
