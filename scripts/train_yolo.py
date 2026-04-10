import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gpu_config import get_nvidia_device
from ultralytics import YOLO

# YOLO specifically cannot run on DirectML due to missing torch.unique operator implementation
device = "cpu"
print(f"Using device: {device} (YOLO DirectML Hardware Limitation)")


# Load model
model = YOLO('yolov8n.pt')

# Train
# YOLOv8 might still struggle with the 'privateuseone:0' string in some parts of its engine.
# We will pass the device if possible, or use 'cpu' if DirectML is not natively mapped to '0'.
try:
    model.train(
        data=os.path.abspath('data/bootstrapped/yolo/data.yaml'),
        epochs=40,
        imgsz=384,
        device=device,
        batch=32,
        amp=False,
        name='neurologix_yolo_bt'
    )
except Exception as e:
    print(f"DirectML training error: {e}")
    print("Attempting CPU fallback for detection head (lightweight)...")
    model.train(
        data=os.path.abspath('data/bootstrapped/yolo/data.yaml'),
        epochs=40,
        imgsz=384,
        device='cpu',
        batch=4,
        name='neurologix_yolo_bt_cpu'
    )
