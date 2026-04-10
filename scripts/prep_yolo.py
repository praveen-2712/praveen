import os
import yaml

# YOLOv8 data configuration
data_config = {
    'train': os.path.abspath('data/bootstrapped/yolo/images'),
    'val': os.path.abspath('data/bootstrapped/yolo/images'), 
    'nc': 4,
    'names': ['glioma', 'meningioma', 'no_tumor', 'pituitary']
}

os.makedirs('models', exist_ok=True)
with open('data/bootstrapped/yolo/data.yaml', 'w') as f:
    yaml.dump(data_config, f)

print("YOLOv8 configuration generated.")
