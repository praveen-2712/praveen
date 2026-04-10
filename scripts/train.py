import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
import cv2
import json
import timm
from tqdm import tqdm
import sys

# Import our unified preprocessing so the model trains on exactly what inference sees
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import get_train_transforms, get_inference_transforms, crop_brain_contour, IMG_SIZE
from utils.gpu_config import get_nvidia_device

# Config
DATA_DIR = "data"
BATCH_SIZE = 16 # Increased batch size since we removed the complex Hybrid branches
EPOCHS = 20 # 20 epochs is more than enough for EfficientNet on small datasets
MODEL_PATH = "models/tumor_classifier.pth"
LABEL_MAP_PATH = "models/label_map.json"

class MedicalDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Read with OpenCV (BGR)
        image = cv2.imread(path)
        # Convert to RGB (Ultralytics and PIL standard)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # APPLY EXACT SAME CROP LOGIC AS INFERENCE
        image = crop_brain_contour(image)
        
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target

class EfficientNetTumorModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetTumorModel, self).__init__()
        # SOTA highly accurate feature extractor
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def train():
    device = get_nvidia_device()
    print(f"Training on: {device} | Using Unified Architecture: EfficientNetB4 Grad-CAM Extractor")
    
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    # Validation uses inference transforms (but we must re-wrap using A.Compose compatible Dataset)
    val_transform = get_inference_transforms()
    
    train_data = MedicalDataset(train_dir, transform=get_train_transforms())
    val_data = MedicalDataset(val_dir, transform=val_transform)
    
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(train_data.class_to_idx, f)
        
    print(f"Training Classes Found: {train_data.class_to_idx}")
    
    # Class weights for mild imbalance
    class_counts = np.array([len([x for x in train_data.samples if x[1] == i]) for i in range(len(train_data.classes))])
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for _, t in train_data.samples])
    sampler = WeightedRandomSampler(sample_weights, len(train_data.samples))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = EfficientNetTumorModel(len(train_data.classes))
    model = model.to(device)
    
    # Using standard high-stability CrossEntropy
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=3e-4) # Higher initial LR for new architecture
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() 
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(pbar):.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            print(f"-> Saving new best model ({val_acc:.2f}%)")
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()
