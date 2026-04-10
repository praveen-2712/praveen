import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from tqdm import tqdm
import json
# Config
DATA_DIR = "data"
IMG_SIZE = 260
BATCH_SIZE = 16 
EPOCHS = 40
MODEL_PATH = "models/tumor_classifier.pth"
LABEL_MAP_PATH = "models/label_map.json"

# Custom Medical Dataset for Albumentations (NumPy bridging)
class MedicalDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, target

# Focal Loss Implementation for Difficult Cases
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def get_train_transforms():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.5),
        A.GridDistortion(p=0.3), # Medical mesh distortion
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def train():
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.gpu_config import get_nvidia_device
    device = get_nvidia_device()
    print(f"Training on: {device} (DirectML NVIDIA Forced)")
    
    # ... (Data Loading logic same) ...
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    train_data = MedicalDataset(train_dir, transform=get_train_transforms())
    val_data = datasets.ImageFolder(val_dir, transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]))
    
    class_counts = np.array([len([x for x in train_data.samples if x[1] == i]) for i in range(len(train_data.classes))])
    class_weights = 1. / class_counts
    sampler = WeightedRandomSampler(np.array([class_weights[t] for _, t in train_data.samples]), len(train_data.samples))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model Setup
    print("Loading EfficientNet-B2...")
    model = models.efficientnet_b2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
    
    # FINE-TUNING MODE: Load existing weights if available
    if os.path.exists(MODEL_PATH):
        print(f"Found existing weights: {MODEL_PATH}. Entering Fine-Tuning Mode.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    model = model.to(device)
    
    criterion = FocalLoss(gamma=2.5) # Focal Loss for accuracy boost
    optimizer = optim.AdamW(model.parameters(), lr=5e-5 if os.path.exists(MODEL_PATH) else 1e-4) # Lower LR for fine-tuning
    # ... rest of training loop same ...
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision
    
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
            
            pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100.*correct/total})
            
        # Validation
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
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            print(f"Saving best model ({val_acc:.2f}%)...")
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    # Fix for multiprocessing on Windows
    train()
