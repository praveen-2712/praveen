import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import torch_directml
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Config
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
IMG_SIZE = (256, 256)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gpu_config import get_nvidia_device
DEVICE = get_nvidia_device()
DATA_DIR = "data/bootstrapped/unet"
MODEL_SAVE_PATH = "models/unet_segmentor.pth"

class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx]) 
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.functional.to_tensor(mask.resize(IMG_SIZE))
            
        return image, mask

def train_unet():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BrainTumorSegmentationDataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)
    
    # Joint Loss: Dice + BCE
    criterion_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting U-Net Training for {EPOCHS} epochs on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.5 * criterion_dice(outputs, masks) + 0.5 * criterion_bce(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_unet()
