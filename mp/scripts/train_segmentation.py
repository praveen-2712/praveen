import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import BrainTumorDataset

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
            
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f*2, f))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.final_conv(x)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return bce + dice_loss

def calculate_metrics(preds, masks, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return dice.item(), iou.item()

def main():
    print("Setting up Segmentation Training Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data/processed/labels.csv")
    img_dir = os.path.join(base_dir, "data/processed/images")
    mask_dir = os.path.join(base_dir, "data/processed/masks")
    
    if not os.path.exists(csv_path):
        print("Dataset not found. Please run download.py and preprocess.py first.")
        # Setup dummy skeleton so it never crashes
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame({"image_id": [], "mask_id": [], "label": [], "source": []}).to_csv(csv_path, index=False)
        return
        
    full_ds = BrainTumorDataset(csv_path, img_dir, mask_dir=mask_dir, is_train=True, task="segmentation")
    if len(full_ds) == 0:
        print("Empty dataset.")
        return
        
    train_len = int(0.7 * len(full_ds))
    val_len = int(0.15 * len(full_ds))
    test_len = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    epochs = 50
    patience = 7
    best_loss = float('inf')
    epochs_no_improve = 0
    t_losses, v_losses, v_dices, v_ious = [], [], [], []
    
    print("Beginning PyTorch U-Net Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                d, iou = calculate_metrics(outputs, masks)
                val_dice += d
                val_iou += iou
                
        t_l = train_loss/max(1, len(train_loader))
        v_l = val_loss/max(1, len(val_loader))
        v_d = val_dice/max(1, len(val_loader))
        v_i = val_iou/max(1, len(val_loader))
        
        t_losses.append(t_l)
        v_losses.append(v_l)
        v_dices.append(v_d)
        v_ious.append(v_i)
        
        print(f"Epoch {epoch+1}/{epochs} | T-Loss: {t_l:.4f} | V-Loss: {v_l:.4f} | Dice: {v_d:.4f} | IoU: {v_i:.4f}")
        
        scheduler.step(v_l)
        if v_l < best_loss:
            best_loss = v_l
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(base_dir, 'segmentation_model.pth'))
            print(" -> Saving best model!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break
                
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_losses, label='Train Loss')
    plt.plot(v_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(v_dices, label='Val Dice')
    plt.plot(v_ious, label='Val IoU')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'segmentation_learning_curves.png'))
    print("Training Complete. Artifacts saved.")

if __name__ == "__main__":
    main()
