import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
from dataset import BrainTumorDataset

class MetricsHistory:
    def __init__(self):
        self.train_losses, self.val_losses = [], []
        self.val_accs = []
        
    def plot(self, save_path):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accs, label='Val Acc')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig(save_path)
        plt.close()

def main():
    print("Setting up Classification Training Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data/processed/labels.csv")
    img_dir = os.path.join(base_dir, "data/processed/images")
    
    if not os.path.exists(csv_path):
        print("Dataset not found. Please run download.py and preprocess.py first.")
        # Setup dummy skeleton so the script never crashes even if no data downloaded.
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame({"image_id": [], "mask_id": [], "label": [], "source": []}).to_csv(csv_path, index=False)
        return
        
    full_ds = BrainTumorDataset(csv_path, img_dir, is_train=True, task="classification")
    if len(full_ds) == 0:
        print("Empty dataset.")
        return
        
    # 70/15/15 split
    train_len = int(0.7 * len(full_ds))
    val_len = int(0.15 * len(full_ds))
    test_len = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Configure 4 classes with Dropout / BatchNorm
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.BatchNorm1d(num_features=num_ftrs),
        nn.Linear(num_ftrs, 4)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    epochs = 50
    patience = 7
    best_f1 = 0.0
    epochs_no_improve = 0
    history = MetricsHistory()
    
    print("Beginning Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        t_loss = train_loss/max(1, len(train_loader))
        v_loss = val_loss/max(1, len(val_loader))
        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        history.train_losses.append(t_loss)
        history.val_losses.append(v_loss)
        history.val_accs.append(acc)
        
        print(f"Epoch {epoch+1}/{epochs} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(base_dir, 'classification_model.pth'))
            print(" -> Saving best model!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break
                
    history.plot(os.path.join(base_dir, 'classification_learning_curves.png'))
    print("Training finished. Artifacts saved.")

if __name__ == "__main__":
    main()
