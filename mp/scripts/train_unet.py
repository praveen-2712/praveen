import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

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
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            
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
                import torchvision.transforms.functional as TF
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return torch.sigmoid(self.final_conv(x))

class BrainTumorDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir) if os.path.exists(img_dir) else []
        self.transform = transform
        
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_mask.png')
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add a mock mask generation fallback because not all paths guarantees file
        if not os.path.exists(mask_path):
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, 0)
        
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1.0 # normalize
        
        image = cv2.resize(image, (256, 256))
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        mask = cv2.resize(mask, (256, 256))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
        return image, mask

if __name__ == "__main__":
    print("Initiating U-Net Semantic Segmentation Training from Scratch...")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'masks')
    
    train_dir = os.path.join(data_dir, 'train', 'images')
    if not os.path.exists(train_dir):
        print(f"Dataset not found at {train_dir}. Please run download_data.py first.")
        exit(1)
        
    train_ds = BrainTumorDataset(train_dir, os.path.join(data_dir, 'train', 'masks'), transform=True)
    val_ds = BrainTumorDataset(os.path.join(data_dir, 'val', 'images'), os.path.join(data_dir, 'val', 'masks'), transform=True)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 1
    print(f"Beginning U-Net Training Loop on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/max(1, len(train_loader)):.4f}, Val Loss: {val_loss/max(1, len(val_loader)):.4f}")

    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/unet_brain.pth')
    print("Model saved to weights/unet_brain.pth")
