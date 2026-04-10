import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, is_train=True, task="classification"):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.task = task
        
        # Albumentations pipeline
        if is_train:
            self.transform = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((256, 256, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.task == "classification":
            label = int(row['label'])
            augmented = self.transform(image=image)
            return augmented['image'], torch.tensor(label, dtype=torch.long)
            
        elif self.task == "segmentation":
            mask_path = os.path.join(self.mask_dir, row['mask_id'])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros((256, 256), dtype=np.uint8)
            mask = mask / 255.0 # Binary
            
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
            return image, mask
