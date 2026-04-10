import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# EfficientNet-B2 optimal size is 260
IMG_SIZE = (260, 260)

def get_inference_transforms():
    """
    Advanced medical preprocessing for inference.
    Includes CLAHE for contrast and Z-score normalization.
    """
    return A.Compose([
        A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def crop_brain_contour(image_np):
    """
    Standard MRI brain cropping via contour detection.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        import imutils
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0: return image_np
        c = max(cnts, key=cv2.contourArea)
        ext_left  = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top   = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot   = tuple(c[c[:, :, 1].argmax()][0])
        cropped = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        return cropped if (cropped.shape[0] > 10 and cropped.shape[1] > 10) else image_np
    except:
        return image_np

def load_and_preprocess_image(file_storage):
    """
    PyTorch pipeline for MRI preprocessing.
    """
    pil_image = Image.open(file_storage).convert("RGB")
    image_np = np.array(pil_image)
    
    # 1. Brain region crop
    image_np = crop_brain_contour(image_np)
    
    # 2. Resize original for display
    pil_display = Image.fromarray(cv2.resize(image_np, (IMG_SIZE[1], IMG_SIZE[0])))
    
    # 3. Augmentations & Normalization
    transform = get_inference_transforms()
    augmented = transform(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0) # (1, 3, 380, 380)
    
    return image_tensor, pil_display
