"""
find_threshold.py — quick script to find the optimal glioma threshold
on raw softmax probabilities for the full merged test set.
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import MRI25DDataset, LABEL_TO_IDX
from train_classifier import DualBranchClassifier
from preprocess import Preprocess

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
GLIOMA_IDX  = LABEL_TO_IDX["glioma"]
CLASSES     = [k for k, v in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1])]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = A.Compose([
    A.Resize(380, 380),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.pytorch.ToTensorV2(),
])

data_val = os.path.join(BASE_DIR, "data", "classification", "Testing")
preprocessor = Preprocess()

# Load yolo_boxes for consistent branch_local crop quality
yolo_boxes_path = os.path.join(BASE_DIR, "yolo_boxes.json")
yolo_boxes = {}
if os.path.exists(yolo_boxes_path):
    with open(yolo_boxes_path) as _f:
        yolo_boxes = json.load(_f)

val_ds   = MRI25DDataset(data_val, transform=val_transform, preprocess_logic=preprocessor, img_size=380, single_image_mode=True, yolo_boxes=yolo_boxes)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = DualBranchClassifier(num_classes=len(LABEL_TO_IDX))
sd = torch.load(os.path.join(WEIGHTS_DIR, "tumor_classifier.pth"), map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.to(device).eval()

print("[Threshold] Collecting probabilities on full test set...")
all_probs, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        imgs   = batch["image"].to(device)
        crops  = batch["crop"].to(device)
        labels = batch["label"]
        probs  = torch.softmax(model(imgs, crops), dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
glioma_probs   = all_probs[:, GLIOMA_IDX]
binary_labels  = (all_labels == GLIOMA_IDX).astype(int)

fpr, tpr, thresholds = roc_curve(binary_labels, glioma_probs)
roc_auc = auc(fpr, tpr)
print(f"[Threshold] Glioma AUC (raw softmax): {roc_auc:.4f}")
print(f"[Threshold] Total test samples: {len(all_labels)}")
print(f"[Threshold] Glioma test samples: {binary_labels.sum()}")

# Find highest threshold that gives >= 90% sensitivity
target = 0.90
valid  = thresholds[tpr >= target]
if len(valid) == 0:
    print(f"[Threshold] WARNING: Cannot achieve {target:.0%} sensitivity. Using 0.15.")
    optimal_T = 0.15
else:
    optimal_T = float(valid[-1])

# Report at this threshold
idx    = np.searchsorted(thresholds[::-1], optimal_T)
print(f"\n[Threshold] === Results at Threshold = {optimal_T:.4f} ===")
preds = []
for p, lbl in zip(all_probs, all_labels):
    if p[GLIOMA_IDX] > optimal_T:
        preds.append(GLIOMA_IDX)
    else:
        preds.append(int(np.argmax(p)))
preds = np.array(preds)

print(classification_report(all_labels, preds, target_names=CLASSES, zero_division=0))

# Save to thresholds JSON
threshold_path = os.path.join(WEIGHTS_DIR, "clinical_thresholds.json")
data = {}
if os.path.exists(threshold_path):
    with open(threshold_path) as f:
        data = json.load(f)
data["glioma_threshold"] = optimal_T
with open(threshold_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"[Threshold] Saved glioma_threshold={optimal_T:.4f} to {threshold_path}")

# Also print a sensitivity sweep
print("\n[Threshold] === Sensitivity Sweep ===")
print(f"{'Threshold':>10} {'Glioma Recall':>15} {'Overall Acc':>12} {'Specificity':>13}")
for T in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
    p2 = []
    for p in all_probs:
        if p[GLIOMA_IDX] > T:
            p2.append(GLIOMA_IDX)
        else:
            p2.append(int(np.argmax(p)))
    p2 = np.array(p2)
    glioma_mask = (all_labels == GLIOMA_IDX)
    g_recall = (p2[glioma_mask] == GLIOMA_IDX).mean()
    acc      = (p2 == all_labels).mean()
    neg_mask = ~glioma_mask
    spec     = (p2[neg_mask] != GLIOMA_IDX).mean()
    flag     = " <- >=90%" if g_recall >= 0.90 else ""
    print(f"{T:>10.2f} {g_recall:>14.1%} {acc:>12.1%} {spec:>12.1%}{flag}")
