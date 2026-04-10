# Brain Tumor ML Pipeline Instructions

Follow these steps sequentially to build your dataset pipeline and train the models.

### Step 1: Download Datasets
Ensure your `kaggle.json` is configured globally in your system. Then run:
```bash
py scripts/download.py
```
This automatically leverages the Kaggle API to fetch the core datasets into `data/raw/kaggle/` and prints instructions for acquiring the NIfTI arrays for BraTS.

### Step 2: Preprocess & Standardize
Run the preprocessing script to homogenize all datasets to 256x256 PNGs, enforce uniform 3-channel depth, map classification labels to numeric arrays (0: glioma, 1: meningioma, etc.), and enforce binary morphology on segmentation masks.
```bash
py scripts/preprocess.py
```
*Outputs are mapped correctly into `data/processed/images/`, `data/processed/masks/`, and fully indexed in `data/processed/labels.csv`.*

### Step 3: Train the Classification Model
Execute the PyTorch dataloaders (with dynamic Albumentations) and train the `EfficientNet-B4` model utilizing CrossEntropyLoss, AdamW, and ReduceLROnPlateau.
```bash
py scripts/train_classifier.py
```
*Outputs: `classification_model.pth` and metric plots `classification_learning_curves.png`.*

### Step 4: Train the Semantic Segmentation Model
Train the `U-Net` architecture using a custom combined `DiceLoss + BCEWithLogitsLoss`. Progress is monitored via raw Dice coefficient and Intersection-over-Union (IoU) tracking.
```bash
py scripts/train_segmentation.py
```
*Outputs: `segmentation_model.pth` and metric plots `segmentation_learning_curves.png`.*
