Write-Host "Starting Complete Neurologix Pro V3 Overhaul Training..." -ForegroundColor Cyan

Write-Host "`n[1/3] Training YOLOv8 Tumor Detector..." -ForegroundColor Yellow
.\venv\Scripts\python src\train_yolo.py

Write-Host "`n[2/3] Training Coarse-to-Fine U-Net Segmentor..." -ForegroundColor Yellow
.\venv\Scripts\python src\train_unet.py

Write-Host "`n[3/3] Training Attention-Guided ViT Classifier..." -ForegroundColor Yellow
.\venv\Scripts\python src\train_classifier.py

Write-Host "`nAll models trained successfully!" -ForegroundColor Green
