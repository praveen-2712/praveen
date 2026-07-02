@echo off
echo ============================================================
echo   Neurologix Pro V3 - Complete System Overhaul Training
echo   Running all 3 models sequentially (YOLO, U-Net, Classifier)
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Training YOLOv8 Tumor Detector...
echo --------------------------------------------------------
.\venv\Scripts\python src\train_yolo.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] YOLO training encountered an error. Continuing...
)
echo.

echo [2/3] Training Coarse-to-Fine U-Net Segmentor...
echo --------------------------------------------------------
.\venv\Scripts\python src\train_unet.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] U-Net training encountered an error. Continuing...
)
echo.

echo [3/3] Training Attention-Guided Dual-Branch Classifier...
echo --------------------------------------------------------
.\venv\Scripts\python src\train_classifier.py --single_channel
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Classifier training encountered an error.
)
echo.

echo ============================================================
echo   All training complete! Check weights\ folder for outputs.
echo ============================================================
pause
