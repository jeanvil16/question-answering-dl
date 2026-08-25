@echo off
echo ============================================
echo  QA System - Full Setup (Windows)
echo ============================================
echo.
echo [1/3] Installing PyTorch (CPU build)...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.
echo [2/3] Installing Python dependencies...
python -m pip install -r requirements.txt
echo.
echo [3/3] Installing frontend dependencies...
cd frontend && npm install && cd ..
echo.
echo ============================================
echo  Setup complete!
echo.
echo  To train the model:  python training\train.py
echo  To start backend:    run_backend.bat
echo  To start frontend:   run_frontend.bat
echo ============================================
