#!/usr/bin/env bash
set -e
echo "============================================"
echo " QA System - Full Setup (Unix)"
echo "============================================"
echo
echo "[1/3] Installing PyTorch (CPU build)..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo
echo "[2/3] Installing Python dependencies..."
python3 -m pip install -r requirements.txt
echo
echo "[3/3] Installing frontend dependencies..."
cd frontend && npm install && cd ..
echo
echo "============================================"
echo " Setup complete!"
echo ""
echo "  To train the model:  python3 training/train.py"
echo "  To start backend:    ./run_backend.sh"
echo "  To start frontend:   ./run_frontend.sh"
echo "============================================"
