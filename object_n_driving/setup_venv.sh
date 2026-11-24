#!/usr/bin/env bash
# ============================================================
# QCar1 TensorRT - Virtual Environment Setup (Python 3.6)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "QCar1 TensorRT Virtual Environment Setup"
echo "============================================================"

# 1. Install PyCUDA if not already present on Jetson
if ! dpkg -s python3-pycuda >/dev/null 2>&1; then
    echo "[INFO] Installing system python3-pycuda..."
    sudo apt-get update && sudo apt-get install -y python3-pycuda
else
    echo "[PASS] System python3-pycuda is already installed."
fi

# 2. Create Python 3.6 venv with system site-packages
# Crucial: --system-site-packages allows access to pre-installed
# TensorRT, OpenCV, and Quanser PAL/HAL packages!
if [ ! -d "venv" ]; then
    echo "[INFO] Creating venv with --system-site-packages..."
    python3 -m venv --system-site-packages venv
else
    echo "[INFO] Existing venv detected."
fi

source venv/bin/activate

# 3. Install PyYAML
echo "[INFO] Installing project dependencies..."
pip install PyYAML

echo "============================================================"
echo "[PASS] Setup complete!"
echo "       To activate: source venv/bin/activate"
echo "       To check   : python3 test_environment.py"
echo "============================================================"
