#!/usr/bin/env bash
# ============================================================
# QCar1 Autonomous Driving Runner
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "[INFO] Launching QCar1 Autonomous Driving with TensorRT..."
sudo PYTHONPATH="${PYTHONPATH}" python3 qcar1_autonomous_driving.py "$@"
