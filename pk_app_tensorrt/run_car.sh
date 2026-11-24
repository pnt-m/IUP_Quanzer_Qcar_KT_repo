#!/usr/bin/env bash
# ============================================================
# QCar1 Real-Time Execution Runner
# ============================================================
# Usage:
#   ./run_car.sh --no-display   (Recommended for SSH headless)
#   ./run_car.sh                (If GUI/X11 display is connected)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Launch main pedestrian detection application
echo "[INFO] Starting QCar1 TensorRT Pedestrian Detection..."
sudo PYTHONPATH="${PYTHONPATH}" python3 qcar1_pedestrian.py "$@"
