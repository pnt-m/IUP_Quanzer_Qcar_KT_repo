#!/usr/bin/env bash
# ============================================================
# QCar1 Subsystem Health Check Runner
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "[INFO] Running QCar1 Integrated Subsystem Health Check..."
sudo PYTHONPATH="${PYTHONPATH}" python3 test_all_components.py "$@"
