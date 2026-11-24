#!/usr/bin/env bash
# ==============================================================================
# Helper Script: Re-export YOLOv5 at Opset 12 for Jetson TX2 / ONNX Runtime 1.10
# ==============================================================================
# Run this on a development workstation (Linux/macOS/WSL) with internet access.
# Output will be: yolov5n_opset12.onnx
# ==============================================================================

set -euo pipefail

WORK_DIR="$(mktemp -d -t yolov5_export_XXXXXX)"
echo "[INFO] Working directory: ${WORK_DIR}"
cd "${WORK_DIR}"

echo "[INFO] Cloning YOLOv5 release v6.1..."
git clone --depth 1 --branch v6.1 https://github.com/ultralytics/yolov5.git
cd yolov5

echo "[INFO] Installing minimal export dependencies..."
pip install --upgrade pip
pip install -r requirements.txt onnx>=1.9.0

echo "[INFO] Exporting yolov5n.pt to ONNX with opset 12..."
python3 export.py \
    --weights yolov5n.pt \
    --include onnx \
    --opset 12 \
    --imgsz 640

OUTPUT_FILE="${WORK_DIR}/yolov5/yolov5n.onnx"

if [ -f "${OUTPUT_FILE}" ]; then
    cp "${OUTPUT_FILE}" ./yolov5n_opset12.onnx
    echo "=============================================================================="
    echo "[PASS] Successfully exported: $(pwd)/yolov5n_opset12.onnx"
    echo "Transfer to QCar1 using:"
    echo "  scp $(pwd)/yolov5n_opset12.onnx <user>@<qcar1-ip>:/path/to/models/yolov5n.onnx"
    echo "=============================================================================="
else
    echo "[FAIL] Export failed. Check output logs above."
    exit 1
fi
