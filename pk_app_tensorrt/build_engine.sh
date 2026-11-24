#!/usr/bin/env bash
# ============================================================
# QCar1 TensorRT Engine Builder (Jetson TX2)
# ============================================================
# Compiles models/yolov5n.onnx into an optimized TensorRT FP16
# execution engine using NVIDIA's built-in trtexec utility.
#
# Run directly on the QCar:
#     chmod +x build_engine.sh
#     ./build_engine.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ONNX_MODEL="models/yolov5n.onnx"
ENGINE_OUTPUT="models/yolov5n_fp16.engine"
WORKSPACE_MB=1024

echo "============================================================"
echo "QCar1 TensorRT FP16 Engine Build"
echo "============================================================"
echo "Target Device: NVIDIA Jetson TX2 (ARM64 / sm_62)"
echo "Input ONNX   : ${ONNX_MODEL}"
echo "Output Engine: ${ENGINE_OUTPUT}"
echo "Precision    : FP16 (Half-Precision GPU Acceleration)"
echo "Workspace    : ${WORKSPACE_MB} MB"
echo "============================================================"

# 1. Verify trtexec compiler exists
if [ ! -f "${TRTEXEC}" ]; then
    echo "[FAIL] trtexec compiler not found at ${TRTEXEC}."
    echo "       Ensure NVIDIA TensorRT 8.0+ is installed on this Jetson TX2."
    exit 1
fi

# 2. Verify input ONNX model exists
if [ ! -f "${ONNX_MODEL}" ]; then
    echo "[FAIL] Input ONNX model not found at ${ONNX_MODEL}."
    echo "       Please ensure yolov5n.onnx is present in the models/ directory."
    exit 1
fi

echo "[INFO] Starting TensorRT compilation. This takes 2 to 4 minutes on the Jetson TX2..."
echo "[INFO] Running: ${TRTEXEC} --onnx=${ONNX_MODEL} --saveEngine=${ENGINE_OUTPUT} --workspace=${WORKSPACE_MB} --fp16"

"${TRTEXEC}" \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${ENGINE_OUTPUT}" \
    --workspace="${WORKSPACE_MB}" \
    --fp16

if [ -f "${ENGINE_OUTPUT}" ]; then
    ENGINE_SIZE=$(ls -lh "${ENGINE_OUTPUT}" | awk '{print $5}')
    echo "============================================================"
    echo "[PASS] TensorRT engine generated successfully!"
    echo "       Engine path : ${ENGINE_OUTPUT}"
    echo "       Engine size : ${ENGINE_SIZE}"
    echo "============================================================"
    echo "Next steps:"
    echo "  1. Test engine performance : python3 test_tensorrt.py"
    echo "  2. Run live on the car     : python3 qcar1_pedestrian.py --no-display"
    echo "============================================================"
else
    echo "[FAIL] Engine compilation finished but ${ENGINE_OUTPUT} was not found."
    exit 1
fi
