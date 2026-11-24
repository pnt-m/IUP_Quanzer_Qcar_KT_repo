#!/usr/bin/env bash
# ============================================================
# Build NVIDIA TensorRT FP16 Engine for best.onnx (Jetson TX2)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ONNX_PATH="models/best.onnx"
ENGINE_PATH="models/best_fp16.engine"
WORKSPACE_MB=1024

echo "============================================================"
echo "QCar1 TensorRT Engine Builder (best.onnx -> FP16)"
echo "============================================================"

if [ ! -f "${ONNX_PATH}" ]; then
    echo "[FAIL] Input model '${ONNX_PATH}' not found."
    exit 1
fi

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -x "${TRTEXEC}" ]; then
    TRTEXEC=$(which trtexec 2>/dev/null || true)
fi

if [ -z "${TRTEXEC}" ] || [ ! -x "${TRTEXEC}" ]; then
    echo "[FAIL] trtexec binary not found."
    exit 1
fi

echo "[INFO] Input Model : ${ONNX_PATH}"
echo "[INFO] Output Engine: ${ENGINE_PATH}"
echo "[INFO] Target GPU   : NVIDIA Jetson TX2 (sm_62)"
echo "[INFO] Precision    : FP16"
echo "[INFO] Workspace    : ${WORKSPACE_MB} MB"
echo "============================================================"

${TRTEXEC} \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --workspace=${WORKSPACE_MB} \
    --fp16

echo "============================================================"
if [ -f "${ENGINE_PATH}" ]; then
    SIZE=$(ls -lh "${ENGINE_PATH}" | awk '{print $5}')
    echo "[PASS] TensorRT engine generated successfully!"
    echo "       Engine path : ${ENGINE_PATH}"
    echo "       Engine size : ${SIZE}"
    echo "============================================================"
    echo "Next steps:"
    echo "  1. Test benchmark  : python3 test_tensorrt.py"
    echo "  2. Run car live    : python3 qcar1_detector.py --no-display"
    echo "============================================================"
else
    echo "[FAIL] Engine file was not generated."
    exit 1
fi
