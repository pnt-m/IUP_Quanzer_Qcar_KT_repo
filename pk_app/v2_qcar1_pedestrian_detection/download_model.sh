#!/usr/bin/env bash
# ============================================================
# QCar1 YOLO Pedestrian Detection - Model Downloader
# ============================================================
#
# Downloads a small YOLOv5-nano ONNX model (COCO-pretrained,
# 80 classes, class id 0 = "person") suitable for real-time-ish
# CPU inference on a Jetson TX2 via ONNX Runtime.
#
# This script does NOT touch the system Python/CUDA/TensorRT
# installation. It only downloads a file into ./models/.
#
# ------------------------------------------------------------
# MODEL INFORMATION (see README.md section 9 for full detail)
# ------------------------------------------------------------
#   Architecture : YOLOv5n (nano)
#   Source       : Ultralytics YOLOv5 (COCO pretrained)
#   License      : AGPL-3.0 (Ultralytics YOLOv5) - verify current
#                  license terms at the source repository before
#                  any commercial use
#   Input        : 1x3x640x640, RGB, float32, normalized 0-1
#   Output       : 1x25200x85
#                  (cx, cy, w, h, objectness, 80 class scores)
#   Person class : COCO class id 0
# ------------------------------------------------------------
#
# If the URL below is stale or unreachable (Ultralytics release
# asset URLs occasionally move), edit MODEL_URL and re-run, or
# follow the manual-copy procedure printed on failure / in the
# README "Offline Installation" section.

set -uo pipefail

# ============================================================
# CONFIGURABLE VARIABLES
# ============================================================
MODEL_URL="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
MODEL_NAME="yolov5n.onnx"
EXPECTED_INPUT_SIZE="640x640"
# ============================================================
#
# KNOWN ISSUE: the v7.0 release asset above has been observed to
# be exported at ONNX opset 17, which ONNX Runtime 1.10.0 (the
# version on this QCar1) does NOT support (it supports up to
# opset 15). If `python3 test_onnx.py` fails with an "opset"
# error after downloading, you MUST re-export the model at
# opset 12 on a separate machine WITH internet access, then copy
# the result into models/ manually (see README.md section 9/18):
#
#   git clone https://github.com/ultralytics/yolov5.git
#   cd yolov5 && git checkout v6.1
#   pip install -r requirements.txt
#   python export.py --weights yolov5n.pt --include onnx --opset 12 --imgsz 640
#   scp yolov5n.onnx <user>@<qcar1-ip>:<project>/models/yolov5n.onnx
#
# ============================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"

echo "============================================================"
echo "QCar1 Pedestrian Detection - YOLO ONNX model downloader"
echo "============================================================"

mkdir -p "${MODELS_DIR}"

# ------------------------------------------------------------
# Never silently overwrite an existing model
# ------------------------------------------------------------
if [ -f "${MODEL_PATH}" ]; then
    echo "[INFO] A model already exists at: ${MODEL_PATH}"
    read -r -p "Overwrite it? [y/N] " REPLY
    case "${REPLY}" in
        [Yy]*)
            echo "[INFO] Proceeding with overwrite."
            ;;
        *)
            echo "[INFO] Keeping existing model. Nothing downloaded."
            echo ""
            echo "Existing model info:"
            echo "  Path : ${MODEL_PATH}"
            echo "  Size : $(du -h "${MODEL_PATH}" | cut -f1)"
            exit 0
            ;;
    esac
fi

# ------------------------------------------------------------
# Check for a download tool
# ------------------------------------------------------------
DL_TOOL=""
if command -v curl >/dev/null 2>&1; then
    DL_TOOL="curl"
elif command -v wget >/dev/null 2>&1; then
    DL_TOOL="wget"
else
    echo "[FAIL] Neither 'curl' nor 'wget' is available on this system."
    echo "       Cannot download automatically."
    echo ""
    echo "MANUAL DOWNLOAD PROCEDURE:"
    echo "  1. On a computer with internet access, download:"
    echo "       ${MODEL_URL}"
    echo "  2. Save it as: ${MODEL_NAME}"
    echo "  3. Copy it to the QCar1 at:"
    echo "       ${MODEL_PATH}"
    echo "     e.g. via USB drive, or:"
    echo "       scp ${MODEL_NAME} <user>@<qcar1-ip>:${MODEL_PATH}"
    exit 1
fi

echo "[INFO] Download tool: ${DL_TOOL}"
echo "[INFO] Model URL   : ${MODEL_URL}"
echo "[INFO] Destination : ${MODEL_PATH}"
echo "[INFO] Downloading (this may take a while on a slow connection)..."

TMP_PATH="${MODEL_PATH}.part"
DOWNLOAD_OK=0

if [ "${DL_TOOL}" = "curl" ]; then
    if curl -L --fail --connect-timeout 15 -o "${TMP_PATH}" "${MODEL_URL}"; then
        DOWNLOAD_OK=1
    fi
else
    if wget --timeout=15 -O "${TMP_PATH}" "${MODEL_URL}"; then
        DOWNLOAD_OK=1
    fi
fi

if [ "${DOWNLOAD_OK}" -ne 1 ]; then
    rm -f "${TMP_PATH}"
    echo ""
    echo "[FAIL] Download failed. This QCar1 may not have internet"
    echo "       access, or the URL may have moved."
    echo ""
    echo "MANUAL DOWNLOAD PROCEDURE:"
    echo "  1. On a computer WITH internet access, download:"
    echo "       ${MODEL_URL}"
    echo "     (if that link is dead, search the Ultralytics YOLOv5"
    echo "     releases page for a 'yolov5n.onnx' asset, or export"
    echo "     one yourself with the 'ultralytics' pip package:"
    echo "       pip install ultralytics"
    echo "       yolo export model=yolov5n.pt format=onnx opset=12 imgsz=640"
    echo "     )"
    echo "  2. Save/rename it as: ${MODEL_NAME}"
    echo "  3. Copy it to this QCar1 project's models/ directory:"
    echo "       PC  --(USB or scp)-->  QCar1:${MODEL_PATH}"
    echo "     Example scp command (run FROM the PC):"
    echo "       scp ${MODEL_NAME} <qcar1-user>@<qcar1-ip>:${MODEL_PATH}"
    echo "  4. Re-run this script, or just proceed directly to:"
    echo "       python3 test_onnx.py"
    exit 1
fi

mv "${TMP_PATH}" "${MODEL_PATH}"

# ------------------------------------------------------------
# Verify the file actually exists and is non-trivial in size
# ------------------------------------------------------------
if [ ! -f "${MODEL_PATH}" ]; then
    echo "[FAIL] Download reported success but file is missing: ${MODEL_PATH}"
    exit 1
fi

FILE_SIZE_BYTES=$(wc -c < "${MODEL_PATH}" | tr -d ' ')

if [ "${FILE_SIZE_BYTES}" -lt 100000 ]; then
    echo "[FAIL] Downloaded file is suspiciously small (${FILE_SIZE_BYTES} bytes)."
    echo "       This is likely an error page, not a real ONNX model."
    echo "       Removing invalid file: ${MODEL_PATH}"
    rm -f "${MODEL_PATH}"
    echo "       Please use the manual download procedure (see README.md"
    echo "       section 18, 'Offline Installation')."
    exit 1
fi

FILE_SIZE_HUMAN=$(du -h "${MODEL_PATH}" | cut -f1)

echo ""
echo "============================================================"
echo "[PASS] Model downloaded and verified successfully."
echo "============================================================"
echo "Model path          : ${MODEL_PATH}"
echo "Model size           : ${FILE_SIZE_HUMAN} (${FILE_SIZE_BYTES} bytes)"
echo "Model name            : ${MODEL_NAME}"
echo "Expected input size : ${EXPECTED_INPUT_SIZE}"
echo "============================================================"
echo ""
echo "[INFO] A downloaded file passing the size check does NOT guarantee"
echo "       it will load in ONNX Runtime 1.10.0 - some release assets"
echo "       are exported at a newer opset than 15 (the max this ONNX"
echo "       Runtime version supports). The next step will tell you"
echo "       immediately if that's the case, with re-export instructions."
echo ""
echo "Next step: python3 test_onnx.py"
