#!/usr/bin/env bash
# ============================================================
# QCar1 YOLO Pedestrian Detection - Virtual Environment Setup
# ============================================================
#
# This script creates an ISOLATED Python virtual environment
# for this project WITHOUT modifying the existing system Python,
# NumPy, OpenCV, ONNX Runtime, TensorRT, or Quanser packages.
#
# It uses --system-site-packages so the venv can still SEE the
# already-installed system packages (opencv, onnxruntime,
# tensorrt, pal/quanser) rather than needing to reinstall them.
#
# This script does NOT:
#   - upgrade Python
#   - upgrade pip globally
#   - upgrade numpy/opencv/onnxruntime/tensorrt globally
#   - touch system-wide site-packages
#
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================================"
echo "QCar1 Pedestrian Detection - venv setup"
echo "============================================================"
echo "Project directory: ${PROJECT_DIR}"

# ------------------------------------------------------------
# 1. Locate a Python 3.6 interpreter
# ------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3.6 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        version_str="$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        if [ "${version_str}" = "3.6" ]; then
            PYTHON_BIN="${candidate}"
            break
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "[WARN] Could not find a Python 3.6 interpreter on PATH."
    echo "       Falling back to 'python3' (version check below)."
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[FAIL] No python3 interpreter found at all. Aborting."
        exit 1
    fi
fi

FOUND_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "[INFO] Using interpreter: ${PYTHON_BIN} (version ${FOUND_VERSION})"

case "${FOUND_VERSION}" in
    3.6*)
        echo "[PASS] Python 3.6 confirmed."
        ;;
    *)
        echo "[WARN] Expected Python 3.6.x but found ${FOUND_VERSION}."
        echo "       Continuing anyway, but compatibility is not guaranteed."
        echo "       This script will NEVER install a different Python version."
        ;;
esac

# ------------------------------------------------------------
# 2. Create the virtual environment (isolated, but can see
#    system site-packages such as opencv/onnxruntime/tensorrt/pal)
# ------------------------------------------------------------
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] venv already exists at ${VENV_DIR} - reusing it."
else
    echo "[INFO] Creating venv at ${VENV_DIR} with --system-site-packages ..."
    if "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"; then
        echo "[PASS] venv created."
    else
        echo "[FAIL] 'python3 -m venv --system-site-packages' failed."
        echo "       This can happen if the 'venv' module is not installed."
        echo "       Try: sudo apt-get install python3-venv"
        echo "       Aborting without modifying system packages."
        exit 1
    fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[INFO] venv activated: $(which python3)"

# ------------------------------------------------------------
# 3. Install ONLY the additional requirements for this project
#    (see requirements.txt - should be minimal / PyYAML only)
# ------------------------------------------------------------
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo "[INFO] Installing project-specific requirements (no system package upgrades) ..."
    python3 -m pip install --no-input -r "${PROJECT_DIR}/requirements.txt" || {
        echo "[WARN] pip install of project requirements failed."
        echo "       You may need to install PyYAML manually, or it may"
        echo "       already be provided by --system-site-packages."
    }
else
    echo "[WARN] requirements.txt not found - skipping pip install step."
fi

# ------------------------------------------------------------
# 4. Verify that critical system packages are visible from
#    inside the venv (they should be, thanks to
#    --system-site-packages). We do NOT install/replace them.
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Verifying access to existing system packages from venv"
echo "============================================================"

check_import() {
    local module="$1"
    local label="$2"
    if python3 -c "import ${module}" >/dev/null 2>&1; then
        echo "[PASS] ${label}"
        return 0
    else
        echo "[FAIL] ${label} (module '${module}' not importable)"
        return 1
    fi
}

OVERALL_OK=1

check_import "numpy" "NumPy"        || OVERALL_OK=0
check_import "cv2" "OpenCV"         || OVERALL_OK=0
check_import "onnxruntime" "ONNX Runtime" || OVERALL_OK=0
check_import "tensorrt" "TensorRT" || echo "[INFO] TensorRT not importable from Python here - this is OK, it is only needed for future Phase 2 acceleration."
check_import "pal" "Quanser 'pal' package" || echo "[INFO] Quanser 'pal' package not importable here - required only when running on/against the actual QCar1."

echo ""
echo "============================================================"
echo "Setup summary"
echo "============================================================"
echo "Venv location: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo ""
if [ "${OVERALL_OK}" -eq 1 ]; then
    echo "[SUMMARY] Core dependencies (NumPy, OpenCV, ONNX Runtime) are available."
    echo "Next steps:"
    echo "  1. ./download_model.sh"
    echo "  2. python3 test_environment.py"
    echo "  3. python3 test_onnx.py"
else
    echo "[SUMMARY] One or more core dependencies were NOT importable."
    echo "This project does not attempt to auto-install/replace them,"
    echo "because they are expected to already exist in the QCar1"
    echo "system environment. Please investigate before proceeding."
fi
echo "============================================================"
