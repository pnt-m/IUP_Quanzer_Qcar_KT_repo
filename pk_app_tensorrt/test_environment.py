#!/usr/bin/env python3
# ============================================================
# QCar1 Pedestrian Detection (TensorRT) - Environment Check
# ============================================================
"""
Audits system readiness on the NVIDIA Jetson TX2 for TensorRT acceleration
and Quanser hardware I/O, without making any modifications.
"""

from __future__ import print_function

import platform
import sys


def _pass(label, detail=""):
    msg = "[PASS] {}".format(label)
    if detail:
        msg += " - {}".format(detail)
    print(msg)


def _fail(label, detail=""):
    msg = "[FAIL] {}".format(label)
    if detail:
        msg += " - {}".format(detail)
    print(msg)


def _info(msg):
    print("[INFO] {}".format(msg))


def main():
    print("============================================================")
    print("QCar1 TensorRT Environment Health Check")
    print("============================================================")

    # 1. Python Check
    version = sys.version.split()[0]
    print("Python Executable : {}".format(sys.executable))
    print("Python Version    : {}".format(version))
    print("Architecture      : {}".format(platform.machine()))
    if version.startswith("3.6"):
        _pass("Python 3.6", "Target ABI matched (Python 3.6.9)")
    else:
        _fail("Python 3.6", "Found {} instead. QCar hardware bindings require Python 3.6.".format(version))

    # 2. NumPy Check
    try:
        import numpy
        _pass("NumPy", "version {}".format(numpy.__version__))
    except ImportError as exc:
        _fail("NumPy", str(exc))

    # 3. OpenCV Check
    try:
        import cv2
        _pass("OpenCV", "version {}".format(cv2.__version__))
    except ImportError as exc:
        _fail("OpenCV", str(exc))

    # 4. TensorRT Check
    try:
        import tensorrt as trt
        _pass("TensorRT", "version {}".format(trt.__version__))
    except ImportError as exc:
        _fail("TensorRT", "Could not import tensorrt. Check PYTHONPATH or system packages.")

    # 5. PyCUDA Check
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        _pass("PyCUDA", "GPU CUDA Context Initialized Successfully ({})".format(cuda.Device(0).name()))
    except ImportError as exc:
        _fail("PyCUDA", "Missing. Install via: sudo apt-get install -y python3-pycuda")
    except Exception as exc:
        _fail("PyCUDA Context", str(exc))

    # 6. Quanser PAL Check
    try:
        import pal
        from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR
        mode_str = "Physical QCar1 Hardware" if IS_PHYSICAL_QCAR else "Virtual QLabs Emulation"
        _pass("Quanser PAL", "Found pal package ({})".format(mode_str))
    except ImportError as exc:
        _fail("Quanser PAL", "Could not import pal.products.qcar. Ensure venv has --system-site-packages.")

    # 7. PyYAML Check
    try:
        import yaml
        _pass("PyYAML", "version {}".format(yaml.__version__))
    except ImportError:
        _fail("PyYAML", "Install with: pip install pyyaml")

    print("============================================================")


if __name__ == "__main__":
    main()
