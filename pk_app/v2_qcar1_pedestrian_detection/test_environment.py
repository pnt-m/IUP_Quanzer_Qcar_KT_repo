#!/usr/bin/env python3
# ============================================================
# QCar1 YOLO Pedestrian Detection - Environment check
# ============================================================
"""
Reports the status of every component this project depends on,
WITHOUT installing, upgrading, or modifying anything.

Run:
    python3 test_environment.py
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


def check_python():
    version = sys.version.split()[0]
    print("Python executable : {}".format(sys.executable))
    print("Python version    : {}".format(version))
    print("Architecture      : {}".format(platform.machine()))
    if version.startswith("3.6"):
        _pass("Python 3.6")
    else:
        _fail("Python 3.6", "found {} instead - project targets 3.6.9".format(version))


def check_numpy():
    try:
        import numpy
        _pass("NumPy", "version {}".format(numpy.__version__))
    except ImportError as exc:
        _fail("NumPy", str(exc))


def check_opencv():
    try:
        import cv2
        _pass("OpenCV", "version {}".format(cv2.__version__))
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            _info("OpenCV CUDA-enabled devices: {}".format(cuda_count))
            if cuda_count == 0:
                _info("OpenCV has no CUDA devices available - this is expected "
                      "on this environment; CPU-only OpenCV ops will be used.")
        except Exception:
            _info("OpenCV was not built with CUDA support (cv2.cuda unavailable). "
                  "This is expected; not treated as an error.")
    except ImportError as exc:
        _fail("OpenCV", str(exc))


def check_onnxruntime():
    try:
        import onnxruntime as ort
        _pass("ONNX Runtime", "version {}".format(ort.__version__))
        providers = ort.get_available_providers()
        print("ONNX Runtime providers: {}".format(providers))
        if providers == ["CPUExecutionProvider"]:
            _info("ONNX Runtime is CPU-only.")
            _info("TensorRT can be used for future GPU acceleration.")
        elif "CPUExecutionProvider" in providers:
            _info("Additional execution providers detected beyond CPU: {}".format(
                [p for p in providers if p != "CPUExecutionProvider"]
            ))
    except ImportError as exc:
        _fail("ONNX Runtime", str(exc))


def check_tensorrt():
    try:
        import tensorrt as trt
        _pass("TensorRT", "version {}".format(trt.__version__))
    except ImportError as exc:
        _fail("TensorRT", "{} (not required for Phase 1 / ONNX Runtime CPU inference)".format(exc))


def check_quanser():
    try:
        import pal  # noqa: F401
        _pass("Quanser 'pal' package")
    except ImportError as exc:
        _fail("Quanser 'pal' package", str(exc))


def check_qcar_realsense():
    try:
        from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR  # noqa: F401
        _pass("QCarRealSense import")
        _info("IS_PHYSICAL_QCAR = {}".format(IS_PHYSICAL_QCAR))
        if not IS_PHYSICAL_QCAR:
            _info("Running against QLabs virtual QCar (not physical hardware). "
                  "qlabs_setup.setup() will be called by qcar1_pedestrian.py.")
    except ImportError as exc:
        _fail("QCarRealSense import", str(exc))


def check_pyyaml():
    try:
        import yaml  # noqa: F401
        _pass("PyYAML", "available for config.yaml parsing")
    except ImportError as exc:
        _fail("PyYAML", "{} - install via: pip install -r requirements.txt".format(exc))


def main():
    print("============================================================")
    print("QCar1 Pedestrian Detection - environment check")
    print("============================================================")
    check_python()
    print("------------------------------------------------------------")
    check_numpy()
    check_opencv()
    check_onnxruntime()
    check_tensorrt()
    check_pyyaml()
    check_quanser()
    check_qcar_realsense()
    print("============================================================")
    print("Notes:")
    print(" - TensorRT and Quanser/QCarRealSense failures are expected")
    print("   and non-fatal if you are testing on a development machine")
    print("   that is not the QCar1 itself (e.g. only running")
    print("   test_onnx.py / detect_image.py on saved images).")
    print(" - They ARE required to run qcar1_pedestrian.py on the")
    print("   actual robot.")
    print("============================================================")


if __name__ == "__main__":
    main()
