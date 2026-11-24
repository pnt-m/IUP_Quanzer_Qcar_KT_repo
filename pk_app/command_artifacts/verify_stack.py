#!/usr/bin/env python3
# ==============================================================================
# Cross-Platform Project Stack & Health Verifier
# ==============================================================================
"""
Inspects the local project directory structure, checks model files,
verifies installed libraries, and validates the configuration.
Can run on both Windows host and Jetson TX2 target.
"""

from __future__ import print_function

import os
import sys
import platform

def header(text):
    print("\n" + "=" * 65)
    print(" " + text)
    print("=" * 65)

def check_file(path, label, min_size=0):
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size >= min_size:
            print("[PASS] {:<30} (Size: {:>10,} bytes)".format(label, size))
            return True
        else:
            print("[WARN] {:<30} (File suspiciously small: {} bytes)".format(label, size))
            return False
    else:
        print("[FAIL] {:<30} (File missing: {})".format(label, path))
        return False

def check_module(mod_name, friendly_label):
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "unknown")
        print("[PASS] {:<25} - Version: {}".format(friendly_label, ver))
        return True
    except ImportError:
        print("[INFO] {:<25} - Not installed in current Python environment".format(friendly_label))
        return False

def main():
    header("QCar1 Pedestrian Detection - Project Diagnostic & Health Verifier")
    print("Platform       : {}".format(platform.platform()))
    print("Python Exec    : {}".format(sys.executable))
    print("Python Version : {}".format(sys.version.split()[0]))

    # Locate project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "v2_qcar1_pedestrian_detection"))
    print("Target Root    : {}".format(project_root))

    header("1. Core Project Files & Model Assets")
    check_file(os.path.join(project_root, "config.yaml"), "Configuration File")
    check_file(os.path.join(project_root, "qcar1_pedestrian.py"), "Main Application")
    check_file(os.path.join(project_root, "detect_image.py"), "Single Image Detector")
    check_file(os.path.join(project_root, "test_onnx.py"), "Model Tester")
    check_file(os.path.join(project_root, "utils", "yolo_onnx.py"), "YOLO ONNX Engine")
    check_file(os.path.join(project_root, "utils", "depth_utils.py"), "Depth Estimator")
    check_file(os.path.join(project_root, "models", "yolov5n.onnx"), "YOLOv5n ONNX Weights", min_size=100000)
    check_file(os.path.join(project_root, "test_images", "pedestrian.jpg"), "Sample Test Image")

    header("2. Python Environment Modules (Active Runtime)")
    check_module("numpy", "NumPy")
    check_module("cv2", "OpenCV")
    check_module("onnxruntime", "ONNX Runtime")
    check_module("yaml", "PyYAML")
    check_module("tensorrt", "TensorRT (Optional)")
    check_module("pal", "Quanser PAL (Robot Hardware)")

    header("Summary & Guidance")
    print("  - If running on a workstation/PC, missing 'pal' or 'tensorrt' is normal.")
    print("  - If running on the QCar1, all modules except TensorRT are essential.")
    print("  - Run 'python detect_image.py --image test_images/pedestrian.jpg' to test inference.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
