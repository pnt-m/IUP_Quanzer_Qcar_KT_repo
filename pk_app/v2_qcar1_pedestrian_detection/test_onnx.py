#!/usr/bin/env python3
# ============================================================
# QCar1 YOLO Pedestrian Detection - ONNX model validation
# ============================================================
"""
Loads the YOLO ONNX model from models/, prints its I/O
signature, runs one dummy inference, and reports timing.

Does NOT require the QCar camera - this is the first
functional test, meant to be run right after download_model.sh.

Run:
    python3 test_onnx.py
    python3 test_onnx.py --model models/yolov5n.onnx
"""

from __future__ import print_function

import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[FAIL] onnxruntime is not importable. See README.md troubleshooting.")
    sys.exit(1)


def find_default_model(models_dir):
    onnx_files = sorted(glob.glob(os.path.join(models_dir, "*.onnx")))
    return onnx_files[0] if onnx_files else None


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    default_models_dir = os.path.join(project_dir, "models")

    parser = argparse.ArgumentParser(description="Validate the YOLO ONNX model.")
    parser.add_argument("--model", default=None,
                         help="Path to the .onnx model. Defaults to the first "
                              "*.onnx file found in models/.")
    args = parser.parse_args()

    print("============================================================")
    print("QCar1 Pedestrian Detection - ONNX model validation")
    print("============================================================")

    model_path = args.model
    if model_path is None:
        model_path = find_default_model(default_models_dir)
        if model_path is None:
            print("[FAIL] No .onnx model found in '{}'.".format(default_models_dir))
            print("       Run ./download_model.sh first, or pass --model <path>.")
            sys.exit(1)

    if not os.path.isfile(model_path):
        print("[FAIL] Model file does not exist: {}".format(model_path))
        sys.exit(1)

    print("Model path: {}".format(model_path))

    try:
        session = ort.InferenceSession(model_path)
    except Exception as exc:
        exc_str = str(exc)
        print("[FAIL] Could not load model with ONNX Runtime.")
        print("       Error: {}".format(exc_str))
        if "opset" in exc_str.lower() and ("official support" in exc_str.lower() or "under development" in exc_str.lower()):
            print("")
            print("------------------------------------------------------------")
            print("[INFO] This looks like an ONNX OPSET VERSION MISMATCH.")
            print("       ONNX Runtime 1.10.0 (installed on QCar1) only")
            print("       officially supports opset versions up to 15.")
            print("       The .onnx file you have was exported at a newer,")
            print("       unsupported opset (e.g. 16 or 17).")
            print("")
            print("       FIX: re-export the model at opset 12 on a machine")
            print("       WITH internet access (not the QCar1), then copy it")
            print("       over:")
            print("")
            print("         git clone https://github.com/ultralytics/yolov5.git")
            print("         cd yolov5 && git checkout v6.1")
            print("         pip install -r requirements.txt")
            print("         python export.py --weights yolov5n.pt \\")
            print("             --include onnx --opset 12 --imgsz 640")
            print("")
            print("       Then: scp yolov5n.onnx <user>@<qcar1-ip>:{}".format(model_path))
            print("       See README.md section 9 / 18 for details.")
            print("------------------------------------------------------------")
        sys.exit(1)

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("------------------------------------------------------------")
    for inp in inputs:
        print("Input name  : {}".format(inp.name))
        print("Input shape : {}".format(inp.shape))
        print("Input type  : {}".format(inp.type))
    print("------------------------------------------------------------")
    for out in outputs:
        print("Output name  : {}".format(out.name))
        print("Output shape : {}".format(out.shape))
    print("------------------------------------------------------------")
    providers = session.get_providers()
    print("Execution providers: {}".format(providers))
    print("------------------------------------------------------------")

    # Build a dummy input matching the model's declared shape,
    # substituting a sensible default for any dynamic dimensions.
    input_meta = inputs[0]
    shape = []
    for i, dim in enumerate(input_meta.shape):
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            if i == 0:
                shape.append(1)      # batch
            elif i == 1:
                shape.append(3)      # channels
            else:
                shape.append(640)    # spatial dims fallback

    print("Using dummy input shape: {}".format(shape))
    dummy_input = np.random.rand(*shape).astype(np.float32)

    output_names = [o.name for o in outputs]

    t0 = time.time()
    try:
        results = session.run(output_names, {input_meta.name: dummy_input})
    except Exception as exc:
        print("[FAIL] Dummy inference failed.")
        print("       Error: {}".format(exc))
        sys.exit(1)
    elapsed_ms = (time.time() - t0) * 1000.0

    print("------------------------------------------------------------")
    print("[PASS] Dummy inference completed successfully.")
    print("Inference time: {:.2f} ms".format(elapsed_ms))
    for name, arr in zip(output_names, results):
        print("Output '{}' shape: {}".format(name, np.array(arr).shape))
    print("============================================================")
    print("Next step: python3 detect_image.py --image test_images/<your_image>.jpg")


if __name__ == "__main__":
    main()
