#!/usr/bin/env python3
# ============================================================
# QCar1 Pedestrian Detection (TensorRT) - Static Image Test
# ============================================================
"""
Runs pedestrian detection on a single saved image using TensorRT GPU acceleration
(or ONNX Runtime fallback). Saves the annotated result to output/ and prints
inference timing.

Usage:
    python3 detect_image.py --image test_images/pedestrian.jpg
    python3 detect_image.py --image test_images/pedestrian.jpg --show
    python3 detect_image.py --image test_images/pedestrian.jpg --model models/yolov5n_fp16.engine
"""

from __future__ import print_function

import argparse
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.yolo_onnx import YOLOONNXDetector
try:
    from utils.yolo_trt import YOLOTensorRTDetector
except ImportError:
    YOLOTensorRTDetector = None


def find_default_model(project_dir):
    models_dir = os.path.join(project_dir, "models")
    if not os.path.isdir(models_dir):
        return None

    # Prefer TensorRT engine if present
    engine_path = os.path.join(models_dir, "yolov5n_fp16.engine")
    if os.path.isfile(engine_path):
        return engine_path

    # Fallback to ONNX
    onnx_path = os.path.join(models_dir, "yolov5n.onnx")
    if os.path.isfile(onnx_path):
        return onnx_path

    return None


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Detect pedestrians in a single image via TensorRT.")
    parser.add_argument("--image", default=os.path.join(project_dir, "test_images", "pedestrian.jpg"),
                        help="Path to the input image.")
    parser.add_argument("--model", default=None,
                        help="Path to .engine or .onnx model (defaults to models/yolov5n_fp16.engine).")
    parser.add_argument("--output", default=os.path.join(project_dir, "output", "result_tensorrt.jpg"),
                        help="Path to save annotated image.")
    parser.add_argument("--confidence", type=float, default=0.40,
                        help="Confidence threshold (default: 0.40).")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45).")
    parser.add_argument("--show", action="store_true",
                        help="Display result in window (requires X11/GUI display).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("[FAIL] Input image not found: {}".format(args.image))
        sys.exit(1)

    model_path = args.model or find_default_model(project_dir)
    if model_path is None or not os.path.isfile(model_path):
        print("[FAIL] Model file not found at '{}'.".format(model_path))
        print("       Run ./build_engine.sh first to generate the TensorRT engine.")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print("[FAIL] Could not load image: {}".format(args.image))
        sys.exit(1)

    print("============================================================")
    print("QCar1 Pedestrian Detection - Static Image Test")
    print("============================================================")
    print("Image : {}".format(args.image))
    print("Model : {}".format(model_path))

    is_trt = model_path.endswith(".engine")
    try:
        if is_trt:
            if YOLOTensorRTDetector is None:
                raise ImportError("TensorRT detector could not be loaded. Ensure tensorrt and pycuda are installed.")
            print("[INFO] Initializing TensorRT GPU detector...")
            detector = YOLOTensorRTDetector(
                engine_path=model_path,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou
            )
        else:
            print("[INFO] Initializing ONNX Runtime CPU detector...")
            detector = YOLOONNXDetector(
                model_path=model_path,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou
            )
    except Exception as exc:
        print("[FAIL] Could not initialize detector: {}".format(exc))
        sys.exit(1)

    print("Execution Providers: {}".format(detector.get_providers()))
    print("Input Size         : {}".format(detector.get_input_size()))

    # Run Detection
    t0 = time.time()
    detections = detector.detect(image, persons_only=True)
    total_ms = (time.time() - t0) * 1000.0

    print("------------------------------------------------------------")
    print("Detections Found: {}".format(len(detections)))
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        print("  Person {}: Confidence {:.2f} at [{}, {}, {}, {}]".format(
            i + 1, det["confidence"], x1, y1, x2, y2
        ))
    print("GPU Kernel Latency : {:.2f} ms".format(detector.last_inference_ms))
    print("Total Pipeline Time: {:.2f} ms ({:.1f} FPS equivalent)".format(
        total_ms, 1000.0 / total_ms if total_ms > 0 else 0
    ))

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    annotated = detector.draw_detections(image, detections)
    cv2.imwrite(args.output, annotated)
    print("[PASS] Saved annotated result to: {}".format(args.output))

    if args.show:
        cv2.imshow("QCar1 TensorRT Detection", annotated)
        print("Press any key in display window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("============================================================")


if __name__ == "__main__":
    main()
