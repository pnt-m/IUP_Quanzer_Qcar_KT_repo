#!/usr/bin/env python3
# ============================================================
# QCar1 YOLO Pedestrian Detection - image-based detector
# ============================================================
"""
Run pedestrian detection on a single saved image (no camera or
QCar hardware required). Useful for validating the model and
tuning thresholds before running on the robot.

Usage:
    python3 detect_image.py --image test_images/test.jpg
    python3 detect_image.py --image test_images/test.jpg \\
        --model models/yolov5n.onnx --output output/result.jpg \\
        --confidence 0.40 --iou 0.45 --show
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.yolo_onnx import YOLOONNXDetector  # noqa: E402


def find_default_model(project_dir):
    models_dir = os.path.join(project_dir, "models")
    if not os.path.isdir(models_dir):
        return None
    onnx_files = sorted(
        f for f in os.listdir(models_dir) if f.lower().endswith(".onnx")
    )
    return os.path.join(models_dir, onnx_files[0]) if onnx_files else None


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Detect pedestrians in a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default=None,
                         help="Path to the ONNX model (defaults to first *.onnx in models/).")
    parser.add_argument("--output", default=os.path.join(project_dir, "output", "result.jpg"),
                         help="Path to save the annotated output image.")
    parser.add_argument("--confidence", type=float, default=0.40,
                         help="Confidence threshold (default: 0.40).")
    parser.add_argument("--iou", type=float, default=0.45,
                         help="IoU threshold for NMS (default: 0.45).")
    parser.add_argument("--show", action="store_true",
                         help="Display the result in a window (requires a GUI/display).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("[FAIL] Input image not found: {}".format(args.image))
        sys.exit(1)

    model_path = args.model or find_default_model(project_dir)
    if model_path is None or not os.path.isfile(model_path):
        print("[FAIL] No ONNX model found. Run ./download_model.sh first, "
              "or pass --model <path>.")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print("[FAIL] Could not read image (unsupported format or corrupt file): {}".format(args.image))
        sys.exit(1)

    print("============================================================")
    print("QCar1 Pedestrian Detection - single image test")
    print("============================================================")
    print("Image : {}".format(args.image))
    print("Model : {}".format(model_path))

    try:
        detector = YOLOONNXDetector(
            model_path=model_path,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
        )
    except Exception as exc:
        print("[FAIL] Could not initialize detector: {}".format(exc))
        sys.exit(1)

    print("Execution providers: {}".format(detector.get_providers()))
    print("Model input size: {}".format(detector.get_input_size()))

    t0 = time.time()
    detections = detector.detect(image, persons_only=True)
    inference_ms = detector.last_inference_ms
    total_ms = (time.time() - t0) * 1000.0

    print("------------------------------------------------------------")
    print("Detections: {}".format(len(detections)))
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        print("Person {}: {:.2f}".format(i + 1, det["confidence"]))
        print("Bounding box: {},{},{},{}".format(x1, y1, x2, y2))
    print("------------------------------------------------------------")
    print("Inference time : {:.2f} ms".format(inference_ms))
    print("Total time     : {:.2f} ms".format(total_ms))

    annotated = detector.draw_detections(image, detections)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(args.output, annotated)
    print("Saved annotated image to: {}".format(args.output))

    if args.show:
        try:
            cv2.imshow("QCar1 Pedestrian Detection - detect_image.py", annotated)
            print("Press any key in the image window to close it...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as exc:
            print("[WARN] Could not display window (no GUI/display available?): {}".format(exc))

    print("============================================================")


if __name__ == "__main__":
    main()
