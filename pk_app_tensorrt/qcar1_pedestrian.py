#!/usr/bin/env python3
# ============================================================
# QCar1 Real-Time Pedestrian Detection - TensorRT Edition
# ============================================================
"""
Real-time pedestrian detection and metric distance estimation on the
Quanser QCar1, powered by NVIDIA TensorRT FP16 GPU acceleration
and the Intel RealSense RGB-D camera.

Usage:
    # Standard headless run (ideal for SSH terminal):
    python3 qcar1_pedestrian.py --no-display

    # With X11 GUI display window:
    python3 qcar1_pedestrian.py

    # Using custom config:
    python3 qcar1_pedestrian.py --config config.yaml --no-display
"""

from __future__ import print_function

import argparse
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.depth_utils import estimate_distance_meters, normalize_for_display
from utils.yolo_onnx import YOLOONNXDetector
try:
    from utils.yolo_trt import YOLOTensorRTDetector
except ImportError:
    YOLOTensorRTDetector = None

try:
    import yaml
except ImportError:
    print("[FAIL] PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


DEFAULT_CONFIG = {
    "model": {
        "path": "models/yolov5n_fp16.engine",
        "input_width": 640,
        "input_height": 640,
        "confidence_threshold": 0.40,
        "iou_threshold": 0.45,
        "person_class_id": 0,
    },
    "camera": {
        "mode": "RGB, Depth",
    },
    "depth": {
        "scale": 1.0,
        "min_distance": 0.2,
        "max_distance": 10.0,
        "roi_fraction": 0.5,
    },
    "display": {
        "show_window": True,
        "show_fps": True,
        "show_depth": False,
    },
    "behavior": {
        "nearest_person_only": False,
    },
}


def load_config(config_path):
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if not os.path.isfile(config_path):
        print("[WARN] Config not found at '{}' - using defaults.".format(config_path))
        return cfg

    with open(config_path, "r") as f:
        try:
            loaded = yaml.safe_load(f) or {}
        except Exception as exc:
            print("[WARN] Failed to parse config ({}) - using defaults.".format(exc))
            return cfg

    for section, values in loaded.items():
        if section not in cfg:
            cfg[section] = {}
        if isinstance(values, dict):
            cfg[section].update(values)
    return cfg


class FPSCounter(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last_time = None

    def tick(self):
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                if self.fps == 0.0:
                    self.fps = instant_fps
                else:
                    self.fps = (1 - self.alpha) * self.fps + self.alpha * instant_fps
        self._last_time = now
        return self.fps


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="QCar1 TensorRT Real-Time Pedestrian Detection.")
    parser.add_argument("--config", default=os.path.join(project_dir, "config.yaml"),
                        help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--model", default=None,
                        help="Optional override path for .engine or .onnx model.")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display window completely (Headless mode for maximum FPS).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    camera_cfg = cfg["camera"]
    depth_cfg = cfg["depth"]
    display_cfg = cfg["display"]
    behavior_cfg = cfg["behavior"]

    model_path = args.model or model_cfg["path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_dir, model_path)

    print("============================================================")
    print("QCar1 Pedestrian Detection (NVIDIA TensorRT Edition)")
    print("============================================================")
    print("Config      : {}".format(args.config))
    print("Target Model: {}".format(model_path))
    print("Headless    : {}".format(args.no_display))

    # Auto-resolve model: if .engine is requested but doesn't exist, check fallback
    if not os.path.isfile(model_path):
        if model_path.endswith(".engine"):
            fallback_onnx = os.path.join(project_dir, "models", "yolov5n.onnx")
            if os.path.isfile(fallback_onnx):
                print("[WARN] TensorRT engine not found at '{}'.".format(model_path))
                print("[WARN] Falling back to CPU ONNX model: {}".format(fallback_onnx))
                print("[TIP]  Run ./build_engine.sh to build the 30+ FPS GPU engine!")
                model_path = fallback_onnx
            else:
                print("[FAIL] Neither engine nor ONNX model found.")
                print("       Run ./build_engine.sh first.")
                sys.exit(1)
        else:
            print("[FAIL] Model file not found: {}".format(model_path))
            sys.exit(1)

    # 1. Verify Quanser Hardware PAL API
    try:
        from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR
    except ImportError as exc:
        print("[FAIL] Could not import pal.products.qcar.QCarRealSense.")
        print("       Run inside the QCar1 Python environment with: sudo PYTHONPATH=$PYTHONPATH python3 ...")
        print("       Original error: {}".format(exc))
        sys.exit(1)

    if not IS_PHYSICAL_QCAR:
        print("[INFO] Initializing QLabs virtual environment...")
        try:
            import qlabs_setup
            qlabs_setup.setup()
        except Exception as exc:
            print("[FAIL] qlabs_setup failed: {}".format(exc))
            sys.exit(1)

    # 2. Initialize Detector (TensorRT or ONNX fallback)
    is_engine = model_path.endswith(".engine")
    try:
        if is_engine:
            if YOLOTensorRTDetector is None:
                raise ImportError("TensorRT detector failed to load. Check pycuda and tensorrt.")
            print("[INFO] Loading TensorRT GPU Engine...")
            detector = YOLOTensorRTDetector(
                engine_path=model_path,
                confidence_threshold=model_cfg.get("confidence_threshold", 0.40),
                iou_threshold=model_cfg.get("iou_threshold", 0.45),
                person_class_id=model_cfg.get("person_class_id", 0),
                input_width=model_cfg.get("input_width", 640),
                input_height=model_cfg.get("input_height", 640),
            )
        else:
            print("[INFO] Loading ONNX Runtime CPU Detector...")
            detector = YOLOONNXDetector(
                model_path=model_path,
                confidence_threshold=model_cfg.get("confidence_threshold", 0.40),
                iou_threshold=model_cfg.get("iou_threshold", 0.45),
                person_class_id=model_cfg.get("person_class_id", 0),
                input_width=model_cfg.get("input_width", 640),
                input_height=model_cfg.get("input_height", 640),
            )
    except Exception as exc:
        print("[FAIL] Failed to initialize detector: {}".format(exc))
        sys.exit(1)

    print("Execution Providers: {}".format(detector.get_providers()))
    print("Model Resolution   : {}".format(detector.get_input_size()))

    depth_scale = float(depth_cfg.get("scale", 1.0))
    min_dist = float(depth_cfg.get("min_distance", 0.2))
    max_dist = float(depth_cfg.get("max_distance", 10.0))
    roi_frac = float(depth_cfg.get("roi_fraction", 0.5))

    show_window = display_cfg.get("show_window", True) and not args.no_display
    show_fps = display_cfg.get("show_fps", True)
    nearest_only = behavior_cfg.get("nearest_person_only", False)

    fps_counter = FPSCounter()

    print("------------------------------------------------------------")
    if args.no_display:
        print("Mode: HEADLESS (Display disabled via --no-display).")
        print("Press Ctrl+C to terminate.")
    else:
        print("Mode: GUI (Display active). Press 'q' in window or Ctrl+C to stop.")
    print("------------------------------------------------------------")

    cam_mode = camera_cfg.get("mode", "RGB, Depth")
    cam_w = int(camera_cfg.get("frame_width", 640))
    cam_h = int(camera_cfg.get("frame_height", 480))
    cam_fps = int(camera_cfg.get("frame_rate", 30))

    try:
        with QCarRealSense(
            mode=cam_mode,
            frameWidthRGB=cam_w,
            frameHeightRGB=cam_h,
            frameRateRGB=cam_fps,
            frameWidthDepth=cam_w,
            frameHeightDepth=cam_h,
            frameRateDepth=cam_fps
        ) as myCam:
            while True:
                loop_t0 = time.time()

                # Read Camera Frames
                myCam.read_RGB()
                frame = myCam.imageBufferRGB
                if frame is None:
                    continue

                # TensorRT GPU Inference
                detections = detector.detect(frame, persons_only=True)

                # Read Depth Frame
                myCam.read_depth(dataMode='PX')
                depth_frame = myCam.imageBufferDepthPX

                # Distance Sampling
                distances = []
                for det in detections:
                    dist = estimate_distance_meters(
                        depth_frame, det["bbox"],
                        depth_scale=depth_scale,
                        roi_fraction=roi_frac,
                        min_distance=min_dist,
                        max_distance=max_dist
                    )
                    distances.append(dist)

                highlight_idx = None
                if detections:
                    valid_pairs = [(i, d) for i, d in enumerate(distances) if d is not None]
                    if valid_pairs:
                        highlight_idx = min(valid_pairs, key=lambda p: p[1])[0]

                fps = fps_counter.tick()
                total_loop_ms = (time.time() - loop_t0) * 1000.0

                # Telemetry Print
                if detections:
                    for i, det in enumerate(detections):
                        dist_str = "{:.2f} m".format(distances[i]) if distances[i] is not None else "unknown"
                        print("  >> Pedestrian {}: Conf {:.2f} | Dist: {}".format(i + 1, det["confidence"], dist_str))

                print("[TELEMETRY] Persons: {} | FPS: {:>4.1f} | GPU Inference: {:>4.1f} ms | Total Loop: {:>4.1f} ms".format(
                    len(detections), fps, detector.last_inference_ms, total_loop_ms
                ))

                # Optional GUI rendering
                if show_window:
                    display_dets = [detections[highlight_idx]] if (nearest_only and highlight_idx is not None) else detections
                    display_dists = [distances[highlight_idx]] if (nearest_only and highlight_idx is not None) else distances

                    annotated = detector.draw_detections(
                        frame, display_dets, distances=display_dists,
                        highlight_index=(0 if (nearest_only and highlight_idx is not None) else highlight_idx)
                    )

                    if show_fps:
                        cv2.putText(
                            annotated, "FPS: {:.1f} (TensorRT)".format(fps), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                        )
                        cv2.putText(
                            annotated, "Pedestrians: {}".format(len(detections)), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                        )

                    cv2.imshow("QCar1 Pedestrian Detection (TensorRT)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] 'q' pressed. Terminating...")
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Exiting cleanly...")
    except Exception as exc:
        print("[FAIL] Unexpected error in camera loop: {}".format(exc))
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("============================================================")
        print("QCar1 TensorRT application terminated cleanly.")
        print("============================================================")


if __name__ == "__main__":
    main()
