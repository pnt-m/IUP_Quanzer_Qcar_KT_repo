#!/usr/bin/env python3
# ============================================================
# QCar1 YOLO Pedestrian Detection - main application
# ============================================================
"""
Real-time pedestrian detection and distance estimation on the
Quanser QCar1, using the RealSense RGB+Depth camera, a YOLO
ONNX model (via ONNX Runtime), and this project's depth-ROI
distance estimator.

Run:
    python3 qcar1_pedestrian.py
    python3 qcar1_pedestrian.py --config config.yaml
    python3 qcar1_pedestrian.py --no-display

Press 'q' in the display window (or Ctrl+C in the terminal) to
exit cleanly.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.yolo_onnx import YOLOONNXDetector       # noqa: E402
from utils.depth_utils import estimate_distance_meters  # noqa: E402

try:
    import yaml
except ImportError:
    print("[FAIL] PyYAML is required to read config.yaml.")
    print("       Install it with: pip install -r requirements.txt")
    sys.exit(1)


DEFAULT_CONFIG = {
    "model": {
        "path": "models/yolov5n.onnx",
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
    """Load config.yaml, falling back to DEFAULT_CONFIG for any
    missing keys so the app can still run with a partial file."""
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}

    if not os.path.isfile(config_path):
        print("[WARN] Config file not found at '{}' - using built-in defaults.".format(config_path))
        return cfg

    with open(config_path, "r") as f:
        try:
            loaded = yaml.safe_load(f) or {}
        except Exception as exc:
            print("[WARN] Failed to parse '{}' ({}) - using built-in defaults.".format(config_path, exc))
            return cfg

    for section, values in loaded.items():
        if section not in cfg:
            cfg[section] = {}
        if isinstance(values, dict):
            cfg[section].update(values)

    return cfg


class FPSCounter(object):
    """Simple exponential-moving-average FPS counter."""

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

    parser = argparse.ArgumentParser(description="QCar1 real-time pedestrian detection.")
    parser.add_argument("--config", default=os.path.join(project_dir, "config.yaml"),
                         help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the on-screen display window even if config enables it.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    camera_cfg = cfg["camera"]
    depth_cfg = cfg["depth"]
    display_cfg = cfg["display"]
    behavior_cfg = cfg["behavior"]

    model_path = model_cfg["path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_dir, model_path)

    print("============================================================")
    print("QCar1 Pedestrian Detection - main application")
    print("============================================================")
    print("Config file : {}".format(args.config))
    print("Model path  : {}".format(model_path))

    if not os.path.isfile(model_path):
        print("[FAIL] Model file not found: {}".format(model_path))
        print("       Run ./download_model.sh first.")
        sys.exit(1)

    # ------------------------------------------------------------
    # Import Quanser QCar API. This MUST work on the real QCar1 /
    # QLabs environment; it is intentionally not wrapped in a
    # silent fallback so hardware problems are visible immediately.
    # ------------------------------------------------------------
    try:
        from pal.products.qcar import QCarRealSense, IS_PHYSICAL_QCAR
    except ImportError as exc:
        print("[FAIL] Could not import pal.products.qcar.QCarRealSense.")
        print("       This script must be run inside the QCar1 software")
        print("       environment. Original error: {}".format(exc))
        sys.exit(1)

    if not IS_PHYSICAL_QCAR:
        print("[INFO] Not running on physical QCar1 hardware - "
              "initializing QLabs virtual environment.")
        try:
            import qlabs_setup
            qlabs_setup.setup()
        except Exception as exc:
            print("[FAIL] qlabs_setup.setup() failed: {}".format(exc))
            sys.exit(1)

    # ------------------------------------------------------------
    # Load the detector
    # ------------------------------------------------------------
    try:
        detector = YOLOONNXDetector(
            model_path=model_path,
            confidence_threshold=model_cfg.get("confidence_threshold", 0.40),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            person_class_id=model_cfg.get("person_class_id", 0),
            input_width=model_cfg.get("input_width", 640),
            input_height=model_cfg.get("input_height", 640),
        )
    except Exception as exc:
        print("[FAIL] Could not initialize YOLO ONNX detector: {}".format(exc))
        sys.exit(1)

    print("Execution providers: {}".format(detector.get_providers()))
    print("Model input size: {}".format(detector.get_input_size()))

    depth_scale = float(depth_cfg.get("scale", 1.0))
    min_distance = float(depth_cfg.get("min_distance", 0.2))
    max_distance = float(depth_cfg.get("max_distance", 10.0))
    roi_fraction = float(depth_cfg.get("roi_fraction", 0.5))

    if depth_scale == 1.0:
        print("[WARN] depth.scale in config.yaml is at the default placeholder")
        print("       value of 1.0. Distances printed below may NOT be in")
        print("       real meters until this is calibrated. See README.md,")
        print("       section 'Understanding Depth'.")

    show_window = display_cfg.get("show_window", True) and not args.no_display
    show_fps = display_cfg.get("show_fps", True)
    nearest_only = behavior_cfg.get("nearest_person_only", False)

    fps_counter = FPSCounter()

    print("------------------------------------------------------------")
    print("Starting camera loop. Press 'q' in the display window (or")
    print("Ctrl+C in this terminal) to stop.")
    print("------------------------------------------------------------")

    try:
        with QCarRealSense(mode=camera_cfg.get("mode", "RGB, Depth")) as myCam:
            while True:
                loop_t0 = time.time()

                # -------------------- Read RGB --------------------
                myCam.read_RGB()
                frame = myCam.imageBufferRGB
                if frame is None:
                    print("[WARN] No RGB frame received - skipping this iteration.")
                    continue

                # -------------------- Detect pedestrians --------------------
                detections = detector.detect(frame, persons_only=True)

                # -------------------- Read depth --------------------
                myCam.read_depth(dataMode='PX')
                depth_frame = myCam.imageBufferDepthPX

                # -------------------- Distance per detection --------------------
                distances = []
                for det in detections:
                    dist = estimate_distance_meters(
                        depth_frame, det["bbox"],
                        depth_scale=depth_scale,
                        roi_fraction=roi_fraction,
                        min_distance=min_distance,
                        max_distance=max_distance,
                    )
                    distances.append(dist)

                # -------------------- Determine nearest --------------------
                highlight_index = None
                if detections:
                    valid_pairs = [(i, d) for i, d in enumerate(distances) if d is not None]
                    if valid_pairs:
                        highlight_index = min(valid_pairs, key=lambda p: p[1])[0]

                # -------------------- FPS --------------------
                fps = fps_counter.tick()
                total_loop_ms = (time.time() - loop_t0) * 1000.0

                # -------------------- Console output --------------------
                if detections:
                    for i, det in enumerate(detections):
                        dist_str = "{:.2f} m".format(distances[i]) if distances[i] is not None else "unknown"
                        print("Person {}: {:.2f}, {}".format(i + 1, det["confidence"], dist_str))
                print("Pedestrians: {} | FPS: {:.1f} | Inference: {:.1f} ms | Loop: {:.1f} ms".format(
                    len(detections), fps, detector.last_inference_ms, total_loop_ms
                ))

                # -------------------- Display --------------------
                if show_window:
                    if nearest_only and highlight_index is not None:
                        display_detections = [detections[highlight_index]]
                        display_distances = [distances[highlight_index]]
                    else:
                        display_detections = detections
                        display_distances = distances

                    annotated = detector.draw_detections(
                        frame, display_detections, distances=display_distances,
                        highlight_index=(0 if (nearest_only and highlight_index is not None) else highlight_index)
                    )

                    if show_fps:
                        cv2.putText(
                            annotated, "FPS: {:.1f}".format(fps), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(
                            annotated, "Pedestrians: {}".format(len(detections)), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )

                    cv2.imshow("QCar1 Pedestrian Detection", annotated)

                    if display_cfg.get("show_depth", False) and depth_frame is not None:
                        from utils.depth_utils import normalize_for_display
                        depth_display = normalize_for_display(depth_frame, max_distance=2.0)
                        if depth_display is not None:
                            cv2.imshow("Depth (cosmetic normalization)", depth_display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] 'q' pressed - exiting.")
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received - exiting cleanly.")
    except Exception as exc:
        print("[FAIL] Unexpected error in camera loop: {}".format(exc))
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("============================================================")
        print("QCar1 Pedestrian Detection stopped. Resources released.")
        print("============================================================")


if __name__ == "__main__":
    main()
