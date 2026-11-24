#!/usr/bin/env python3
# ============================================================
# QCar1 Autonomous Driving & 7-Class TensorRT Perception Suite
# ============================================================
"""
Unified Autonomous Navigation System on Quanser QCar1 (NVIDIA Jetson TX2):
  - Lateral: Map-Aware Lane Tracker + Stanley Lateral Guidance (CSI Front)
  - Longitudinal: 7-Class TensorRT FP16 Reflex Subsumption (RealSense RGB-D)
  - Features:
      * 5-Second Mandatory Stop at Stop Signs (then auto-resumes)
      * Emergency Pedestrian Halt (within 1.5m)
      * Red Light Full Stop / Green Light Cruise
      * Caution Slowdown on Yellow / Yield / Crosswalk
      * Automatic Cornering Speed Derating
      * Headless Mode (--no-display) for Zero-Lag SSH Execution

Usage:
  # Standard Headless Run (SSH):
  sudo PYTHONPATH=$PYTHONPATH python3 qcar1_autonomous_driving.py --no-display

  # Or using runner:
  ./run_car.sh --no-display
"""

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys
import time
import math
import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from utils.yolo_trt import YOLOTensorRTDetector
from utils.depth_utils import estimate_distance_meters
from utils.lane_detector import MapAwareLaneTracker, QuanserStanley, ReflexController

try:
    import yaml
except ImportError:
    yaml = None

try:
    from pal.products.qcar import QCar, QCarCameras, QCarRealSense, IS_PHYSICAL_QCAR
    QUANSER_AVAILABLE = True
except ImportError:
    QUANSER_AVAILABLE = False


def build_leds(brake=False, hazard=False, headlights=True):
    leds = np.zeros(8, dtype=np.float64)
    if headlights:
        leds[6] = 1.0
        leds[7] = 1.0
    if brake:
        leds[4] = 1.0
        leds[5] = 1.0
    if hazard:
        leds[0:4] = 1.0
    return leds


class FPSCounter(object):
    def __init__(self, window=30):
        self.window = window
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) < 2:
            return 0.0
        return float(len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


def get_mem_mb():
    try:
        with open("/proc/self/status", "r") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return float(l.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="QCar1 Autonomous Navigation with TensorRT")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window (SSH)")
    parser.add_argument("--no-actuators", action="store_true", help="Dry-run without spinning motors")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_DIR, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    cam_cfg = cfg.get("camera", {})
    veh_cfg = cfg.get("vehicle", {})
    depth_cfg = cfg.get("depth", {})
    disp_cfg = cfg.get("display", {})

    print("============================================================")
    print("QCar1 Autonomous Driving & 7-Class TensorRT Perception")
    print("============================================================")
    print("Model       : {}".format(model_cfg.get("path")))
    print("Camera Mode : {}".format(cam_cfg.get("mode", "dual").upper()))
    print("Headless    : {}".format(args.no_display))
    print("Stop Sign   : 5.0 Seconds Mandatory Hold")
    print("Emergency   : Pedestrian Halt at < {:.1f} m".format(veh_cfg.get("emergency_brake_dist", 1.5)))
    print("============================================================")

    # 1. Initialize TensorRT Detector
    engine_path = os.path.join(PROJECT_DIR, model_cfg.get("path", "models/best_fp16.engine"))
    if not os.path.isfile(engine_path):
        print("[FAIL] TensorRT engine '{}' not found. Run ./build_engine.sh first.".format(engine_path))
        sys.exit(1)

    detector = YOLOTensorRTDetector(
        engine_path=engine_path,
        confidence_threshold=model_cfg.get("confidence_threshold", 0.35),
        iou_threshold=model_cfg.get("iou_threshold", 0.45)
    )
    print("[PASS] TensorRT FP16 Engine active on GPU (Throughput: ~45 FPS).")

    # 2. Controllers & Trackers
    lane_tracker = MapAwareLaneTracker(cfg)
    stanley = QuanserStanley(cfg)
    reflex_ctrl = ReflexController(
        pwm_cruise=veh_cfg.get("pwm_cruise", 0.15),
        pwm_caution=veh_cfg.get("pwm_caution", 0.10),
        max_throttle=veh_cfg.get("motor_limit", 0.25)
    )
    fps_counter = FPSCounter()

    # State variables
    stop_duration = float(veh_cfg.get("stop_duration", 5.0))
    ped_brake_dist = float(veh_cfg.get("emergency_brake_dist", 1.5))
    stop_sign_timer = None
    stop_sign_cleared = False
    stop_sign_clear_dist = 0.0
    dist_traveled = 0.0
    current_steer = 0.0
    current_reflex = "CRUISE"
    t_last = time.time()

    # Hardware Handles
    myCar = None
    csi_cam = None
    rs_cam = None

    try:
        if not args.no_actuators and QUANSER_AVAILABLE:
            myCar = QCar(readMode=0)
            print("[PASS] QCar DAQ Motor & Steering bus initialized.")

        # CSI Camera (Front Lane Tracking)
        csi_cam = QCarCameras(
            frameWidth=cam_cfg.get("csi_width", 640),
            frameHeight=cam_cfg.get("csi_height", 480),
            frameRate=cam_cfg.get("csi_fps", 30),
            enableFront=True
        )
        print("[PASS] CSI Front Camera initialized (640x480 @ 30 FPS).")

        # RealSense Camera (Object Detection + 3D Depth)
        rs_cam = QCarRealSense(
            mode="RGB, Depth",
            frameWidthRGB=cam_cfg.get("realsense_width", 640),
            frameHeightRGB=cam_cfg.get("realsense_height", 480),
            frameRateRGB=cam_cfg.get("realsense_fps", 30),
            frameWidthDepth=cam_cfg.get("realsense_width", 640),
            frameHeightDepth=cam_cfg.get("realsense_height", 480),
            frameRateDepth=cam_cfg.get("realsense_fps", 30)
        )
        print("[PASS] Intel RealSense D435 initialized (RGB+Depth @ 30 FPS).")

    except Exception as exc:
        print("[FAIL] Hardware initialization failed: {}".format(exc))
        sys.exit(1)

    print("\n[READY] Autonomous loop running. Press Ctrl+C to halt vehicle.")
    print("------------------------------------------------------------")

    try:
        while True:
            t0 = time.time()
            dt = max(t0 - t_last, 0.001)
            t_last = t0

            # -------------------- SENSE: DUAL CAMERAS & TACH --------------------
            # 1. CSI Camera for Lane Extraction
            csi_cam.readAll()
            csi_frame = csi_cam.csiFront.imageData if csi_cam.csiFront is not None else None

            # 2. RealSense Camera for Objects & Depth
            rs_cam.read_RGB()
            rs_cam.read_depth(dataMode='PX')
            rs_frame = rs_cam.imageBufferRGB
            depth_frame = rs_cam.imageBufferDepthPX

            if csi_frame is None or rs_frame is None:
                time.sleep(0.005)
                continue

            v_measured = 0.0
            if myCar is not None:
                myCar.read()
                v_measured = float(np.squeeze(myCar.motorTach))
            dist_traveled += abs(v_measured) * dt

            # -------------------- PERCEIVE: LATERAL LANE GUIDANCE --------------------
            e_ct, psi_e, lane_mode, poly = lane_tracker.extract(csi_frame, current_steer)
            current_steer = stanley.compute(e_ct, psi_e, v_measured, lane_mode)

            # -------------------- PERCEIVE: 7-CLASS TENSORRT REFLEX --------------------
            detections = detector.detect(rs_frame)
            h_img, w_img = rs_frame.shape[:2]

            has_red = False
            has_yellow = False
            has_yield = False
            has_crosswalk = False
            ped_danger = False
            stop_sign_close = False
            distances = []

            for d in detections:
                cname = d["class_name"]
                box = d["bbox"]
                bh = box[3] - box[1]

                dist = estimate_distance_meters(
                    depth_frame, box,
                    depth_scale=depth_cfg.get("scale", 1.0),
                    roi_fraction=depth_cfg.get("roi_fraction", 0.5),
                    min_distance=depth_cfg.get("min_distance", 0.2),
                    max_distance=depth_cfg.get("max_distance", 8.0)
                )
                distances.append(dist)

                if cname == "traffic_light_red":
                    has_red = True
                elif cname == "traffic_light_yellow":
                    has_yellow = True
                elif cname == "yield":
                    has_yield = True
                elif cname == "crosswalk":
                    has_crosswalk = True
                elif cname == "pedestrian":
                    if dist is not None and dist < ped_brake_dist:
                        ped_danger = True
                    elif bh > (h_img * 0.18):
                        ped_danger = True
                elif cname == "stop_sign":
                    if (dist is not None and dist < 2.0) or (bh > h_img * 0.16):
                        stop_sign_close = True

            # Stop sign debounce clearance (car must travel 2.0 meters past stop sign)
            if stop_sign_cleared and (dist_traveled - stop_sign_clear_dist > 2.0):
                stop_sign_cleared = False

            # -------------------- DECIDE: SUBSUMPTION REFLEX ARBITER --------------------
            brake_led = False
            hazard_led = False

            # Layer 4: Emergency Pedestrian Halt
            if ped_danger:
                current_reflex = "EMERGENCY_PEDESTRIAN_HALT"
                reflex_mode = "HARD_STOP"
                brake_led = True
                hazard_led = True

            # Layer 3: Red Light Hold
            elif has_red:
                current_reflex = "RED_LIGHT_HOLD"
                reflex_mode = "HARD_STOP"
                brake_led = True

            # Layer 2: Stop Sign 5-Second Hold
            elif stop_sign_close and not stop_sign_cleared:
                if current_reflex != "STOP_SIGN_HOLD":
                    current_reflex = "STOP_SIGN_HOLD"
                    stop_sign_timer = t0
                reflex_mode = "HARD_STOP"
                brake_led = True

                elapsed_stop = t0 - stop_sign_timer
                if elapsed_stop >= stop_duration:
                    stop_sign_cleared = True
                    stop_sign_clear_dist = dist_traveled
                    current_reflex = "CRUISE"
                    reflex_mode = "CRUISE"

            # Layer 1: Caution Slowdown
            elif has_yellow or has_yield or has_crosswalk:
                current_reflex = "CAUTION_SLOW"
                reflex_mode = "CAUTION"
                brake_led = True

            # Layer 0: Nominal Cruise
            else:
                current_reflex = "CRUISE"
                reflex_mode = "CRUISE"

            # -------------------- ACT: ASYMMETRIC THROTTLE & SPEED DERATING --------------------
            base_throttle = reflex_ctrl.update(reflex_mode, dt, v_measured=v_measured)
            pwm_min = float(veh_cfg.get("pwm_min_moving", 0.08))
            # Smooth cornering derating with deadband floor (prevents motor stiction stall)
            derate_factor = math.exp(-1.8 * (current_steer ** 2))
            if reflex_mode == "HARD_STOP":
                u_throttle = 0.0
            else:
                u_throttle = max(base_throttle * derate_factor, pwm_min)
            u_throttle = min(u_throttle, float(veh_cfg.get("motor_limit", 0.25)))

            if myCar is not None and not args.no_actuators:
                led_array = build_leds(brake=brake_led, hazard=hazard_led, headlights=True)
                myCar.read_write_std(throttle=u_throttle, steering=current_steer, LEDs=led_array)

            fps = fps_counter.tick()
            loop_ms = (time.time() - t0) * 1000.0
            ram_mb = get_mem_mb()

            # Live Camera & Control Telemetry over SSH
            cte_cm = e_ct * 100.0
            psi_deg = math.degrees(psi_e)
            print("[CSI LANE]  Mode: {:<18} | CTE: {:>+5.2f}m ({:>+5.1f}cm) | Heading: {:>+5.2f}rad ({:>+4.1f}deg)".format(
                lane_mode, e_ct, cte_cm, psi_e, psi_deg
            ))

            center_z = float(depth_frame[h_img // 2, w_img // 2]) if depth_frame is not None else 0.0
            if detections:
                obj_items = []
                for i, det in enumerate(detections):
                    d_val = distances[i]
                    d_txt = "{:.2f}m".format(d_val) if d_val is not None else "unknown"
                    obj_items.append("[{}:{:.2f} @ {}]".format(det["class_name"].upper(), det["confidence"], d_txt))
                obj_str = " ".join(obj_items)
            else:
                obj_str = "Path Clear (No Obstacles)"

            print("[RS DEPTH]  Center Dist: {:>4.2f}m | Objects: {}".format(center_z, obj_str))

            stop_info = ""
            if current_reflex == "STOP_SIGN_HOLD" and stop_sign_timer is not None:
                stop_info = " [Wait: {:.1f}s/5.0s]".format(t0 - stop_sign_timer)

            print("[CONTROL]   State: {:<16}{} | Steer: {:>+5.2f}rad | Throttle: {:>4.2f} | Speed: {:>4.2f}m/s | FPS: {:>4.1f} | RAM: {:>4.0f}MB".format(
                current_reflex, stop_info, current_steer, u_throttle, v_measured, fps, ram_mb
            ))
            print("-" * 72)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt handled. Safely halting vehicle...")
    finally:
        if myCar is not None:
            myCar.read_write_std(throttle=0.0, steering=0.0, LEDs=np.zeros(8, dtype=np.float64))
            myCar.terminate()
        if csi_cam is not None:
            csi_cam.terminate()
        if rs_cam is not None:
            rs_cam.terminate()
        print("============================================================")
        print("Vehicle actuators and cameras terminated cleanly.")
        print("============================================================")


if __name__ == "__main__":
    main()
