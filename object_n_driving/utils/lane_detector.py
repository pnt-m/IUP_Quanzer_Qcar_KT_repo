#!/usr/bin/env python3
# ============================================================
# QCar1 Lateral Lane Tracking & Stanley Control Module
# ============================================================
"""
Map-aware multi-cue lane extractor and Lyapunov-stable Stanley lateral controller
tailored for Quanser QCar1 and the SDCS physical track environment.
"""

from __future__ import print_function, division, absolute_import

import math
import numpy as np
import cv2


class MapAwareLaneTracker(object):
    """
    Robust multi-cue lane extractor with dual-lane, single-lane, and color fallbacks.
    Mitigates:
      1. Single-boundary visibility: Handled for both left (yellow) and right (white).
      2. Worn paint / lighting variations: Canny + Hough tuned with HSV yellow centroid fallback.
      3. Horizontal crosswalk line rejection: Slope deadzone (|slope| > 0.12).
      4. Optical dropouts: 6-frame temporal persistence buffer prevents rapid state flickering.
    """
    def __init__(self, cfg):
        cam_cfg = cfg.get("camera", {})
        lat_cfg = cfg.get("lateral", {})

        self.w = cam_cfg.get("csi_width", 640)
        self.h = cam_cfg.get("csi_height", 480)
        self.slope_deadzone = lat_cfg.get("slope_deadzone", 0.12)
        self.m_per_pix = lat_cfg.get("m_per_pix", 0.00125)
        self.curb_threshold = lat_cfg.get("curb_threshold", 215)
        self.curvature_decay = lat_cfg.get("curvature_decay", 0.85)

        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Quanser track yellow line HSV range (calibrated for track lighting)
        self.yellow_lo = np.array([16, 60, 60], dtype=np.uint8)
        self.yellow_hi = np.array([45, 255, 255], dtype=np.uint8)

        # White lane brightness threshold
        self.white_thresh = 180

        # Temporal persistence buffer
        self.last_target_x = self.w * 0.5
        self.last_psi = 0.0
        self.persistence_count = 0
        self.max_persistence = 6

    def extract(self, frame, current_steer):
        h, w = self.h, self.w
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Stable Trapezoidal Forward Road ROI (Wide base, narrow horizon)
        poly = np.array([[
            (0, h),
            (w, h),
            (int(w * 0.88), int(h * 0.55)),
            (int(w * 0.12), int(h * 0.55))
        ]], dtype=np.int32)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, poly, 255)

        # 2. Canny Edges + Probabilistic Hough Lines
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, roi_mask)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=45)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = float(y2 - y1) / float(x2 - x1)
                intercept = float(y1 - slope * x1)

                # Reject horizontal crosswalk / stop lines
                if abs(slope) < self.slope_deadzone:
                    continue

                if slope < -self.slope_deadzone:
                    left_lines.append((slope, intercept))
                elif slope > self.slope_deadzone:
                    right_lines.append((slope, intercept))

        # 3. Yellow Dividing Line Color Mask
        mask_yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        mask_yellow = cv2.bitwise_and(mask_yellow, roi_mask)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.k_close)
        pts_y = cv2.findNonZero(mask_yellow)

        # 4. Curb / High Contrast Mask (Roundabout Fallback)
        _, mask_curb = cv2.threshold(gray, self.curb_threshold, 255, cv2.THRESH_BINARY)
        mask_curb = cv2.bitwise_and(mask_curb, roi_mask)
        pts_c = cv2.findNonZero(mask_curb)

        y_la = int(0.75 * h)
        x_center = w * 0.5
        lane_width_px = 350.0  # approximate lane width in pixels at lookahead horizon
        half_lane_px = lane_width_px * 0.5

        mode = "BLIND"
        e_ct = 0.0
        psi_e = 0.0
        target_x = None

        # Tier 1: Dual Optical Lines Detected
        if len(left_lines) > 0 and len(right_lines) > 0:
            fit_l = np.mean(left_lines, axis=0)
            fit_r = np.mean(right_lines, axis=0)
            x_l = (y_la - fit_l[1]) / fit_l[0]
            x_r = (y_la - fit_r[1]) / fit_r[0]
            target_x = (x_l + x_r) * 0.5
            psi_e = math.atan((fit_l[0] + fit_r[0]) * 0.5)
            mode = "OPTICAL_DUAL"

        # Tier 2: Left Line Only (Yellow divider)
        elif len(left_lines) > 0:
            fit_l = np.mean(left_lines, axis=0)
            x_l = (y_la - fit_l[1]) / fit_l[0]
            target_x = x_l + half_lane_px
            psi_e = math.atan(fit_l[0])
            mode = "OPTICAL_LEFT"

        # Tier 3: Right Line Only (White outer boundary)
        elif len(right_lines) > 0:
            fit_r = np.mean(right_lines, axis=0)
            x_r = (y_la - fit_r[1]) / fit_r[0]
            target_x = x_r - half_lane_px
            psi_e = math.atan(fit_r[0])
            mode = "OPTICAL_RIGHT"

        # Tier 4: Color Mask Fallback (Yellow line pixels)
        elif pts_y is not None and len(pts_y) > 35:
            pts_arr = pts_y.reshape(-1, 2)
            y_filter = np.abs(pts_arr[:, 1] - y_la) < 60
            if np.any(y_filter):
                yellow_x = float(np.mean(pts_arr[y_filter, 0]))
            else:
                yellow_x = float(np.mean(pts_arr[:, 0]))
            target_x = yellow_x + half_lane_px
            psi_e = 0.0
            mode = "COLOR_YELLOW"

        # Tier 5: Roundabout Inner Curb Ring
        elif pts_c is not None and len(pts_c) > 30:
            pts_c_arr = pts_c.reshape(-1, 2)
            curb_x = float(np.mean(pts_c_arr[:, 0]))
            target_x = curb_x + 140.0
            psi_e = 0.15
            mode = "ROUNDABOUT_CURB"

        # Tier 6: Temporal Persistence (hold track during brief 1-5 frame optical gap)
        elif self.persistence_count < self.max_persistence:
            target_x = self.last_target_x
            psi_e = self.last_psi
            self.persistence_count += 1
            mode = "PERSISTENCE"

        # Tier 7: Blind Intersection (no lane visible)
        else:
            target_x = x_center
            psi_e = 0.0
            mode = "BLIND_INTERSECTION"

        if target_x is not None and mode not in ["PERSISTENCE", "BLIND_INTERSECTION"]:
            self.last_target_x = target_x
            self.last_psi = psi_e
            self.persistence_count = 0

        e_ct = (target_x - x_center) * self.m_per_pix
        return e_ct, psi_e, mode, poly


class QuanserStanley(object):
    """Lyapunov-Stable Stanley Lateral Guidance Controller with Trim Injection."""
    def __init__(self, cfg):
        lat_cfg = cfg.get("lateral", {})
        veh_cfg = cfg.get("vehicle", {})

        self.k = lat_cfg.get("stanley_k", 1.15)
        self.v_soft = lat_cfg.get("stanley_v_soft", 0.05)
        self.max_steer = veh_cfg.get("max_steer", 0.42)
        self.rate_limit = veh_cfg.get("steer_rate_limit", 0.04)
        self.steer_bias = veh_cfg.get("steer_bias", 0.05)
        self.decay = lat_cfg.get("curvature_decay", 0.85)
        self.prev_steer = 0.0

    def compute(self, e_ct, psi_e, v_measured, mode):
        if mode == "BLIND_INTERSECTION":
            steer_target = self.prev_steer * self.decay
        else:
            v_eff = max(abs(v_measured), 0.001)
            cte_term = math.atan2(self.k * e_ct, v_eff + self.v_soft)
            steer_target = -(psi_e + cte_term)

        # Anti-surge slew rate limiter
        steer_rate_bounded = float(np.clip(
            steer_target,
            self.prev_steer - self.rate_limit,
            self.prev_steer + self.rate_limit
        ))

        # Hardware saturation & steering bias (+0.05 rad)
        steer_out = float(np.clip(
            steer_rate_bounded,
            -self.max_steer,
            self.max_steer
        )) + self.steer_bias

        self.prev_steer = steer_rate_bounded
        return steer_out


class ReflexController(object):
    """Asymmetric Slew-Rate Throttle Smoother with Instant Safety Bypass."""
    def __init__(self, pwm_cruise=0.15, pwm_caution=0.10, max_throttle=0.25,
                 ramp_up_rate=0.35, ramp_down_rate=0.60):
        self.pwm_cruise = pwm_cruise
        self.pwm_caution = pwm_caution
        self.max_throttle = max_throttle
        self.ramp_up_rate = ramp_up_rate
        self.ramp_down_rate = ramp_down_rate
        self.current_pwm = 0.0

    def reset(self):
        self.current_pwm = 0.0

    def update(self, reflex_state, dt, v_measured=0.0):
        dt_eff = float(np.clip(dt, 0.001, 0.5))

        if reflex_state == "HARD_STOP":
            self.current_pwm = 0.0
            return 0.0
        elif reflex_state == "CAUTION":
            target_pwm = self.pwm_caution
        else:
            target_pwm = self.pwm_cruise

        if target_pwm > self.current_pwm:
            step = self.ramp_up_rate * dt_eff
            self.current_pwm = min(target_pwm, self.current_pwm + step)
        else:
            step = self.ramp_down_rate * dt_eff
            self.current_pwm = max(target_pwm, self.current_pwm - step)

        self.current_pwm = float(np.clip(self.current_pwm, 0.0, self.max_throttle))
        return self.current_pwm
