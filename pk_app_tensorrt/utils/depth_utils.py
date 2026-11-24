# ============================================================
# QCar1 YOLO Pedestrian Detection - Depth processing
# ============================================================
"""
Utilities to estimate a pedestrian's distance from a
QCarRealSense depth frame and a person's bounding box.

IMPORTANT - display-normalized vs. metric depth
-------------------------------------------------
The Quanser example scripts show depth on screen using:

    display_frame = myCam.imageBufferDepthPX / max_distance

That division is ONLY a cosmetic normalization so the depth
image can be shown with cv2.imshow() (which expects roughly
0..1 float or 0..255 8-bit values). It does NOT establish the
physical unit of `myCam.imageBufferDepthPX`.

This module keeps that normalization completely separate from
metric distance calculation. Metric distance is computed as:

    metric_meters = raw_depth_value * depth_scale

`depth_scale` MUST be verified by the user against a real,
physically-measured distance (see README.md, "Understanding
Depth") before the printed distance values can be trusted. This
module does NOT invent or guess a conversion factor - it uses
whatever `depth_scale` is provided via config.yaml (default:
1.0, i.e. "unscaled / unverified").
"""

import numpy as np


def extract_roi_depth(depth_frame, bbox, roi_fraction=0.5):
    """
    Extract the raw (unscaled) depth values from an inner region
    of interest inside a bounding box, with invalid/zero values
    removed and simple outlier rejection applied.

    Args:
        depth_frame: 2D numpy array of raw depth values
                     (as returned by myCam.imageBufferDepthPX).
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        roi_fraction: fraction (0-1] of the box width/height to
                      sample, centered on the box center. Using
                      an inner region avoids picking up
                      background pixels near the box edges.

    Returns:
        A 1D numpy array of valid, filtered raw depth samples.
        May be empty if no valid samples were found.
    """
    if depth_frame is None:
        return np.array([], dtype=np.float32)

    frame_h, frame_w = depth_frame.shape[:2]

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0

    roi_fraction = max(min(roi_fraction, 1.0), 0.05)
    roi_w = box_w * roi_fraction
    roi_h = box_h * roi_fraction

    rx1 = int(round(cx - roi_w / 2.0))
    rx2 = int(round(cx + roi_w / 2.0))
    ry1 = int(round(cy - roi_h / 2.0))
    ry2 = int(round(cy + roi_h / 2.0))

    rx1 = max(0, min(rx1, frame_w - 1))
    rx2 = max(0, min(rx2, frame_w))
    ry1 = max(0, min(ry1, frame_h - 1))
    ry2 = max(0, min(ry2, frame_h))

    if rx2 <= rx1 or ry2 <= ry1:
        return np.array([], dtype=np.float32)

    roi = depth_frame[ry1:ry2, rx1:rx2].astype(np.float32).flatten()

    # Remove invalid values: NaN, Inf, and exact zero (RealSense
    # commonly uses 0 to indicate "no valid depth" for a pixel).
    valid_mask = np.isfinite(roi) & (roi > 0)
    roi = roi[valid_mask]

    if roi.size == 0:
        return roi

    # Simple outlier rejection: drop samples further than 2
    # standard deviations from the median, but only if we have
    # enough samples for this to be meaningful.
    if roi.size >= 8:
        median = np.median(roi)
        std = np.std(roi)
        if std > 0:
            outlier_mask = np.abs(roi - median) <= (2.0 * std)
            filtered = roi[outlier_mask]
            if filtered.size > 0:
                roi = filtered

    return roi


def estimate_distance_meters(depth_frame, bbox, depth_scale=1.0,
                              roi_fraction=0.5, min_distance=0.2,
                              max_distance=10.0):
    """
    Estimate a single metric distance (in meters) for the object
    inside `bbox`, using the median of a filtered ROI of raw
    depth samples multiplied by `depth_scale`.

    Returns:
        float distance in meters, or None if no valid depth
        samples were available in range.
    """
    raw_samples = extract_roi_depth(depth_frame, bbox, roi_fraction=roi_fraction)
    if raw_samples.size == 0:
        return None

    raw_median = float(np.median(raw_samples))
    metric_meters = raw_median * float(depth_scale)

    if metric_meters < min_distance or metric_meters > max_distance:
        return None

    return metric_meters


def normalize_for_display(depth_frame, max_distance=2.0):
    """
    Reproduce the Quanser example's COSMETIC normalization for
    on-screen display purposes only:

        display_frame = imageBufferDepthPX / max_distance

    This is NOT a metric conversion - see module docstring. Use
    estimate_distance_meters() for actual distance values.
    """
    if depth_frame is None:
        return None
    max_distance = max(float(max_distance), 1e-6)
    normalized = depth_frame.astype(np.float32) / max_distance
    return np.clip(normalized, 0.0, 1.0)
