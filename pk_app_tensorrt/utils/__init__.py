"""
QCar1 Pedestrian Detection utilities.
Provides TensorRT GPU detector with fallback to ONNX Runtime detector,
plus depth ROI metric distance estimators.
"""

from utils.yolo_onnx import YOLOONNXDetector, COCO_CLASS_NAMES
from utils.depth_utils import estimate_distance_meters, normalize_for_display

try:
    from utils.yolo_trt import YOLOTensorRTDetector
except ImportError:
    YOLOTensorRTDetector = None

__all__ = [
    "YOLOTensorRTDetector",
    "YOLOONNXDetector",
    "estimate_distance_meters",
    "normalize_for_display",
    "COCO_CLASS_NAMES",
]
