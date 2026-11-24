#!/usr/bin/env python3
# ============================================================
# QCar1 7-Class YOLO TensorRT Detector (NVIDIA Jetson TX2)
# ============================================================
"""
High-performance TensorRT FP16 inference engine for custom 7-class
autonomous driving model on Quanser QCar1 (Jetson TX2 sm_62).

Classes:
  0: crosswalk
  1: pedestrian
  2: stop_sign
  3: traffic_light_green
  4: traffic_light_red
  5: traffic_light_yellow
  6: yield
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import time
import numpy as np
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

DEFAULT_CLASS_NAMES = {
    0: "crosswalk",
    1: "pedestrian",
    2: "stop_sign",
    3: "traffic_light_green",
    4: "traffic_light_red",
    5: "traffic_light_yellow",
    6: "yield"
}

DEFAULT_COLORS = {
    "crosswalk": (255, 200, 0),
    "pedestrian": (0, 140, 255),
    "stop_sign": (0, 0, 255),
    "traffic_light_green": (0, 255, 0),
    "traffic_light_red": (0, 0, 220),
    "traffic_light_yellow": (0, 230, 255),
    "yield": (200, 0, 200)
}


def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = image.shape[:2]
    r = min(float(new_shape[0]) / h0, float(new_shape[1]) / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if (w0, h0) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, dw, dh


def _xywh_to_xyxy(boxes):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


class YOLOTensorRTDetector(object):
    """NVIDIA TensorRT FP16 execution engine for 7-class QCar detection."""

    def __init__(self, engine_path, confidence_threshold=0.35, iou_threshold=0.45,
                 class_names=None, input_width=640, input_height=640):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT or PyCUDA is not installed in this environment.")

        self.engine_path = os.path.abspath(engine_path)
        if not os.path.isfile(self.engine_path):
            raise IOError("TensorRT engine file not found: {}".format(self.engine_path))

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_names = class_names if class_names else DEFAULT_CLASS_NAMES
        self.colors = DEFAULT_COLORS

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine from '{}'.".format(self.engine_path))

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        self.input_width = int(input_width)
        self.input_height = int(input_height)

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))

            binding_dict = {
                'name': binding,
                'host': host_mem,
                'dev': dev_mem,
                'shape': shape,
                'dtype': dtype,
                'size': size
            }

            if self.engine.binding_is_input(binding):
                if len(shape) == 4:
                    self.input_height = shape[2]
                    self.input_width = shape[3]
                self.inputs.append(binding_dict)
            else:
                self.outputs.append(binding_dict)

        if not self.inputs or not self.outputs:
            raise RuntimeError("Invalid engine binding setup.")

        # Identify primary output tensor (shape ending in 12, or largest volume)
        self.main_output = None
        for out in self.outputs:
            if out['name'] in ['output', 'output0'] or (len(out['shape']) >= 2 and out['shape'][-1] == 12):
                self.main_output = out
                break
        if self.main_output is None:
            self.main_output = max(self.outputs, key=lambda o: o['size'])

        self.num_classes = len(self.class_names)
        self.last_inference_ms = 0.0

    def get_input_size(self):
        return self.input_width, self.input_height

    def get_providers(self):
        return ["TensorRT Execution Provider (GPU sm_62 FP16)"]

    def _nms(self, boxes_xyxy, scores, class_ids):
        keep_indices = []
        for cls in np.unique(class_ids):
            cls_mask = np.where(class_ids == cls)[0]
            if cls_mask.size == 0:
                continue
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            nms_boxes = []
            for b in cls_boxes:
                x1, y1, x2, y2 = b
                nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(nms_boxes, cls_scores.tolist(), self.confidence_threshold, self.iou_threshold)
            if indices is not None and len(indices) > 0:
                for idx in np.array(indices).reshape(-1):
                    keep_indices.append(cls_mask[idx])
        return keep_indices

    def detect(self, image_bgr, target_classes=None):
        if image_bgr is None or image_bgr.size == 0:
            return []

        orig_h, orig_w = image_bgr.shape[:2]

        padded, scale, pad_left, pad_top = _letterbox(image_bgr, (self.input_height, self.input_width))
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)

        # 1. DMA H2D
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['dev'], self.inputs[0]['host'], self.stream)

        # 2. Kernel execution
        t0 = time.time()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 3. DMA D2H
        cuda.memcpy_dtoh_async(self.main_output['host'], self.main_output['dev'], self.stream)
        self.stream.synchronize()
        self.last_inference_ms = (time.time() - t0) * 1000.0

        # 4. Decode 25200x12 matrix
        raw = self.main_output['host']
        predictions = raw.reshape(-1, 12) if (self.main_output['size'] % 12 == 0) else raw.reshape(self.main_output['shape'])[0]

        if predictions.shape[0] == 0:
            return []

        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:12]

        obj_mask = obj_conf > self.confidence_threshold
        if not np.any(obj_mask):
            return []

        predictions = predictions[obj_mask]
        obj_conf = obj_conf[obj_mask]
        class_scores = class_scores[obj_mask]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
        final_scores = obj_conf * class_conf

        keep_mask = final_scores > self.confidence_threshold
        if not np.any(keep_mask):
            return []

        boxes_xywh = predictions[keep_mask, 0:4]
        final_scores = final_scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # Filter by target_classes if specified
        if target_classes is not None:
            cls_filter = np.isin(class_ids, target_classes)
            if not np.any(cls_filter):
                return []
            boxes_xywh = boxes_xywh[cls_filter]
            final_scores = final_scores[cls_filter]
            class_ids = class_ids[cls_filter]

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        keep_indices = self._nms(boxes_xyxy, final_scores, class_ids)
        if not keep_indices:
            return []

        detections = []
        for idx in keep_indices:
            cls_id = int(class_ids[idx])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = boxes_xyxy[idx]
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(final_scores[idx]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def draw_detections(self, image_bgr, detections, distances=None):
        out = image_bgr.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            cls_name = det["class_name"]
            color = self.colors.get(cls_name, (0, 255, 0))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = "{} {:.2f}".format(cls_name.upper(), det["confidence"])
            if distances is not None and i < len(distances) and distances[i] is not None:
                label += " | {:.2f}m".format(distances[i])

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(y1 - th - 4, 0)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, max(y1 - 2, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return out
