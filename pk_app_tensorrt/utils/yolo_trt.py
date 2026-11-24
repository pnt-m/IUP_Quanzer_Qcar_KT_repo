# ============================================================
# QCar1 YOLO Pedestrian Detection - TensorRT GPU detector
# ============================================================
"""
TensorRT 8.0.x GPU-accelerated detector for YOLOv5 models.
Provides high-speed inference (25-35+ FPS) on the NVIDIA Jetson TX2
using FP16 precision.

Fully compatible with Python 3.6 on Jetson Linux (L4T 32.6.1).
Implements the exact same interface as YOLOONNXDetector:
  - detect(image_bgr, persons_only=True)
  - draw_detections(image_bgr, detections, distances=None, highlight_index=None)
  - get_input_size()
  - last_inference_ms
"""

from __future__ import print_function

import os
import time
import numpy as np
import cv2

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    cuda = None

from utils.yolo_onnx import _letterbox, _xywh_to_xyxy, COCO_CLASS_NAMES


class YOLOTensorRTDetector(object):
    """
    Loads an FP16 TensorRT engine file (.engine) and runs person detection
    using Jetson TX2 Pascal GPU (sm_62).
    """

    def __init__(self, engine_path, confidence_threshold=0.40,
                 iou_threshold=0.45, person_class_id=0,
                 input_width=640, input_height=640):
        if trt is None:
            raise ImportError(
                "tensorrt Python package is required. On Jetson TX2, ensure "
                "/usr/lib/python3.6/dist-packages is in your PYTHONPATH."
            )
        if cuda is None:
            raise ImportError(
                "pycuda is required for TensorRT inference. "
                "Install it on the QCar with: sudo apt-get install -y python3-pycuda"
            )

        self.engine_path = str(engine_path)
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(
                "TensorRT engine file not found: '{}'.\n"
                "Please run ./build_engine.sh on the Jetson TX2 first to generate it."
                .format(self.engine_path)
            )

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.person_class_id = int(person_class_id)
        self.class_names = COCO_CLASS_NAMES
        self.last_inference_ms = 0.0

        # Initialize TensorRT Logger and Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine from '{}'.".format(self.engine_path))

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context for TensorRT engine.")

        # Allocate page-locked host and device memory for input and output bindings
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
            raise RuntimeError("TensorRT engine has invalid binding configuration.")

        # Identify the primary detection output tensor (shape ending in 85, e.g. 1x25200x85)
        self.main_output = None
        for out in self.outputs:
            if out['name'] == 'output' or (len(out['shape']) >= 2 and out['shape'][-1] == 85 and out['size'] % 85 == 0):
                self.main_output = out
                break
        if self.main_output is None:
            self.main_output = max(self.outputs, key=lambda o: o['size'])

    def get_input_size(self):
        """Return (width, height) expected by the engine."""
        return self.input_width, self.input_height

    def get_providers(self):
        return ["TensorRT Execution Provider (GPU sm_62 FP16)"]

    def _nms_per_class(self, boxes_xyxy, scores, class_ids):
        """Run NMS independently per class using OpenCV."""
        keep_indices = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = np.where(class_ids == cls)[0]
            if cls_mask.size == 0:
                continue
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            nms_boxes = []
            for b in cls_boxes:
                x1, y1, x2, y2 = b
                nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(
                nms_boxes, cls_scores.tolist(),
                self.confidence_threshold, self.iou_threshold
            )
            if indices is None or len(indices) == 0:
                continue

            flat_indices = np.array(indices).reshape(-1)
            for local_idx in flat_indices:
                keep_indices.append(cls_mask[local_idx])

        return keep_indices

    def detect(self, image_bgr, persons_only=True):
        """
        Run TensorRT GPU inference on a single BGR image.
        Returns list of detection dictionaries:
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Preprocess: letterbox resize to engine dimensions
        padded, scale, pad_left, pad_top = _letterbox(
            image_bgr, new_shape=(self.input_height, self.input_width)
        )
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.ascontiguousarray(img)

        # 2. Asynchronous DMA transfer to GPU memory
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['dev'], self.inputs[0]['host'], self.stream)

        # 3. Execute TensorRT Kernel on Jetson GPU
        t0 = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # 4. Asynchronous DMA transfer back to host memory
        cuda.memcpy_dtoh_async(self.main_output['host'], self.main_output['dev'], self.stream)
        self.stream.synchronize()
        self.last_inference_ms = (time.time() - t0) * 1000.0

        # 5. Decode output tensor (reshape to [N, 85])
        raw_output = self.main_output['host']
        if self.main_output['size'] % 85 == 0:
            predictions = raw_output.reshape(-1, 85)
        else:
            predictions = raw_output.reshape(self.main_output['shape'])
            if predictions.ndim == 3:
                predictions = predictions[0]

        if predictions.shape[0] == 0:
            return []

        # 6. Confidence Filtering & NMS
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]

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

        if persons_only:
            person_mask = class_ids == self.person_class_id
            if not np.any(person_mask):
                return []
            boxes_xywh = boxes_xywh[person_mask]
            final_scores = final_scores[person_mask]
            class_ids = class_ids[person_mask]

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

        # Unpad letterbox back to camera frame coordinates
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= scale

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        keep_indices = self._nms_per_class(boxes_xyxy, final_scores, class_ids)

        detections = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            cls_id = int(class_ids[idx])
            cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(final_scores[idx]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def draw_detections(self, image_bgr, detections, distances=None, highlight_index=None):
        """Draw bounding boxes and distances on the image."""
        out = image_bgr.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]

            is_highlighted = (highlight_index is not None and i == highlight_index)
            color = (0, 0, 255) if is_highlighted else (0, 255, 0)
            thickness = 3 if is_highlighted else 2

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label = "{} {:.2f}".format(det["class_name"].upper(), det["confidence"])
            if distances is not None and i < len(distances) and distances[i] is not None:
                label += " | {:.2f} m".format(distances[i])

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(out, (x1, label_y1), (x1 + text_w + 4, label_y1 + text_h + baseline + 4), color, -1)
            cv2.putText(
                out, label, (x1 + 2, label_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return out
