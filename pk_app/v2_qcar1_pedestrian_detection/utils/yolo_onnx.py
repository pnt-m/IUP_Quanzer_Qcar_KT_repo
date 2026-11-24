# ============================================================
# QCar1 YOLO Pedestrian Detection - ONNX detector
# ============================================================
"""
YOLOv5-style ONNX Runtime detector.

Implements preprocessing (letterbox resize + normalize),
inference via onnxruntime, decoding of the standard YOLOv5
export output tensor (1 x N x (5 + num_classes)), confidence
thresholding, per-class Non-Maximum Suppression, and drawing
utilities.

Written to be compatible with Python 3.6 (no f-string-only
features beyond 3.6 support, no dataclasses, no walrus operator).
"""

import time

import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required but could not be imported. "
        "It is expected to already be installed in the QCar1 "
        "system environment (version 1.10.0). Original error: "
        "{}".format(exc)
    )


# COCO class names (80 classes), index 0 = person.
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize an image to new_shape while preserving aspect ratio,
    padding the remainder with a constant color (YOLOv5-style
    letterbox). Returns the padded image plus the scale factor
    and (left, top) padding, all needed to map boxes back to the
    original image coordinates.
    """
    h0, w0 = image.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / float(w0), new_h / float(h0))
    resized_w, resized_h = int(round(w0 * scale)), int(round(h0 * scale))

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if (w0, h0) != (resized_w, resized_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, scale, pad_left, pad_top


def _xywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) boxes (Nx4 array) to (x1, y1, x2, y2)."""
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # y2
    return xyxy


class YOLOONNXDetector(object):
    """
    Loads a YOLOv5-style ONNX model and runs person detection.

    Expected model output shape: (1, N, 5 + num_classes) where
    each row is [cx, cy, w, h, objectness, class_0_score, ...].
    This matches the standard `torch.hub` / `export.py --include
    onnx` export from Ultralytics YOLOv5 (opset 12).
    """

    def __init__(self, model_path, confidence_threshold=0.40,
                 iou_threshold=0.45, person_class_id=0,
                 input_width=640, input_height=640):
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.person_class_id = int(person_class_id)

        so = ort.SessionOptions()
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=so)
        except Exception as exc:
            exc_str = str(exc)
            hint = ""
            if "opset" in exc_str.lower() and ("official support" in exc_str.lower() or "under development" in exc_str.lower()):
                hint = (
                    " This looks like an unsupported ONNX opset version "
                    "(ONNX Runtime 1.10.0 supports up to opset 15). "
                    "Re-export the model with --opset 12 on a machine with "
                    "internet access and copy it over - see README.md "
                    "section 9 / 18."
                )
            raise RuntimeError(
                "Failed to load ONNX model at '{}'. Make sure "
                "download_model.sh has been run successfully.{} "
                "Original error: {}".format(self.model_path, hint, exc_str)
            )

        self._input_meta = self.session.get_inputs()[0]
        self.input_name = self._input_meta.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Determine input size: prefer the size baked into the model,
        # fall back to the config-provided size if the model reports
        # dynamic ('None' / string) dimensions.
        shape = self._input_meta.shape  # e.g. [1, 3, 640, 640]
        model_h, model_w = None, None
        if len(shape) == 4:
            h_candidate, w_candidate = shape[2], shape[3]
            if isinstance(h_candidate, int):
                model_h = h_candidate
            if isinstance(w_candidate, int):
                model_w = w_candidate

        self.input_height = model_h if model_h else int(input_height)
        self.input_width = model_w if model_w else int(input_width)

        self.class_names = COCO_CLASS_NAMES

        self.last_inference_ms = 0.0

    def get_input_size(self):
        """Return (width, height) expected by the model."""
        return self.input_width, self.input_height

    def get_providers(self):
        return self.session.get_providers()

    def _preprocess(self, image_bgr):
        padded, scale, pad_left, pad_top = _letterbox(
            image_bgr, new_shape=(self.input_height, self.input_width)
        )
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, axis=0)  # add batch dim
        return img, scale, pad_left, pad_top

    def _nms_per_class(self, boxes_xyxy, scores, class_ids):
        """
        Run OpenCV's NMS independently per class so that, e.g., a
        person box is never suppressed by a nearby car box.
        Returns the indices (into the input arrays) that survive.
        """
        keep_indices = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = np.where(class_ids == cls)[0]
            if cls_mask.size == 0:
                continue
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            # cv2.dnn.NMSBoxes expects [x, y, w, h]
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

            # cv2.dnn.NMSBoxes may return a flat array or nested arrays
            # depending on the OpenCV version - normalize both cases.
            flat_indices = np.array(indices).reshape(-1)
            for local_idx in flat_indices:
                keep_indices.append(cls_mask[local_idx])

        return keep_indices

    def detect(self, image_bgr, persons_only=True):
        """
        Run detection on a single BGR image (as returned by
        QCarRealSense / cv2).

        Returns a list of dicts:
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]   # in original image pixel coords
            }
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        orig_h, orig_w = image_bgr.shape[:2]

        input_tensor, scale, pad_left, pad_top = self._preprocess(image_bgr)

        t0 = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        self.last_inference_ms = (time.time() - t0) * 1000.0

        predictions = outputs[0]
        # Normalize shape to (N, 5 + num_classes)
        if predictions.ndim == 3:
            predictions = predictions[0]

        if predictions.shape[0] == 0:
            return []

        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]

        # Quick pre-filter on objectness to reduce work before argmax.
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

        # Map boxes from the padded/letterboxed input space back to
        # the original image pixel space.
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

        # Sort by confidence, highest first.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def draw_detections(self, image_bgr, detections, distances=None,
                         highlight_index=None):
        """
        Draw bounding boxes, class label, confidence, and
        (optionally) distance text onto a copy of image_bgr.

        distances: optional list, same length/order as detections,
                   of distance-in-meters floats (or None per entry).
        highlight_index: optional index into detections to draw in
                   a distinct highlight color (e.g. nearest person).
        """
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
