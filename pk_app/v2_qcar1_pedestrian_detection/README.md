# QCar1 YOLO Pedestrian Detection

Real-time pedestrian detection and distance estimation for the **Quanser
QCar1** (NVIDIA Jetson TX2), using a YOLO ONNX model run through
**ONNX Runtime** and the QCar1's RealSense RGB+Depth camera.

This project is designed to run **inside the existing QCar1 software
environment without modifying it in any way** (no Python upgrade, no
CUDA/TensorRT reinstall, no OpenCV/ONNX Runtime replacement).

---

## 1. Overview

```
QCar1 RealSense RGB camera
            |
            v
    RGB image frame
            |
            v
       YOLO ONNX  (ONNX Runtime, CPU)
            |
            v
   Person detection
            |
            v
  Bounding box + confidence
            |
            v
   RealSense depth ROI
            |
            v
   Pedestrian distance
```

The application displays, for every detected pedestrian:

* RGB camera image
* Pedestrian bounding box
* Confidence score
* Estimated distance in meters
* FPS
* Number of detected pedestrians

Example console/on-screen output:

```
PERSON 0.92
Distance: 1.84 m
```

---

## 2. Hardware

* Quanser QCar1
* NVIDIA Jetson TX2 / Tegra X2
* ARM64 / aarch64
* RealSense RGB + Depth camera (accessed via `QCarRealSense`)

---

## 3. Existing QCar1 Software Environment

This project targets, and assumes as **already installed**:

| Component        | Version           |
|-------------------|-------------------|
| Ubuntu             | 18.04.6 LTS       |
| Linux kernel        | 4.9.x             |
| NVIDIA L4T          | 32.6.1            |
| CUDA                | 10.2              |
| cuDNN               | 8.2.1             |
| TensorRT            | 8.0.1.6           |
| Python              | 3.6.9             |
| pip                 | 21.3.1            |
| NumPy               | 1.19.4            |
| OpenCV              | 4.5.5 (no CUDA support: `CUDA devices: 0`) |
| ONNX Runtime        | 1.10.0 (`CPUExecutionProvider` only) |
| PyTorch             | 1.10.0 (not required at runtime) |
| torchvision          | 0.11.0 (not required at runtime) |
| Quanser packages    | 2023.4.13         |
| ROS                 | Melodic           |

Because ONNX Runtime currently reports only `CPUExecutionProvider`, and
OpenCV reports `CUDA devices: 0`, all inference in this project's first
version runs on CPU. See section 17 for the future TensorRT path.

---

## 4. Architecture

```
qcar1_pedestrian.py  (main app, uses QCarRealSense)
    |
    +--> utils/yolo_onnx.py   (YOLOONNXDetector: preprocess, infer, NMS, draw)
    |
    +--> utils/depth_utils.py (ROI depth sampling -> metric distance)
    |
    +--> config.yaml          (all tunable parameters)

detect_image.py   (same detector, but on a single saved image - no camera)
test_onnx.py      (validates the ONNX model loads and runs - no camera)
test_environment.py (validates the whole software stack)
download_model.sh (fetches the ONNX model into models/)
setup_venv.sh     (isolated venv, --system-site-packages)
```

---

## 5. Project Directory

```
qcar1_pedestrian_detection/
├── README.md
├── requirements.txt
├── config.yaml
├── setup_venv.sh
├── download_model.sh
├── test_environment.py
├── test_onnx.py
├── detect_image.py
├── qcar1_pedestrian.py
├── utils/
│   ├── __init__.py
│   ├── yolo_onnx.py
│   └── depth_utils.py
├── models/
│   └── .gitkeep      (model goes here after download_model.sh)
├── test_images/
│   └── .gitkeep      (put your own test .jpg/.png files here)
└── output/
    └── .gitkeep      (annotated results are saved here)
```

---

## 6. Safety / Do Not Modify Existing Environment

This project intentionally:

* Does **not** upgrade Python, pip, NumPy, OpenCV, ONNX Runtime,
  TensorRT, PyTorch, torchvision, Quanser packages, or ROS.
* Does **not** replace the QCar1's camera API (`QCarRealSense`) with
  `cv2.VideoCapture`, direct `/dev/video` access, `pyrealsense2`, or any
  other RealSense SDK.
* Uses an **isolated virtual environment** (`--system-site-packages`) so
  the only new thing installed is `PyYAML`, and only inside that venv.
* Never auto-converts the ONNX model to TensorRT; that conversion is
  documented as an optional, manual, future step (section 17).

If `setup_venv.sh` finds that a required system package is missing, it
**reports** this rather than attempting to install or replace it.

---

## 7. Setup

Copy this whole project folder onto the QCar1, then run the steps below
**in order** from inside the project directory.

---

## 8. Create Virtual Environment

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

This creates `venv/` using `python3 -m venv --system-site-packages venv`,
so it can see the already-installed `numpy`, `cv2`, `onnxruntime`, and
`pal`/Quanser packages without reinstalling them, and installs only
`PyYAML` (needed to parse `config.yaml`).

Activate it for the rest of your session:

```bash
source venv/bin/activate
```

---

## 9. Download YOLO ONNX Model

```bash
chmod +x download_model.sh
./download_model.sh
```

**Model information:**

| Field                | Value |
|------------------------|-------|
| Architecture            | YOLOv5n (nano) |
| Source                  | Ultralytics YOLOv5, COCO-pretrained |
| License                 | AGPL-3.0 (Ultralytics YOLOv5) — verify current terms at the source repository before commercial use |
| Input                   | `1x3x640x640`, RGB, float32, normalized to `0..1` |
| Output                  | `1x25200x85` = `(cx, cy, w, h, objectness, 80 class scores)` |
| COCO person class id    | `0` |
| Expected preprocessing  | Letterbox resize to 640x640 (aspect-ratio preserving + padding), BGR→RGB, `/255.0`, HWC→CHW |
| Expected postprocessing | objectness × class score threshold, per-class NMS, map boxes back through the letterbox transform |

The download URL is a variable at the top of `download_model.sh`
(`MODEL_URL`) — edit it there if the default link has moved.

> **Known issue:** the default v7.0 release asset has been observed to
> be exported at **ONNX opset 17**, which ONNX Runtime 1.10.0 (installed
> on this QCar1) does **not** support (it officially supports up to
> opset 15). If `python3 test_onnx.py` fails with an error mentioning
> "opset", re-export the model at opset 12 on a separate machine with
> internet access, then copy the result over manually:
>
> ```bash
> git clone https://github.com/ultralytics/yolov5.git
> cd yolov5 && git checkout v6.1
> pip install -r requirements.txt
> python export.py --weights yolov5n.pt --include onnx --opset 12 --imgsz 640
> scp yolov5n.onnx <user>@<qcar1-ip>:<project>/models/yolov5n.onnx
> ```
>
> `test_onnx.py` and `qcar1_pedestrian.py` both detect this specific
> error and print these same instructions automatically.

`download_model.sh`:

* Creates `models/` if missing.
* Never silently overwrites an existing model (it asks first).
* Verifies the download actually produced a real file (size sanity
  check), not an HTML error page.
* Prints model path, size, name, and expected input size on success.
* Prints a clear manual-download procedure if there is no internet
  access (see section 18).

---

## 10. Verify Environment

```bash
python3 test_environment.py
```

Example output:

```
[PASS] Python 3.6
[PASS] NumPy
[PASS] OpenCV
[PASS] ONNX Runtime
[INFO] ONNX Runtime is CPU-only.
[INFO] TensorRT can be used for future GPU acceleration.
[PASS] TensorRT
[PASS] Quanser 'pal' package
[PASS] QCarRealSense import
```

A CPU-only ONNX Runtime is **expected**, not an error. TensorRT/Quanser
failures are expected and non-fatal if you're testing on a dev machine
that isn't the QCar1 itself.

---

## 11. Test ONNX Model

```bash
python3 test_onnx.py
```

This loads the model, prints its input/output signature and available
execution providers, runs one dummy inference, and reports timing. It
does **not** need the QCar camera.

---

## 12. Test with Image

Put a `.jpg`/`.png` file in `test_images/`, then:

```bash
python3 detect_image.py --image test_images/test.jpg
```

Optional flags:

```bash
python3 detect_image.py \
    --image test_images/test.jpg \
    --model models/yolov5n.onnx \
    --output output/result.jpg \
    --confidence 0.40 \
    --iou 0.45 \
    --show
```

`--show` opens a display window (only works if a GUI/display is
available — see Troubleshooting).

---

## 13. Run QCar1 RealSense Pedestrian Detection

```bash
python3 qcar1_pedestrian.py
```

This is the main application. It opens `QCarRealSense`, runs the
detection + distance loop, and displays annotated results until you
press **`q`** in the display window (or **Ctrl+C** in the terminal).

Use `--no-display` to run headless (console output only):

```bash
python3 qcar1_pedestrian.py --no-display
```

---

## 14. Understanding Depth

The QCar1's example scripts display depth like this:

```python
max_distance = 2
display_frame = myCam.imageBufferDepthPX / max_distance
```

**This is a cosmetic normalization for `cv2.imshow()`, not a metric
conversion.** It does not by itself tell you the physical unit of
`imageBufferDepthPX`.

This project keeps the two concepts separate:

* `utils.depth_utils.normalize_for_display()` reproduces the cosmetic
  0–1 normalization, used only if `display.show_depth: true`.
* `utils.depth_utils.estimate_distance_meters()` computes
  `metric_meters = median(filtered ROI raw depth) * depth_scale`.

`depth_scale` lives in `config.yaml` under `depth.scale` and defaults to
**`1.0`, which is an unverified placeholder**, not a calibrated value.

**How to calibrate `depth_scale`:**

1. Place a person (or a flat object) at a known, physically-measured
   distance from the camera — e.g. exactly 1.00 m, measured with a tape
   measure.
2. Run `qcar1_pedestrian.py` and note the **raw** median depth value for
   that bounding box (temporarily add a print of the un-scaled ROI
   median in `utils/depth_utils.py`, or set `depth.scale: 1.0` and read
   the printed "distance" which at scale 1.0 equals the raw value).
3. Compute `depth_scale = known_distance_meters / raw_value`.
4. Repeat at 2–3 different distances to confirm the relationship is
   linear and consistent; average the resulting scale if needed.
5. Put the confirmed value into `config.yaml` under `depth.scale`.

Do not trust the printed "meters" values until this calibration has
been done at least once for your specific QCar1/camera setup.

---

## 15. Multiple Pedestrians

The app supports and displays multiple simultaneous detections:

```
Person 1: 0.92, 1.84 m
Person 2: 0.87, 3.21 m
```

Each is drawn independently. Set `behavior.nearest_person_only: true`
in `config.yaml` to highlight only the closest pedestrian (a display
convenience only — **this is not a safety-critical behavior** and does
not trigger any automatic robot action).

---

## 16. FPS Measurement

The main loop reports:

```
FPS: 12.4
Inference: 72 ms
```

`FPS` is an exponential moving average of the full loop rate (camera
read + inference + depth + draw). `Inference` is just the ONNX Runtime
`session.run()` time. Actual FPS varies with scene complexity, number of
detections, and system load — no specific FPS is guaranteed.

---

## 17. TensorRT Acceleration

**Phase 1 (this release): ONNX Runtime CPU inference.**
**Phase 2 (future, optional): TensorRT engine.**

Currently, ONNX Runtime on this QCar1 reports only
`CPUExecutionProvider`, and OpenCV reports `CUDA devices: 0` — so
TensorRT (already installed, version 8.0.1.6) is the intended future
optimization path, not something this project sets up automatically.

If/when you want to try it, on a machine with TensorRT 8.0.1.6 tools
available, a conversion compatible with that version looks like:

```bash
# Using the trtexec tool bundled with TensorRT 8.0.1.6:
trtexec --onnx=models/yolov5n.onnx \
        --saveEngine=models/yolov5n_trt8.engine \
        --workspace=1024 \
        --fp16
```

Notes:

* Only use flags/commands documented for **TensorRT 8.0.1.6** — do not
  follow instructions written for TensorRT 8.6, 9.x, or 10.x, as flag
  names and behavior differ across major versions.
* `--fp16` is optional; omit it to keep full FP32 precision (slower but
  numerically closer to the ONNX Runtime CPU results).
* Loading and running a `.engine` file requires the TensorRT Python API
  (`tensorrt` + `pycuda`, or the TensorRT `IExecutionContext` API) — this
  is **not implemented** in this initial version, by design, to keep
  Phase 1 simple and dependency-free. `utils/yolo_onnx.py` is structured
  so a `TensorRTDetector` class with the same `detect()` /
  `draw_detections()` / `get_input_size()` interface could be added
  later as a drop-in replacement for `YOLOONNXDetector`.

---

## 18. Offline Installation

The QCar1 may not have reliable internet access. `download_model.sh` is
fully independent of the rest of the project and handles this:

* If a download tool (`curl`/`wget`) is missing, or the download fails,
  it prints a manual procedure instead of failing silently.

**Manual procedure:**

```
PC (with internet)
    |
    | 1. Download the model
    v
yolov5n.onnx
    |
    | 2. Transfer via USB drive, or:
    |      scp yolov5n.onnx <user>@<qcar1-ip>:/path/to/project/models/
    v
QCar1
    |
    v
models/yolov5n.onnx
```

After manually placing the file, skip straight to:

```bash
python3 test_onnx.py
```

---

## 19. Troubleshooting

**ONNX Runtime import failure**
`pip show onnxruntime` inside the venv (with `--system-site-packages`,
it should already be visible). Do not `pip install onnxruntime` — the
system version (1.10.0) is the one qualified for this Jetson TX2 image.

**CPUExecutionProvider only**
Expected on this QCar1 configuration. Not an error — see section 17.

**TensorRT import failure**
Expected if testing off-QCar1, or if the TensorRT Python bindings
aren't on `PYTHONPATH` inside the venv. Not required for Phase 1.

**Quanser import failure**
Confirm you are running on/against the actual QCar1 or a properly
configured QLabs setup, and that the venv was created with
`--system-site-packages`.

**QCarRealSense import failure**
Same as above — check `pal` package installation and that
`setup_venv.sh` completed the "Quanser 'pal' package" check with PASS.

**RealSense RGB not available**
Check the physical camera connection (or QLabs virtual camera
configuration). Try re-running the original Quanser example RealSense
script to confirm the camera works outside this project.

**Depth values are zero**
Some pixels near object edges, reflective/transparent surfaces, or
out-of-range areas commonly report zero depth. `depth_utils.py` already
filters these out; if an entire bounding box returns zero, try a person
standing closer/farther, or increase `depth.roi_fraction`.

**Wrong depth scale**
See section 14 — `depth.scale` defaults to an unverified `1.0`. Follow
the calibration procedure before trusting distance output.

**No person detected**
Lower `model.confidence_threshold` in `config.yaml` (e.g. to `0.25`),
confirm lighting is reasonable, and confirm the model file isn't
corrupted (`test_onnx.py` should pass first).

**Model file missing**
Run `./download_model.sh`, or follow the manual procedure in section 18.

**Insufficient disk space**
`df -h` to check. The YOLOv5n ONNX model is only a few MB; if space is
tight elsewhere, clear old files in `output/`.

**No GUI/display**
Run with `--no-display` (main app) or without `--show` (`detect_image.py`).
Annotated images are still saved to `output/`.

**Python virtual environment problems**
If `python3 -m venv` fails, install the venv module:
`sudo apt-get install python3-venv` (this only adds the `venv` module,
it does not change your Python version).

**Permission problems**
`chmod +x setup_venv.sh download_model.sh` if you get "Permission
denied" when running the scripts directly.

---

## 20. Future Improvements

* TensorRT `IExecutionContext`-based `TensorRTDetector` (drop-in
  replacement for `YOLOONNXDetector`) for GPU-accelerated inference.
* Automatic `depth_scale` calibration routine.
* ROS Melodic topic publishing of detections/distances for integration
  with other QCar1 nodes.
* Optional pyrealsense2-based fallback camera path (explicitly out of
  scope for this initial version — `QCarRealSense` is the primary and
  only implemented camera path).

---

## Quick Reference — Full Workflow

```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate

chmod +x download_model.sh
./download_model.sh

python3 test_environment.py

python3 test_onnx.py

python3 detect_image.py --image test_images/test.jpg

python3 qcar1_pedestrian.py
```
