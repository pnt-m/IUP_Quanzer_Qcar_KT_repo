# QCar1 Embedded Robotics Command Reference

A comprehensive inventory of operational commands, diagnostic recipes, model export workflows, and troubleshooting procedures for the QCar1 pedestrian detection project.

---

## 1. Environment & Setup Commands

### Create Isolated Virtual Environment
Creates an isolated Python 3.6 virtual environment with access to Jetson system site-packages without altering the root environment:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

### Full Hardware & Software Diagnostics
Runs comprehensive health checks on Python, NumPy, OpenCV (with CUDA status), ONNX Runtime execution providers, TensorRT, and Quanser HAL/PAL:
```bash
python3 test_environment.py
```

### Inspect Jetson Hardware Performance (Jetson TX2)
Monitor real-time CPU/GPU load, thermal throttling, and memory usage:
```bash
# Built-in Tegra stats (CPU/GPU utilization, temperatures, RAM)
sudo tegrastats

# If jtop (jetson-stats) is installed:
jtop
```

---

## 2. Model Management & Conversion

### Automatic Model Download
Downloads `yolov5n.onnx` with size validation and interactive overwrite protection:
```bash
chmod +x download_model.sh
./download_model.sh
```

### Validate Model Graph & Inference (Zero Camera Requirement)
Loads the model into ONNX Runtime, prints input/output tensor shapes and data types, runs a randomized dummy tensor inference, and reports latency in ms:
```bash
# Using default models/yolov5n.onnx:
python3 test_onnx.py

# Using a custom model path:
python3 test_onnx.py --model models/custom_detector.onnx
```

### Re-Exporting YOLOv5 at Opset 12 (Fixing Opset 17 Incompatibility)
Run these commands on a **workstation with internet access** to produce an ONNX file compatible with ONNX Runtime 1.10.0:
```bash
# 1. Clone YOLOv5 v6.1
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export weights to ONNX with opset 12
python export.py --weights yolov5n.pt --include onnx --opset 12 --imgsz 640

# 4. Copy to QCar1 via SCP
scp yolov5n.onnx <user>@<qcar1-ip>:/home/<user>/pk_app/v2_qcar1_pedestrian_detection/models/yolov5n.onnx
```

### Optional Phase 2: Compile TensorRT Engine (TensorRT 8.0.1.6)
To build a high-speed FP16 engine on the Jetson TX2 using the built-in `trtexec` utility:
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov5n.onnx \
    --saveEngine=models/yolov5n_trt8.engine \
    --workspace=1024 \
    --fp16
```
> [!WARNING]
> Do not use flags from newer TensorRT versions (e.g. `--memPoolSize`). Jetson TX2 running TensorRT 8.0.1.6 requires `--workspace=<MB>`.

---

## 3. Testing & Execution Commands

### Test on a Static Image (Benchmarking & Tuning)
Run pedestrian detection on an image without requiring camera or robot hardware:
```bash
# Minimal execution
python3 detect_image.py --image test_images/pedestrian.jpg

# Full parameter control with display window
python3 detect_image.py \
    --image test_images/pedestrian.jpg \
    --model models/yolov5n.onnx \
    --output output/result.jpg \
    --confidence 0.40 \
    --iou 0.45 \
    --show
```

### Run Live QCar1 Detection Loop (Default Hardware)
Executes the main vision pipeline reading RealSense RGB-D and calculating metric pedestrian distances:
```bash
# Standard run with graphical display window
python3 qcar1_pedestrian.py

# Headless mode (SSH/remote terminal without X11 server)
python3 qcar1_pedestrian.py --no-display

# Custom configuration file
python3 qcar1_pedestrian.py --config config.yaml
```

### Run Autonomous Vehicle with Auto-Stop (Integrated Control)
Executes the multi-threaded autonomous pipeline (100Hz motor control + 60Hz lane tracking + 3-10Hz pedestrian detection):
```bash
# Run auto-stop integration
python3 qcar1_auto_stop.py

# Run full autonomous 3-thread controller
python3 qcar1_autonomous.py
```

---

## 4. Hardware & Camera Troubleshooting Commands

### Check RealSense USB Connection
Verify the Intel RealSense camera is enumerated on USB 3.0:
```bash
lsusb -t
# Look for Intel Corp RealSense camera running at 5000M (USB 3.0)
```

### Check Available Video Devices
List all physical video streams registered by the Linux kernel:
```bash
v4l2-ctl --list-devices
```

### Check System Kernel Ring Buffer (dmesg)
Identify any USB disconnects or camera hardware faults:
```bash
dmesg | grep -i -E "realsense|uvcvideo|usb" | tail -n 30
```
