# IUP Quanser QCar Knowledge Transfer (KT) Repository
**Repository Name:** `IUP_Quanzer_Qcar_KT_repo`  
**Platform:** Quanser QCar1 (NVIDIA Jetson TX2 + ARM Cortex-A57 + Pascal GPU + Intel RealSense D435 + 4× CSI Cameras + 2D LiDAR)  
**Software Stack:** MATLAB/Simulink R2022a + QUARC 2022 / Python 3.6 & 3.11 / NVIDIA TensorRT / ONNX Runtime / OpenCV  

---

## 1. Executive Summary & Repository Overview

This repository serves as the definitive engineering Knowledge Transfer (KT) repository for autonomous vehicle perception, control, and deep learning implementations on the **Quanser QCar1** platform. It encapsulates four core autonomous driving architectures ranging from classic Model-Based Design (MBD) in Simulink to GPU-accelerated deep learning using NVIDIA TensorRT:

| Module | Core Purpose | Compute Engine | Perception / Input | Primary Output |
|---|---|---|---|---|
| **[`lane_following/`](#2-lane-following-module-lane_following)** | Model-Based Lane Keeping & Lateral Control | Simulink + QUARC RT (Jetson ARM64) | CSI Camera (HSV Color Thresholding) | PWM Throttle & Steering Servo (`HIL Write`) |
| **[`object_n_driving/`](#3-integrated-object-detection--driving-object_n_driving)** | Integrated Lane Detection + Obstacle Avoidance | Python + TensorRT + OpenCV | RealSense D435 RGB-D + CSI Camera | Closed-Loop Autonomous Driving & Braking |
| **[`pk_app/`](#4-pedestrian-detection--braking-baseline-pk_app)** | Baseline Pedestrian Detection & Safe Braking | Python + ONNX Runtime (CPU) | RealSense D435 RGB-D | Emergency Braking & Obstacle Tracking |
| **[`pk_app_tensorrt/`](#5-pedestrian-detection-gpu-accelerated-pk_app_tensorrt)** | High-Throughput TensorRT FP16 Pedestrian Detection | Python + NVIDIA TensorRT (GPU) | RealSense D435 RGB-D | Real-Time GPU Inference (30+ FPS) & Braking |

---

## 2. Repository Directory Structure

```
IUP_Quanzer_Qcar_KT_repo/
├── .gitignore                                      # Excludes virtual environments, compilation artifacts, and cache
├── README.md                                       # Master Knowledge Transfer (KT) documentation
│
├── lane_following/                                 # Quanser QCar Simulink & QUARC Lane Following
│   ├── doc_images/                                 # Visual figures & reference block diagrams
│   ├── Autonomous_Driving_Car_Example1.slx         # Canonical closed-loop Simulink lane keeping model
│   ├── Autonomous_Driving_Car_Example1.slx.r2019a  # MATLAB R2019a compatibility model
│   ├── Autonomous_Driving_Car_Example1.rt-linux_nvidia # Precompiled deployable ARM64 real-time binary
│   ├── HSV_Selection.slx                           # Calibration model for HSV color segmentation
│   ├── HSV_Selection.slx.r2019a                    # Calibration model (R2019a export)
│   ├── HSV_Selection.rt-linux_nvidia               # Precompiled calibration real-time binary
│   ├── QCar1_autonomous_driving_example2.slx       # Advanced lateral control with dynamic gain scheduling
│   ├── QCar1_autonomous_driving_example2.rt-linux_nvidia # Precompiled advanced real-time binary
│   ├── task_lane_following.py                      # Python-based lane following task script
│   ├── Lane Following.docx                         # Comprehensive technical implementation guide
│   └── Lane Following.pdf                          # Formatted technical documentation
│
├── object_n_driving/                               # Integrated Autonomy Pipeline (Lane + YOLO + Depth)
│   ├── models/
│   │   └── best.onnx                               # Custom-trained YOLO detection model (ONNX format)
│   ├── output/                                     # Output directory for telemetry & annotated frames
│   ├── utils/
│   │   ├── depth_utils.py                          # RealSense depth frame extraction and 3D alignment
│   │   ├── lane_detector.py                        # Polynomial curve fitting & lane center offset computation
│   │   └── yolo_trt.py                             # TensorRT engine executor, CUDA memory streaming & NMS
│   ├── build_engine.sh                             # Automated script to build TensorRT engine from ONNX
│   ├── config.yaml                                 # Hyperparameters, control gains, and camera settings
│   ├── qcar1_autonomous_driving.py                 # Main vehicle control loop (Sense -> Perceive -> Decide -> Act)
│   ├── README.md                                   # Module documentation and usage guide
│   ├── requirements.txt                            # Python dependencies
│   ├── run_car.sh                                  # 1-Click execution shell script
│   ├── setup_venv.sh                               # Jetson virtual environment setup script
│   ├── test_all_components.py                      # Subsystem diagnostic utility
│   ├── test_all_components.sh                      # Component test automation wrapper
│   └── test_tensorrt.py                            # Standalone TensorRT engine benchmark
│
├── pk_app/                                         # Baseline Pedestrian Detection & Systems Engineering
│   ├── command_artifacts/                          # Systems engineering & architectural deep-dive docs
│   │   ├── ARCHITECTURAL_ANALYSIS.md               # Planning philosophy, constraints, and layer verification
│   │   ├── COMMAND_REFERENCE.md                    # Jetson TX2 CLI cheatsheet & operational commands
│   │   ├── EXTRACTION_PLAYBOOK.md                  # Model export & calibration procedures
│   │   ├── QCAR_SYSTEMS_ENGINEERING.md             # Subsystem interface specifications & safety layers
│   │   ├── reexport_yolov5_opset12.sh              # PyTorch to ONNX Opset 12 export script
│   │   └── verify_stack.py                         # Deep stack integrity verification tool
│   └── v2_qcar1_pedestrian_detection/              # Production CPU ONNX detection application
│       ├── hal/                                    # Quanser Hardware Abstraction Layer
│       ├── models/
│       │   └── yolov5n.onnx                        # Lightweight YOLOv5 Nano ONNX weights
│       ├── output/                                 # Test artifacts and evaluation captures
│       ├── pal/                                    # Quanser Product Abstraction Layer
│       ├── pit/                                    # Quanser PIT perception & neural libraries
│       ├── qvl/                                    # Quanser Virtual Labs interface
│       ├── test_images/                            # Validation test dataset (pedestrians, road scenes)
│       ├── utils/
│       │   ├── depth_utils.py                      # RealSense D435 distance calculation
│       │   └── yolo_onnx.py                        # ONNX Runtime inference & bounding box processing
│       ├── config.yaml                             # Detection thresholds and distance limits
│       ├── detect_image.py                         # Static image verification tool
│       ├── download_model.sh                       # Model download and integrity verification
│       ├── qcar1_pedestrian.py                     # Main CPU-based pedestrian detection & braking loop
│       ├── README.md                               # Step-by-step lab walkthrough
│       ├── requirements.txt                        # Dependency manifest
│       ├── setup_venv.sh                           # Non-invasive venv builder
│       ├── test_environment.py                     # Hardware check (CUDA, Camera, Quanser libs)
│       └── test_onnx.py                            # ONNX Runtime benchmark with dummy tensors
│
└── pk_app_tensorrt/                                # GPU-Accelerated TensorRT Pedestrian Detection
    ├── hal/                                        # Quanser Hardware Abstraction Layer
    ├── models/
    │   └── yolov5n.onnx                            # YOLOv5 Nano baseline model
    ├── output/                                     # Output telemetry and processed images
    ├── pal/                                        # Quanser Product Abstraction Layer
    ├── pit/                                        # Quanser PIT perception libraries
    ├── qvl/                                        # Quanser Virtual Labs interface
    ├── test_images/                                # Static test images
    ├── utils/
    │   ├── depth_utils.py                          # Depth sensor processing
    │   ├── yolo_onnx.py                            # Fallback ONNX runner
    │   └── yolo_trt.py                             # TensorRT CUDA runtime handler
    ├── build_engine.sh                             # TensorRT FP16 engine builder via `trtexec`
    ├── config.yaml                                 # Configuration profile
    ├── detect_image.py                             # Image test using TensorRT engine
    ├── qcar1_pedestrian.py                         # High-performance control loop (28-35 FPS)
    ├── README.md                                   # TensorRT optimization technical briefing
    ├── requirements.txt                            # Python dependencies
    ├── run_car.sh                                  # Autonomous vehicle launcher with safety clamping
    ├── setup_venv.sh                               # Environment setup script
    ├── test_environment.py                         # Jetson GPU and CUDA environment audit
    └── test_tensorrt.py                            # CUDA timing benchmark
```

---

## 3. Hardware & Platform Architecture

### 3.1 Quanser QCar1 Physical Specifications
* **Processor Target:** NVIDIA Jetson TX2 (Dual-core Denver 2 64-bit CPU + Quad-core ARM A57 complex).
* **GPU Accelerator:** 256-core NVIDIA Pascal architecture GPU.
* **Onboard DAQ / I/O:** Quanser QCar Data Acquisition card (connected via PCIe/SPI to Jetson).
* **Vision Sensors:**
  * **4× CSI Cameras:** Sony IMX219 (Front, Right, Rear, Left) operating via GStreamer / V4L2.
  * **1× RGB-D Camera:** Intel RealSense D435 (Stereo IR depth + RGB sensor via USB 3.0).
* **Ranging Sensor:** RPLIDAR A2 360° 2D Laser Scanner (UART / USB).
* **Actuation Channels:**
  * **Drive Motor:** Electronic Speed Controller (ESC) driving Brushless DC Motor on PWM Channel 0 (`motor_limit=0.3` for laboratory safety).
  * **Steering Servo:** PWM Channel 1000 ($\pm 0.5\,\text{rad}$ steering range, with neutral trim offset $\approx +0.05\,\text{rad}$).
  * **Vehicle Status:** Multi-color LED indicators on digital output channels.

### 3.2 Network Topology & Target Configuration
* **Local Subnet:** `192.168.2.x` (Host PC and QCar must be connected to the `Quanser_UVs` network).
* **QCar1 Default IP:** `192.168.2.11` (or `.12`, `.13` depending on the physical car identifier).
* **QUARC Target Manager URI:** `tcpip://192.168.2.11:17001`
* **Real-Time Kernel:** Linux for Tegra (L4T R32.6.x / Ubuntu 18.04 LTS).

---

## 4. Module Deep Dives & Operational Guide

### 4.1 Lane Following Module (`lane_following/`)

The lane following architecture implements a classical computer vision and control feedback loop. It processes forward-facing camera imagery, identifies the lane center line, computes lateral position error, and feeds a proportional-integral-derivative (PID) controller to steer the vehicle.

#### Workflow:
1. **Calibrate HSV Masking:**
   * Open `HSV_Selection.slx` in MATLAB/Simulink R2022a.
   * Connect to QCar via External Mode (`Ctrl+B` or Monitor & Tune).
   * Adjust Hue, Saturation, and Value thresholds until the track line (yellow or white) is cleanly segmented from the floor surface.
2. **Execute Lane Keeping:**
   * Open `Autonomous_Driving_Car_Example1.slx`.
   * The model uses the Quanser `Video Capture` block (Device 0 at $640 \times 480$), crops the Region of Interest (ROI) to the lower track half, applies binary thresholding, and computes the blob centroid.
   * Centroid deviation from the image center ($320\text{ px}$) creates the error signal $e_{\text{lat}}$.
   * The PID controller transforms $e_{\text{lat}}$ into steering angle commands sent to `HIL Write` (Analog Channel 1000).
   * Constant or speed-scheduled throttle is dispatched to `HIL Write` (PWM Channel 0).

---

### 4.2 Integrated Object Detection & Driving (`object_n_driving/`)

This module unifies **lateral lane keeping** with **forward obstacle collision avoidance** using an integrated Python pipeline running directly on the Jetson TX2.

```
                    ┌───────────────────────────────┐
                    │  Intel RealSense D435 (RGB-D) │
                    └───────┬───────────────┬───────┘
                            │               │
                 Color Stream (RGB)    Depth Stream (Aligned Z)
                            │               │
                            ▼               ▼
                  ┌───────────────────┐ ┌───────────────────────┐
                  │ YOLO TensorRT FP16│ │ Depth Distance Matrix │
                  └─────────┬─────────┘ └───────────┬───────────┘
                            │                       │
                            ▼                       │
                   Bounding Box & Class             │
                            │                       │
                            ▼                       ▼
                  ┌─────────────────────────────────────────────┐
                  │ Decision Engine: Object in Path & Dist < Z_0│
                  └──────────────────────┬──────────────────────┘
                                         │
                    No Obstacle          │ Obstacle Detected (< 0.8m)
                         ▼               ▼
               ┌──────────────────┐ ┌──────────────────────┐
               │ Lane Follower    │ │ Active Brake Clamping│
               │ (Steering Control│ │ (Throttle -> 0.0)    │
               └─────────┬────────┘ └──────────┬───────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Quanser QCar HAL  │
                         │ (PWM 0 & Servo 1000)│
                         └─────────────────────┘
```

#### Running the Module:
```bash
cd object_n_driving
# 1. Build the TensorRT engine from ONNX model (one-time step on Jetson)
./build_engine.sh

# 2. Run standalone diagnostic to verify cameras, CUDA, and motor channels
python3 test_all_components.py

# 3. Launch autonomous driving
./run_car.sh
```

---

### 4.3 Pedestrian Detection: CPU Baseline vs. GPU TensorRT

The repository preserves both the **reference CPU implementation (`pk_app/`)** and the **optimized GPU implementation (`pk_app_tensorrt/`)** to illustrate the performance leap achieved through hardware acceleration:

| Feature | `pk_app` (Baseline) | `pk_app_tensorrt` (Production) |
|---|---|---|
| **Inference Engine** | ONNX Runtime (CPU) | NVIDIA TensorRT (GPU FP16) |
| **Compute Device** | 4-Core ARM Cortex-A57 | 256-Core NVIDIA Pascal GPU |
| **Inference Latency** | $\approx 160\,\text{ms}$ | **$\approx 25\,\text{ms}$ ($6.4\times$ speedup)** |
| **Processing Throughput**| $5.5\text{–}6.5\,\text{FPS}$ | **$28.0\text{–}35.0\,\text{FPS}$** |
| **RAM Utilization** | $\approx 2.4\,\text{GB}$ | **$< 380\,\text{MB}$ (Prevents OOM)** |
| **Control Loop Safety** | High latency introduces braking delay | Immediate reactive braking within $30\,\text{ms}$ |

#### Building and Running TensorRT Engine (`pk_app_tensorrt/`):
```bash
cd pk_app_tensorrt

# 1. Inspect environment (CUDA 10.2, TensorRT 8.x, Jetson L4T)
python3 test_environment.py

# 2. Compile ONNX to TensorRT FP16 engine via trtexec
./build_engine.sh

# 3. Benchmark latency with dummy tensors
python3 test_tensorrt.py

# 4. Execute vehicle with pedestrian collision mitigation
./run_car.sh
```

---

## 5. Safety Systems & Clamping Guidelines

To avoid physical hardware damage, all modules adhere to these safety constraints:

1. **Velocity / Throttle Clamp:**
   * Maximum forward throttle: $+0.3$ ($30\%$ duty cycle).
   * Do not exceed `throttle = 0.3` indoors or on experimental floor mats.
2. **Steering Angle Bounds:**
   * Physical limit: $\pm 0.5\,\text{rad}$ ($\approx \pm 28.6^\circ$).
   * Steering trim offset: `steer_bias = +0.05` to counteract mechanical linkage bias.
3. **Emergency Stop (E-Stop):**
   * If communication drops or an unhandled exception occurs, the hardware watchdog automatically zeroes all PWM lines.
   * Emergency Python script: `hardware_stop.py` can be triggered to force all actuators to zero instantly:
     ```bash
     python3 -c "from pal.products.qcar import QCar; car = QCar(); car.read_write_std(0.0, 0.0); car.terminate()"
     ```

---

## 6. Knowledge Transfer (KT) Summary for New Researchers

1. **Simulink vs. Python:**
   * Use **Simulink + QUARC (`lane_following/`)** when designing hard real-time controllers requiring deterministic microsecond execution ($500\,\text{Hz}$ loops).
   * Use **Python + TensorRT (`object_n_driving/`, `pk_app_tensorrt/`)** when executing deep learning vision pipelines where GPU streaming and tensor operations dominate.
2. **TensorRT Engine Portability Warning:**
   * TensorRT `.engine` files are binary execution graphs compiled specifically for the GPU microarchitecture and TensorRT version on the device.
   * **Never copy `.engine` files across machines.** Always keep the `.onnx` source weights and generate the `.engine` locally using `./build_engine.sh`.
3. **RealSense Calibration:**
   * The RealSense depth sensor relies on infrared stereoscopy. Ensure protective lens covers are removed and lighting does not introduce specular infrared glare.

---
*Maintained for IUP Autonomous Systems & Quanser QCar Research.*
