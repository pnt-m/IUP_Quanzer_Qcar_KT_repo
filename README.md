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

## 3. Hardware, Fleet IP Table & Network Architecture Guide

### 3.1 Quanser QCar1 Physical Specifications
* **Processor Target:** NVIDIA Jetson TX2 (Dual-core Denver 2 64-bit CPU + Quad-core ARM Cortex-A57 complex).
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

---

### 3.2 QCar Fleet Master IP Registry & Hostname Mapping

Every vehicle in the physical fleet is assigned a static IP address on the dedicated laboratory wireless network:

| Vehicle | Static IP Address | Default Hostname | User Account | Default Password | QUARC Target URI (Port 17001) | SSH Connection String |
|---|---|---|---|---|---|---|
| **QCar 1** | `192.168.2.11` | `qcar1` | `nvidia` | `nvidia` | `tcpip://192.168.2.11:17001` | `ssh nvidia@192.168.2.11` |
| **QCar 2** | `192.168.2.12` | `qcar2` | `nvidia` | `nvidia` | `tcpip://192.168.2.12:17001` | `ssh nvidia@192.168.2.12` |
| **QCar 3** | `192.168.2.13` | `qcar3` | `nvidia` | `nvidia` | `tcpip://192.168.2.13:17001` | `ssh nvidia@192.168.2.13` |

> [!IMPORTANT]
> **Fleet Scaling Rule:** When migrating Simulink models or Python scripts between cars, update **only the last octet** of the IP string (e.g., change `192.168.2.11` to `192.168.2.12` or `192.168.2.13`). All internal channel numbers, DAQ board IDs (`0`), and sensor paths remain identical across cars.

---

### 3.3 Network Topology & Host PC Configuration

All communication occurs over an isolated, high-speed laboratory Wi-Fi network:

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Host Development PC         │           │        Quanser QCar Fleet       │
│  (Windows 10/11 + MATLAB R2022a)│           │       (NVIDIA Jetson TX2)       │
│                                 │  802.11ac │                                 │
│  IP: 192.168.2.100 (DHCP/Static)│◄─────────►│ QCar 1: 192.168.2.11            │
│  Netmask: 255.255.255.0         │ Wi-Fi AP  │ QCar 2: 192.168.2.12            │
│  SSID: Quanser_UVs              │           │ QCar 3: 192.168.2.13            │
└─────────────────────────────────┘           └─────────────────────────────────┘
```

#### Network Parameters:
* **Wi-Fi SSID:** `Quanser_UVs`
* **Subnet:** `192.168.2.0/24`
* **Subnet Mask:** `255.255.255.0`
* **Default Gateway / AP Router:** `192.168.2.1`

#### Host PC Network Setup:
1. Connect your workstation's Wi-Fi adapter to the **`Quanser_UVs`** SSID.
2. In most laboratory setups, the router assigns an IP via DHCP in the range `192.168.2.100` – `192.168.2.200`.
3. If using a manual static IP on the Host PC:
   * **IPv4 Address:** `192.168.2.100`
   * **Subnet Mask:** `255.255.255.0`
   * **Default Gateway:** `192.168.2.1`
   * **DNS:** `8.8.8.8` (or leave blank if offline)

---

### 3.4 Remote Access Protocols: SSH, SCP & SFTP

#### A. Interactive SSH Terminal
Connect to the vehicle's onboard Linux terminal:
```bash
# Terminal (PowerShell, CMD, Git Bash, or Linux terminal)
ssh nvidia@192.168.2.11

# If prompted: "Are you sure you want to continue connecting (yes/no/[fingerprint])?" type: yes
# Password: nvidia
```

#### B. Secure File Transfer (SCP)
Deploy code or download logs between the Host PC and the vehicle:

* **Push a project to QCar1:**
  ```powershell
  # From Host PC PowerShell:
  scp -r C:\Users\Admin\Desktop\git\object_n_driving nvidia@192.168.2.11:/home/nvidia/
  ```

* **Push pk_app_tensorrt to QCar1:**
  ```powershell
  scp -r C:\Users\Admin\Desktop\git\pk_app_tensorrt nvidia@192.168.2.11:/home/nvidia/
  ```

* **Pull runtime logs / photos from QCar1 to Host PC:**
  ```powershell
  scp nvidia@192.168.2.11:/home/nvidia/object_n_driving/output/*.jpg C:\Users\Admin\Desktop\
  ```

#### C. GUI File Management (WinSCP)
* **File Protocol:** `SFTP`
* **Host Name:** `192.168.2.11`
* **Port Number:** `22`
* **User Name:** `nvidia`
* **Password:** `nvidia`

---

### 3.5 QUARC Target Manager & Simulink External Mode Deployment

When deploying Simulink models (`lane_following/`) to the physical vehicle, QUARC establishes a low-latency real-time telemetry link over TCP port **`17001`**.

#### Configuration in Simulink (`Ctrl + E`):
1. In the Simulink model window, press **`Ctrl + E`** to open **Model Configuration Parameters**.
2. In the left navigation pane, select **Code Generation $\rightarrow$ Interface**.
3. Set **Interface** to: `External mode`.
4. Set **Transport layer** to: `tcpip`.
5. In **MEX-file arguments**, paste the exact targeting string:
   ```text
   '-w -d /tmp -uri %u','tcpip://192.168.2.11:17001'
   ```
   *(To target QCar 2 or 3, change `192.168.2.11` to `192.168.2.12` or `192.168.2.13`).*
6. In **Code Generation**, ensure **System target file** is set to **`quarc_linux_nvidia.tlc`**.
7. Click **Apply** $\rightarrow$ **OK**.

#### What Happens During Deployment (`Ctrl + B`):
```
[Host PC: Simulink .slx] 
      │
      ▼  Ctrl + B (Cross-Compiling via GCC aarch64)
[.rt-linux_nvidia ARM64 Binary] 
      │
      ▼  TCP/IP Transfer to tcpip://192.168.2.11:17001
[Jetson TX2: /tmp/ModelName] 
      │
      ▼  QUARC Target Manager Launches Process with Real-Time Priority
[Vehicle Executes Hardware Loop at 500 Hz] 
      │
      ▼  External Mode TCP Telemetry Streams Back
[Host PC Scopes & Sliders Update in Real-Time]
```

---

### 3.6 Port Allocation & Service Matrix

| Port | Protocol | Daemon / Service | Description | Direction |
|---|---|---|---|---|
| **22** | TCP | `sshd` (OpenSSH) | Remote shell terminal and SCP/SFTP file transfers | Host $\rightarrow$ QCar |
| **17001** | TCP | `quarc_target` | QUARC Target Manager (binary deployment & External Mode telemetry) | Host $\rightarrow$ QCar |
| **18001** | TCP | RPLIDAR Daemon | Quanser Stream Server streaming raw LiDAR packets (optional) | Host $\leftarrow$ QCar |
| **18960** | TCP | `QCar_Workspace` | Virtual bridge socket (used ONLY in virtual QLabs simulation) | Localhost loopback |

---

### 3.7 Pre-Flight Network Verification & Diagnostic Cheatsheet

Before running any autonomous drive or building code, run these verification commands on the Host PC:

```powershell
# 1. Verify basic ICMP reachability:
ping -n 4 192.168.2.11

# 2. Verify QUARC Target Manager is listening on port 17001:
Test-NetConnection -ComputerName 192.168.2.11 -Port 17001

# 3. Verify SSH server is listening on port 22:
Test-NetConnection -ComputerName 192.168.2.11 -Port 22
```

If connecting via Linux/macOS:
```bash
ping -c 4 192.168.2.11
nc -zv 192.168.2.11 17001
nc -zv 192.168.2.11 22
```

---

### 3.8 Network & Connectivity Troubleshooting Matrix

| Symptom / Error | Root Cause | Exact Recovery Procedure |
|---|---|---|
| **`Destination host unreachable` / `Request timed out`** | Workstation is connected to the wrong Wi-Fi network or IP subnet mismatch. | 1. Open Windows Wi-Fi menu and ensure connected to **`Quanser_UVs`**.<br>2. Run `ipconfig` in PowerShell; verify IPv4 is `192.168.2.xxx`.<br>3. Power cycle QCar if vehicle Wi-Fi failed to associate. |
| **`Test-NetConnection: Port 17001 Failed`** | QUARC Target Manager daemon on Jetson is stopped or vehicle battery died. | 1. SSH into car: `ssh nvidia@192.168.2.11`.<br>2. Check status: `sudo systemctl status quarc_target`.<br>3. Restart daemon: `sudo systemctl restart quarc_target`.<br>4. Check car battery voltage (must be $\ge 11.1\,\text{V}$). |
| **`Host key verification failed` on SSH** | Another vehicle previously used the same IP, or Jetson OS was reflashed. | Clear the stale host key from your PC:<br>`ssh-keygen -R 192.168.2.11` |
| **External Mode Scope lag / buffer overruns** | Streaming uncompressed, high-bandwidth camera frames over Wi-Fi. | In Simulink scope parameters, increase the **Decimation** factor (e.g., plot 1 out of every 10 samples) or use `Image Compress` block before streaming. |
| **`QERR_CONNECTION_REFUSED` in Simulink** | MEX-file arguments contain the wrong IP or car number. | Press `Ctrl + E` $\rightarrow$ Code Generation $\rightarrow$ Interface $\rightarrow$ MEX-file arguments. Verify IP matches the car you are working with. |

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
