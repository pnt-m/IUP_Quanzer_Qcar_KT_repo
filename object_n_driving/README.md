# QCar1 Autonomous Driving & 7-Class TensorRT Perception (`object_n_driving`)

End-to-end autonomous driving suite for Quanser QCar1 (NVIDIA Jetson TX2 Pascal GPU), uniting:
1. **Lateral Guidance:** Map-Aware Lane Tracker + Lyapunov-Stable Stanley Controller on Front CSI Camera (`640x480` @ 30 FPS).
2. **Longitudinal Reflex:** 7-Class NVIDIA TensorRT FP16 Object Detection & 3D Metric Depth on Intel RealSense D435 (`640x480` @ 30 FPS).
3. **Reflex Behaviors:**
   * **Stop Signs:** Mandatory **5.0-second hold**, then automatically resumes cruise.
   * **Pedestrians:** Emergency brake if within **1.5 meters** with Hazard LEDs.
   * **Traffic Lights:** Full stop on Red Light, cruise on Green Light.
   * **Caution Zones:** Slow down (0.04 PWM) on Yellow Light, Yield Sign, or Crosswalk.
   * **Curve Derating:** Automatic cornering throttle reduction to prevent understeer into walls.
4. **Zero-Lag Headless Mode:** Real-time telemetry printed directly to SSH terminal without OpenCV X11 overhead.

---

## Hardware Timings & Multi-Rate Architecture

| Subsystem | Camera / Interface | Resolution | Frame Rate | Latency |
|---|---|---|---|---|
| **Lane Guidance** | Sony IMX219 (CSI Front) | 640x480 | 30 FPS | ~8 ms (CPU Canny/Hough) |
| **Object Detection** | Intel RealSense D435 (RGB) | 640x480 | 30 FPS | ~20 ms (Pascal GPU FP16) |
| **Metric Depth** | Intel RealSense D435 (Depth PX)| 640x480 | 30 FPS | ~4 ms (3D ROI sampling) |
| **Motor & Steering**| Quanser QCar DAQ (HIL) | N/A | 500 Hz | ~0.5 ms (Synchronous) |

* **Total Concurrent Memory (RAM):** $\approx 1.8\text{–}2.1\,\text{GB}$ / 7.85 GB (Over **5.7 GB free RAM**, 0 MB Swap, **0.00% OOM risk**).

---

## Deployment & Execution Steps

### 1. Copy to the Car (From Host PC PowerShell)
```powershell
scp -r C:\Users\Admin\Desktop\object_n_driving nvidia@192.168.2.11:~/
```

### 2. Build the TensorRT Engine (On Car via SSH)
```bash
cd ~/object_n_driving
chmod +x build_engine.sh run_car.sh setup_venv.sh test_all_components.sh
./setup_venv.sh
./build_engine.sh
```

### 3. Run the Subsystem Health & Memory Test
Before starting road navigation, run the automated integration test:
```bash
./test_all_components.sh
```
*Tests CSI camera, RealSense camera, TensorRT engine, Motor DAQ, and validates that concurrent memory stays safely below 2.5 GB.*

### 4. Launch Autonomous Driving in Headless Mode
```bash
./run_car.sh --no-display
```
*(Press `Ctrl+C` at any time to safely halt the vehicle and release hardware).*
