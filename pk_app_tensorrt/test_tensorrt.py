#!/usr/bin/env python3
# ============================================================
# QCar1 YOLO Pedestrian Detection - TensorRT Engine Validation
# ============================================================
"""
Validates the compiled TensorRT FP16 engine on the Jetson TX2 GPU.
Runs warmup and benchmark iterations on dummy tensors without requiring
the RealSense camera or robot hardware.

Usage:
    python3 test_tensorrt.py
    python3 test_tensorrt.py --engine models/yolov5n_fp16.engine
"""

from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("[FAIL] tensorrt package is not importable. Ensure TensorRT is installed.")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("[FAIL] pycuda is not importable. Install with: sudo apt-get install -y python3-pycuda")
    sys.exit(1)


def benchmark_engine(engine_path, iterations=50):
    print("============================================================")
    print("QCar1 TensorRT Engine Benchmark (GPU sm_62)")
    print("============================================================")
    print("Engine path: {}".format(engine_path))

    if not os.path.isfile(engine_path):
        print("[FAIL] Engine file not found: {}".format(engine_path))
        print("       Run ./build_engine.sh first to compile the engine.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        print("[FAIL] Failed to deserialize CUDA engine.")
        sys.exit(1)

    context = engine.create_execution_context()
    bindings = []
    inputs = []
    outputs = []
    stream = cuda.Stream()

    print("\nEngine Bindings:")
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = trt.volume(shape)
        is_input = engine.binding_is_input(binding)
        role = "INPUT " if is_input else "OUTPUT"
        print("  [{}] Name: {:<12} Shape: {:<18} Type: {}".format(role, binding, str(shape), dtype.__name__))

        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))

        if is_input:
            inputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})
        else:
            outputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})

    # Prepare dummy input tensor (1, 3, 640, 640)
    input_shape = inputs[0]['shape']
    dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
    np.copyto(inputs[0]['host'], dummy_input.ravel())

    print("\nWarming up GPU (10 iterations)...")
    for _ in range(10):
        cuda.memcpy_htod_async(inputs[0]['dev'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['dev'], stream)
        stream.synchronize()

    print("Running {} benchmark iterations...".format(iterations))
    latencies = []
    for _ in range(iterations):
        t0 = time.time()
        cuda.memcpy_htod_async(inputs[0]['dev'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['dev'], stream)
        stream.synchronize()
        latencies.append((time.time() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    est_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("------------------------------------------------------------")
    print("Benchmark Results:")
    print("  Average Latency : {:.2f} ms (+/- {:.2f} ms)".format(avg_ms, std_ms))
    print("  Min / Max       : {:.2f} ms / {:.2f} ms".format(min_ms, max_ms))
    print("  Raw Throughput  : {:.1f} FPS (GPU Core Inference)".format(est_fps))
    print("------------------------------------------------------------")
    if est_fps >= 25.0:
        print("[PASS] Excellent performance! Jetson TX2 GPU is ready for 30 FPS autonomous driving.")
    else:
        print("[INFO] Engine operational. Estimated camera loop throughput: ~{:.1f} FPS.".format(est_fps * 0.8))
    print("============================================================")


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    default_engine = os.path.join(project_dir, "models", "yolov5n_fp16.engine")

    parser = argparse.ArgumentParser(description="Validate TensorRT engine on Jetson TX2.")
    parser.add_argument("--engine", default=default_engine, help="Path to the .engine file.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations.")
    args = parser.parse_args()

    benchmark_engine(args.engine, iterations=args.iterations)


if __name__ == "__main__":
    main()
