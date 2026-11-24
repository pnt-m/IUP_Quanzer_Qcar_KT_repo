#!/usr/bin/env python3
# ============================================================
# QCar1 TensorRT Benchmark for best_fp16.engine
# ============================================================
from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("[FAIL] TensorRT or PyCUDA not importable.")
    sys.exit(1)


def benchmark(engine_path="models/best_fp16.engine", iterations=50):
    print("============================================================")
    print("QCar1 TensorRT Benchmark (best_fp16.engine)")
    print("============================================================")
    if not os.path.isfile(engine_path):
        print("[FAIL] Engine not found: {}".format(engine_path))
        print("       Run ./build_engine.sh first.")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    bindings = []
    inputs = []
    outputs = []
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = trt.volume(shape)
        h_mem = cuda.pagelocked_empty(size, dtype)
        d_mem = cuda.mem_alloc(h_mem.nbytes)
        bindings.append(int(d_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': h_mem, 'dev': d_mem, 'shape': shape})
        else:
            outputs.append({'host': h_mem, 'dev': d_mem, 'shape': shape})

    inp_shape = inputs[0]['shape']
    dummy = np.random.uniform(0.0, 1.0, inp_shape).astype(np.float32)
    np.copyto(inputs[0]['host'], dummy.ravel())

    print("[INFO] Input Shape: {}".format(inp_shape))
    print("[INFO] Output Shape: {}".format(outputs[0]['shape']))
    print("Warming up GPU (10 runs)...")
    for _ in range(10):
        cuda.memcpy_htod_async(inputs[0]['dev'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['dev'], stream)
        stream.synchronize()

    print("Running {} timed benchmark iterations...".format(iterations))
    times = []
    for _ in range(iterations):
        t0 = time.time()
        cuda.memcpy_htod_async(inputs[0]['dev'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['dev'], stream)
        stream.synchronize()
        times.append((time.time() - t0) * 1000.0)

    avg_ms = float(np.mean(times))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print("------------------------------------------------------------")
    print("GPU Mean Latency : {:.2f} ms".format(avg_ms))
    print("Min / Max Latency: {:.2f} ms / {:.2f} ms".format(np.min(times), np.max(times)))
    print("Throughput (FPS) : {:.1f} FPS".format(fps))
    print("============================================================")
    print("[PASS] TensorRT Engine operates successfully on Jetson TX2 GPU!")


if __name__ == "__main__":
    benchmark()
