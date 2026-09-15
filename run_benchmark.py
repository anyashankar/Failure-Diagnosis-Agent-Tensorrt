import time
import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

def run_benchmark(engine_bytes, num_iterations=50, warmup_iterations=10):
    """
    Loads a serialized TensorRT engine and measures average inference latency.
    Returns: dict with 'avg_latency_ms', 'min_latency_ms', 'max_latency_ms'
    """
    if not HAS_PYCUDA:
        return {"error": "pycuda not installed"}

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # Allocate buffers for each I/O tensor
    input_buffers = {}
    output_buffers = {}
    device_buffers = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

        # Resolve dynamic dims to a concrete shape for benchmarking
        concrete_shape = tuple(1 if d == -1 else d for d in shape)

        if is_input:
            context.set_input_shape(name, concrete_shape)
            host_arr = np.random.random(concrete_shape).astype(dtype)
            input_buffers[name] = host_arr
        else:
            host_arr = np.empty(concrete_shape, dtype=dtype)
            output_buffers[name] = host_arr

        device_buffers[name] = cuda.mem_alloc(host_arr.nbytes)
        context.set_tensor_address(name, int(device_buffers[name]))

    stream = cuda.Stream()

    # Copy inputs to device once
    for name, arr in input_buffers.items():
        cuda.memcpy_htod_async(device_buffers[name], arr, stream)
    stream.synchronize()

    # Warmup
    for _ in range(warmup_iterations):
        context.execute_async_v3(stream.handle)
    stream.synchronize()

    # Timed runs
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
    }
