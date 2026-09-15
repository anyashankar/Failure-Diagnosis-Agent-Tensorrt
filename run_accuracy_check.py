import numpy as np
import onnxruntime as ort
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

def _run_single_comparison(onnx_path, engine_bytes, seed):
    np.random.seed(seed)

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {}
    input_shapes = {}

    for inp in ort_session.get_inputs():
        shape = tuple(1 if isinstance(d, str) or d is None or d < 0 else d for d in inp.shape)
        dtype = np.float32 if 'float' in inp.type else np.int64
        arr = np.random.random(shape).astype(dtype) if dtype == np.float32 else np.random.randint(0, 100, shape).astype(dtype)
        ort_inputs[inp.name] = arr
        input_shapes[inp.name] = arr

    output_names = [o.name for o in ort_session.get_outputs()]
    ort_outputs = ort_session.run(None, ort_inputs)
    target_idx = 0
    for i, name in enumerate(output_names):
        if name.lower() not in ('tanh', 'pooler_output'):
            target_idx = i
            break
    ort_output = ort_outputs[target_idx].flatten()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    device_buffers = {}
    output_name = None
    output_shape = None
    output_dtype = None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

        if is_input:
            arr = input_shapes[name]
            context.set_input_shape(name, arr.shape)
            device_buffers[name] = cuda.mem_alloc(arr.nbytes)
        else:
            if name.lower() in ('tanh', 'pooler_output'):
                buf_size = int(np.prod(context.get_tensor_shape(name))) * 4
                device_buffers[name] = cuda.mem_alloc(buf_size)
                context.set_tensor_address(name, int(device_buffers[name]))
                continue
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            arr = np.empty(tuple(shape), dtype=dtype)
            device_buffers[name] = cuda.mem_alloc(arr.nbytes)
            output_name = name
            output_shape = shape
            output_dtype = dtype

        context.set_tensor_address(name, int(device_buffers[name]))

    stream = cuda.Stream()
    for name, arr in input_shapes.items():
        cuda.memcpy_htod_async(device_buffers[name], arr, stream)

    context.execute_async_v3(stream.handle)

    trt_output_host = np.empty(tuple(output_shape), dtype=output_dtype)
    cuda.memcpy_dtoh_async(trt_output_host, device_buffers[output_name], stream)
    stream.synchronize()

    trt_output = trt_output_host.flatten()

    min_len = min(len(ort_output), len(trt_output))
    a, b = ort_output[:min_len], trt_output[:min_len]

    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    max_abs_diff = np.max(np.abs(a - b))

    return float(cosine_sim), float(max_abs_diff)

def run_accuracy_check(onnx_path, engine_bytes, tolerance=1e-3, num_samples=15):
    """
    Compares TensorRT engine output against ONNX Runtime across multiple
    random input samples, and reports the mean and worst-case similarity.
    This smooths out per-sample noise and TensorRT's own build-to-build
    tactic-selection variance, rather than trusting a single comparison.
    """
    if not HAS_PYCUDA:
        return {"error": "pycuda not installed"}

    similarities = []
    max_diffs = []

    for i in range(num_samples):
        cos_sim, max_diff = _run_single_comparison(onnx_path, engine_bytes, seed=42 + i)
        similarities.append(cos_sim)
        max_diffs.append(max_diff)

    mean_sim = float(np.mean(similarities))
    worst_sim = float(np.min(similarities))
    mean_diff = float(np.mean(max_diffs))

    return {
        "mean_cosine_similarity": mean_sim,
        "worst_cosine_similarity": worst_sim,
        "mean_max_abs_diff": mean_diff,
        "passed": bool(worst_sim > (1 - tolerance)),
    }
