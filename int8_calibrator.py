import os
import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False


class BertInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    Minimal INT8 calibrator for BERT: feeds a small set of representative
    tokenized inputs so TensorRT can compute real per-tensor dynamic ranges,
    instead of the naive 'INT8 with no calibrator' failure we handled earlier.
    """
    def __init__(self, input_names, seq_len=128, num_batches=8, cache_file="bert_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_names = input_names
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.current_batch = 0
        self.cache_file = os.path.expanduser(f"~/{cache_file}")

        # Generate representative calibration data: random but realistic-shaped
        # token ids and attention masks (real deployments would use real held-out data)
        np.random.seed(123)
        self.data = {
            "input_ids": np.random.randint(0, 30000, (num_batches, 1, seq_len)).astype(np.int64),
            "attention_mask": np.ones((num_batches, 1, seq_len)).astype(np.int64),
        }

        self.device_buffers = {}
        if HAS_PYCUDA:
            for name in input_names:
                sample = self.data[name][0]
                self.device_buffers[name] = cuda.mem_alloc(sample.nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None

        pointers = []
        for name in names:
            batch = self.data[name][self.current_batch]
            cuda.memcpy_htod(self.device_buffers[name], batch)
            pointers.append(int(self.device_buffers[name]))

        self.current_batch += 1
        return pointers

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
