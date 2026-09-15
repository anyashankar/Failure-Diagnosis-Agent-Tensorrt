import tensorrt as trt

class LogCapture(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.messages = []

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            self.messages.append(f"[{severity}] {msg}")

def run_build(onnx_path: str, config_overrides=None):
    """
    Attempts a TensorRT build from an ONNX file.
    config_overrides: optional callable(builder, network, config) -> None,
                       used to apply patches before building.
    Returns: (success: bool, engine_bytes_or_None, log_text: str)
    """
    log_capture = LogCapture()
    builder = trt.Builder(log_capture)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, log_capture)

    with open(onnx_path, 'rb') as f:
        parse_ok = parser.parse(f.read())
        if not parse_ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return False, None, "PARSE ERROR: " + " | ".join(errors)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if config_overrides:
        config_overrides(builder, network, config)

    engine = builder.build_serialized_network(network, config)
    log_text = "\n".join(log_capture.messages)

    return engine is not None, engine, log_text
