import tensorrt as trt

def apply_patch(error_type: str, builder, network, config):
    """
    Mutates config in place based on diagnosed error_type.
    Returns a short description of what was changed, or None if no known patch.
    """
    if error_type == "workspace_too_small":
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        return "Increased workspace limit to 1GB"

    elif error_type == "missing_optimization_profile":
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            shape = inp.shape
            if -1 in shape:
                min_shape, opt_shape, max_shape = [], [], []
                dynamic_axis_seen = 0
                for d in shape:
                    if d == -1:
                        if dynamic_axis_seen == 0:
                            min_shape.append(1)
                            opt_shape.append(1)
                            max_shape.append(1)
                        else:
                            min_shape.append(1)
                            opt_shape.append(128)
                            max_shape.append(512)
                        dynamic_axis_seen += 1
                    else:
                        min_shape.append(d)
                        opt_shape.append(d)
                        max_shape.append(d)
                profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        config.add_optimization_profile(profile)
        return "Added optimization profile: first dynamic axis treated as batch (fixed at 1), remaining dynamic axes treated as sequence-like (1-512)"

    elif error_type == "naive_fp16_precision":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        integer_types = (trt.DataType.INT32, trt.DataType.INT64, trt.DataType.BOOL)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # Skip layers with any shape-tensor output, or any integer-typed output
            # (e.g. index/id tensors, Int64 constant weights) — these cannot run in
            # FP16 regardless of precision constraints, and forcing them is a genuine
            # TensorRT type error, not a precision tradeoff.
            skip = False
            for out_idx in range(layer.num_outputs):
                out = layer.get_output(out_idx)
                if out.is_shape_tensor or out.dtype in integer_types:
                    skip = True
                    break
            if not skip:
                layer.precision = trt.DataType.HALF
        return "Enabled FP16 precision with forced per-layer precision, excluding shape/integer-typed layers (naive fix — did not validate precision sensitivity)"

    elif error_type == "int8_calibration_missing":
        # INT8 was requested but no calibrator/scales were provided, which TensorRT
        # cannot build from. Rather than fabricate calibration data or guess at a
        # reduced-precision mode (testing showed even plain FP16 degrades this
        # model below tolerance), fall back to full FP32 — the one precision mode
        # already verified to pass validation.
        config.clear_flag(trt.BuilderFlag.INT8)
        config.clear_flag(trt.BuilderFlag.FP16)
        return "Detected missing INT8 calibration data — disabled INT8, reverted to FP32 (verified safe default)"

    return None
