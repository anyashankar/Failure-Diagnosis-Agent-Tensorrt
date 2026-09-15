def generate_report(model_name, onnx_path, attempts_log, final_result):
    """
    attempts_log: list of dicts, one per attempt:
        {"attempt": int, "diagnosis": str or None, "patch": str/dict/None, "outcome": "failed"/"succeeded"}
    final_result: dict with "benchmark" and "accuracy" keys, or None if never succeeded
    """
    lines = []
    lines.append(f"=== TensorRT Build Failure Report: {model_name} ===")
    lines.append(f"Model: {onnx_path}\n")

    lines.append("--- Diagnosis & Patch History ---")
    for a in attempts_log:
        lines.append(f"Attempt {a['attempt']}: {a['outcome'].upper()}")
        if a['diagnosis']:
            lines.append(f"  Diagnosed error: {a['diagnosis']}")

        patch = a.get('patch')
        if isinstance(patch, dict):
            patch_text = patch.get('value')
        else:
            patch_text = patch

        if patch_text:
            lines.append(f"  Patch applied: {patch_text}")
    lines.append("")

    if final_result:
        lines.append("--- Final Validation ---")
        bench = final_result.get("benchmark", {})
        acc = final_result.get("accuracy", {})

        avg = bench.get('avg_latency_ms')
        mn = bench.get('min_latency_ms')
        mx = bench.get('max_latency_ms')
        lines.append(
            f"Latency: avg={avg:.3f}ms, min={mn:.3f}ms, max={mx:.3f}ms"
            if all(isinstance(v, (int, float)) for v in (avg, mn, mx))
            else f"Latency: {bench}"
        )

        mean_cos = acc.get('mean_cosine_similarity')
        worst_cos = acc.get('worst_cosine_similarity')
        diff = acc.get('mean_max_abs_diff')
        passed = acc.get('passed')
        lines.append(
            f"Accuracy: mean_cosine_similarity={mean_cos:.6f}, worst_cosine_similarity={worst_cos:.6f}, "
            f"mean_max_abs_diff={diff:.6f}, passed={passed}"
            if isinstance(mean_cos, (int, float)) and isinstance(worst_cos, (int, float))
            else f"Accuracy: {acc}"
        )
    else:
        lines.append("--- Final Validation ---")
        lines.append("BUILD DID NOT SUCCEED — no validation performed")

    return "\n".join(lines)
