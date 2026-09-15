import signal

class PatchTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise PatchTimeoutError("Generated patch code exceeded time limit")

# Deliberately small, explicit set of names the generated code is allowed to use.
# No imports, no file/network access, no builtins beyond what's needed to mutate
# the TensorRT objects it's handed.
ALLOWED_BUILTINS = {
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "print": print,
    "enumerate": enumerate,
    "list": list,
    "tuple": tuple,
    "min": min,
    "max": max,
}

def run_generated_patch(code: str, builder, network, config, trt_module, timeout_seconds=5):
    """
    Executes untrusted, LLM-generated code in a restricted namespace, with a hard
    timeout. The code is expected to mutate `builder`/`network`/`config` in place,
    matching the same signature as our hand-written apply_patch functions.
    Returns (success: bool, error_message: str or None).
    """
    restricted_globals = {
        "__builtins__": ALLOWED_BUILTINS,
        "trt": trt_module,
    }
    restricted_locals = {
        "builder": builder,
        "network": network,
        "config": config,
    }

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        exec(code, restricted_globals, restricted_locals)
        return True, None
    except PatchTimeoutError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
