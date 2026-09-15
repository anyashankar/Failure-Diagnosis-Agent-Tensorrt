import os
import sys
sys.path.insert(0, '/home/anyashan')
from google import genai
from google.genai import types
import tensorrt as trt
from run_build import run_build
from apply_patch import apply_patch
from run_benchmark import run_benchmark
from run_accuracy_check import run_accuracy_check
from generate_report import generate_report
from agent_memory import get_cached_fix, record_fix
from notifier import send_notification
from sandbox_exec import run_generated_patch

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="fix_workspace_too_small",
            description="Increases the TensorRT builder's workspace memory pool limit. Use this when the build error mentions 'insufficient workspace' or a tactic/kernel implementation could not be found due to memory limits.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="fix_missing_optimization_profile",
            description="Adds a TensorRT optimization profile derived from the network's dynamic input shapes. Use this when the build error mentions 'no optimization profile has been defined' or the network has dynamic/shape inputs, OR when a previous attempt succeeded but failed accuracy validation and no optimization profile has been confirmed present yet.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="fix_int8_calibration_missing",
            description="Disables INT8 precision and falls back to FP32 when the build error mentions 'calibration failure', 'no scaling factors detected', or that INT8 was requested without a calibrator.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="generate_custom_patch",
            description=(
                "Use this ONLY when the error does not match any of the other known fix tools. "
                "Write a short Python code snippet that mutates the existing 'builder', 'network', and "
                "'config' objects (TensorRT Python API objects, already in scope) to resolve the error. "
                "The 'trt' module is available. Do not use import statements — trt is already provided. "
                "Keep the code minimal and targeted to the specific error shown."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(
                        type=types.Type.STRING,
                        description="Python code that mutates builder/network/config to fix the error. No imports.",
                    ),
                },
                required=["code"],
            ),
        ),
    ])
]

def gemini_diagnose(error_log: str, network_summary: str, prior_attempt_note: str = ""):
    truncated_log = error_log[-4000:] if len(error_log) > 4000 else error_log

    prompt = f"""A TensorRT engine build failed. Here is the raw error log (tail shown if truncated):

{truncated_log}

Here is a summary of the network's inputs:
{network_summary}

{prior_attempt_note}

Diagnose the failure and call the appropriate tool to fix it. Only use generate_custom_patch
if none of the other tools apply to this specific error."""

    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(tools=tools),
    )

    for part in response.candidates[0].content.parts:
        if part.function_call:
            return part.function_call.name, dict(part.function_call.args) if part.function_call.args else {}

    return None, None

def gemini_best_effort_diagnosis(error_log: str) -> str:
    truncated_log = error_log[-4000:] if len(error_log) > 4000 else error_log
    prompt = f"""A TensorRT build failed with an error this system has no automated fix for.
Give a brief (2-3 sentence) best-effort diagnosis of what likely went wrong, for a human engineer to review.

Error log:
{truncated_log}"""

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"(Could not generate best-effort diagnosis: {e})"

def summarize_network(network) -> str:
    lines = []
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        lines.append(f"  - {inp.name}: shape={inp.shape}")
    return "\n".join(lines)

def apply_one_patch(patch_entry, builder, network, config):
    if isinstance(patch_entry, tuple) and patch_entry[0] == "custom":
        code = patch_entry[1]
        success, err = run_generated_patch(code, builder, network, config, trt)
        if success:
            return f"Executed Gemini-generated patch:\n{code}"
        else:
            return f"GENERATED PATCH FAILED TO EXECUTE: {err}\nCode was:\n{code}"
    else:
        return apply_patch(patch_entry, builder, network, config)

def diagnose_and_fix_with_gemini(onnx_path, model_name="model", max_attempts=5, force_first_patch=None, on_update=None, should_stop=None):
    applied_patches = []
    attempts_log = []
    prior_attempt_note = ""
    pending_memory_record = None

    if force_first_patch:
        applied_patches = ['missing_optimization_profile', force_first_patch]

    def config_overrides(builder, network, config):
        for entry in applied_patches:
            result = apply_one_patch(entry, builder, network, config)
            print(f"Patch: {result}")

    for attempt in range(1, max_attempts + 1):
        if should_stop and should_stop():
            print("\nStop requested — halting before next attempt.")
            attempts_log.append({
                "attempt": attempt,
                "diagnosis": "Run paused by user",
                "patch": None,
                "outcome": "stopped",
            })
            if on_update:
                on_update(attempts_log)
            report = generate_report(model_name, onnx_path, attempts_log, None)
            print("\n" + report)
            send_notification(model_name, False, {"escalation_reason": "Run stopped by user request"})
            return False, None, None

        print(f"\n--- Attempt {attempt} ---")
        if on_update:
            on_update(attempts_log + [{"attempt": attempt, "diagnosis": None, "patch": None, "outcome": "running"}])

        success, engine, log = run_build(onnx_path, config_overrides=config_overrides)

        already_logged_this_attempt = False

        if success and pending_memory_record:
            record_fix(pending_memory_record[0], pending_memory_record[1])
            print(f"Memory: confirmed fix for this error signature (tool: {pending_memory_record[1]})")
        pending_memory_record = None

        if success:
            print("BUILD SUCCEEDED")
            print("Running validation...")

            benchmark_result = run_benchmark(engine)
            print(f"Latency: {benchmark_result}")

            accuracy_result = run_accuracy_check(onnx_path, engine)
            print(f"Accuracy: {accuracy_result}")

            if accuracy_result.get("passed", False):
                attempts_log.append({
                    "attempt": attempt,
                    "diagnosis": None if attempt == 1 and not force_first_patch else "validation passed",
                    "patch": None,
                    "outcome": "succeeded",
                })
                if on_update:
                    on_update(attempts_log)
                final_result = {"benchmark": benchmark_result, "accuracy": accuracy_result}
                report = generate_report(model_name, onnx_path, attempts_log, final_result)
                print("\n" + report)
                send_notification(model_name, True, final_result)
                return True, engine, final_result

            print("VALIDATION FAILED — build succeeded but accuracy regressed. Retrying with a different fix...")
            attempts_log.append({
                "attempt": attempt,
                "diagnosis": "Build succeeded but accuracy validation failed (regression detected)",
                "patch": "Naive fix applied previously — discarding and asking Gemini for an alternative" if force_first_patch and attempt == 1 else None,
                "outcome": "failed",
            })
            if on_update:
                on_update(attempts_log)
            already_logged_this_attempt = True

            if applied_patches:
                removed = applied_patches.pop()
                print(f"Discarding harmful patch from accumulated set: {removed}")

            prior_attempt_note = (
                f"NOTE: A previous attempt already added an optimization profile and applied a fix, "
                f"but that fix caused the build to pass but FAIL accuracy validation "
                f"(similarity: {accuracy_result.get('worst_cosine_similarity')}). "
                f"Do not repeat that same fix — choose a different tool for a stable, accurate build."
            )

            log = "VALIDATION_REGRESSION: build succeeded but output accuracy failed tolerance check."

        else:
            print("Build failed. Asking Gemini to diagnose...")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        net_summary = summarize_network(network)

        cached_tool = get_cached_fix(log)
        if cached_tool:
            print(f"Memory: found cached fix for this error signature — using '{cached_tool}' without calling Gemini")
            tool_name, tool_args = cached_tool, {}
        else:
            tool_name, tool_args = gemini_diagnose(log, net_summary, prior_attempt_note)

        if tool_name is None:
            best_effort = gemini_best_effort_diagnosis(log)
            print("Gemini did not select a known tool. Best-effort diagnosis for human review:")
            print(best_effort)
            if not already_logged_this_attempt:
                attempts_log.append({
                    "attempt": attempt,
                    "diagnosis": f"UNKNOWN — no automated fix available. Best-effort read: {best_effort}",
                    "patch": None,
                    "outcome": "failed",
                })
                if on_update:
                    on_update(attempts_log)
            report = generate_report(model_name, onnx_path, attempts_log, None)
            print("\n" + report)
            send_notification(model_name, False, {"escalation_reason": best_effort})
            return False, None, None

        print(f"Gemini chose: {tool_name}" if not cached_tool else f"Using memory: {tool_name}")

        error_type_map = {
            "fix_workspace_too_small": "workspace_too_small",
            "fix_missing_optimization_profile": "missing_optimization_profile",
            "fix_int8_calibration_missing": "int8_calibration_missing",
        }

        if tool_name == "generate_custom_patch":
            code = tool_args.get("code", "")
            entry = ("custom", code)
            applied_patches.append(entry)
            pending_memory_record = None
            patch_desc = f"Gemini-generated custom patch:\n{code}"
        else:
            error_type = error_type_map.get(tool_name)
            if error_type not in applied_patches:
                applied_patches.append(error_type)
            pending_memory_record = (log, tool_name)
            patch_desc = f"Applied (accumulated patch set now: {applied_patches})"

        if not already_logged_this_attempt:
            attempts_log.append({
                "attempt": attempt,
                "diagnosis": f"{tool_name} (from memory)" if cached_tool else tool_name,
                "patch": patch_desc,
                "outcome": "failed",
            })
            if on_update:
                on_update(attempts_log)

    print("Max attempts reached without success.")
    report = generate_report(model_name, onnx_path, attempts_log, None)
    print("\n" + report)
    send_notification(model_name, False, None)
    return False, None, None

if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else '/home/anyashan/bert.onnx'
