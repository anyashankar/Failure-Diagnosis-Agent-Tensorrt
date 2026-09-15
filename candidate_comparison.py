import sys
sys.path.insert(0, '/home/anyashan')
import tensorrt as trt
from google import genai
from google.genai import types
from run_build import run_build
from apply_patch import apply_patch
from run_benchmark import run_benchmark
from run_accuracy_check import run_accuracy_check
import os

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# The known, safe patch set — used as candidates to compare when a regression occurs.
CANDIDATE_TOOLS = ["workspace_too_small", "int8_calibration_missing"]

def get_top_candidates(error_context: str, num_candidates: int = 2) -> list:
    """
    Asks Gemini to rank which of the known safe fixes are worth trying, given
    a regression context. Returns a list of error_type strings, best guess first.
    Falls back to trying all known candidates if Gemini's response can't be parsed.
    """
    prompt = f"""A previous fix caused a build to succeed but fail accuracy validation.
Context: {error_context}

From this list of possible fixes, rank the ones most likely to resolve this
WITHOUT causing another accuracy regression: {CANDIDATE_TOOLS}

Respond with ONLY a comma-separated list of the tool names, best first, nothing else."""

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        candidates = [c.strip() for c in response.text.strip().split(",") if c.strip() in CANDIDATE_TOOLS]
        return candidates if candidates else CANDIDATE_TOOLS
    except Exception:
        return CANDIDATE_TOOLS

def evaluate_candidate(onnx_path: str, base_patches: list, candidate_error_type: str):
    """
    Builds and validates ONE candidate fix, on top of the base (already-working)
    patches. Returns a dict with the result, or None if the build failed outright.
    """
    def config_overrides(builder, network, config):
        for err_type in base_patches:
            apply_patch(err_type, builder, network, config)
        apply_patch(candidate_error_type, builder, network, config)

    success, engine, log = run_build(onnx_path, config_overrides=config_overrides)
    if not success:
        return None

    benchmark = run_benchmark(engine)
    accuracy = run_accuracy_check(onnx_path, engine)

    return {
        "candidate": candidate_error_type,
        "benchmark": benchmark,
        "accuracy": accuracy,
    }

def compare_candidates_and_pick_best(onnx_path: str, base_patches: list, error_context: str, num_candidates: int = 2):
    """
    Generates candidate fixes, builds and validates each independently, and
    returns the best-performing one that passes validation (highest worst-case
    cosine similarity among passing candidates), or None if none pass.
    """
    candidates = get_top_candidates(error_context, num_candidates)[:num_candidates]
    print(f"Evaluating {len(candidates)} candidate fixes: {candidates}")

    results = []
    for c in candidates:
        print(f"  Trying candidate: {c}")
        result = evaluate_candidate(onnx_path, base_patches, c)
        if result is None:
            print(f"    Build failed for candidate '{c}' — skipping")
            continue
        acc = result["accuracy"]
        print(f"    Result: passed={acc.get('passed')}, worst_cosine_similarity={acc.get('worst_cosine_similarity')}")
        results.append(result)

    passing = [r for r in results if r["accuracy"].get("passed")]
    if not passing:
        print("No candidates passed validation.")
        return None

    best = max(passing, key=lambda r: r["accuracy"].get("worst_cosine_similarity", 0))
    print(f"Best candidate: {best['candidate']} (worst_cosine_similarity={best['accuracy'].get('worst_cosine_similarity')})")
    return best
