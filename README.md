# TensorRT Build Failure Agent

An autonomous agent (Gemini 3.5 Flash) that diagnoses failed TensorRT builds,
applies targeted patches, rebuilds, and validates latency/accuracy — closing
the loop without human intervention.

## Components
- `gemini_agent.py` — core diagnose → patch → rebuild → validate loop
- `apply_patch.py` — known-fix patch functions (workspace, optimization profile, INT8 calibration, naive FP16 baseline)
- `sandbox_exec.py` — sandboxed execution for Gemini-generated patches on novel error types
- `int8_calibrator.py` — real INT8 post-training calibration
- `candidate_comparison.py` — multi-candidate fix evaluation
- `agent_memory.py` — cross-run error-signature caching
- `async_agent.py` — FastAPI async job server
- `dashboard.html` — live operator dashboard

Note: requires a GPU with TensorRT/CUDA installed and a `GEMINI_API_KEY`
environment variable set. Model files (.onnx) are not included — see
`run_build.py` for expected paths.
