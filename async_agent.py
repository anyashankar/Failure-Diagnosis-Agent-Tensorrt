import sys
import uuid
import multiprocessing
import traceback
import os
sys.path.insert(0, '/home/anyashan')

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

manager = multiprocessing.Manager()
jobs = manager.dict()
stop_flags = manager.dict()

NOTIF_LOG = os.path.expanduser("~/notifications.log")
DASHBOARD_PATH = os.path.expanduser("~/dashboard.html")

class JobRequest(BaseModel):
    onnx_path: str
    model_name: str = "model"
    force_first_patch: str | None = None

def run_job(job_id: str, onnx_path: str, model_name: str, force_first_patch, jobs_dict, stop_dict):
    sys.path.insert(0, '/home/anyashan')
    from gemini_agent import diagnose_and_fix_with_gemini

    entry = dict(jobs_dict[job_id])
    entry["status"] = "running"
    jobs_dict[job_id] = entry

    def on_update(attempts):
        e = dict(jobs_dict[job_id])
        e["attempts"] = attempts
        jobs_dict[job_id] = e

    def should_stop():
        return stop_dict.get(job_id, False)

    try:
        success, engine, result = diagnose_and_fix_with_gemini(
            onnx_path, model_name=model_name, force_first_patch=force_first_patch,
            on_update=on_update, should_stop=should_stop
        )
        entry = dict(jobs_dict[job_id])
        if stop_dict.get(job_id, False):
            entry["status"] = "stopped"
        else:
            entry["status"] = "completed" if success else "failed"
        entry["result"] = result
        jobs_dict[job_id] = entry
    except Exception as e:
        entry = dict(jobs_dict[job_id])
        entry["status"] = "error"
        entry["error"] = str(e)
        entry["traceback"] = traceback.format_exc()
        jobs_dict[job_id] = entry

@app.get("/")
def serve_dashboard():
    return FileResponse(DASHBOARD_PATH)

@app.post("/jobs")
def create_job(req: JobRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "model_name": req.model_name, "result": None, "attempts": []}
    stop_flags[job_id] = False

    process = multiprocessing.Process(
        target=run_job,
        args=(job_id, req.onnx_path, req.model_name, req.force_first_patch, jobs, stop_flags),
        daemon=True,
    )
    process.start()

    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    if job_id not in jobs:
        return {"error": "job not found"}
    stop_flags[job_id] = True
    return {"job_id": job_id, "status": "stop_requested"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        return {"error": "job not found"}
    return {"job_id": job_id, **jobs[job_id]}

@app.get("/jobs")
def list_jobs():
    return {jid: {"status": j["status"], "model_name": j["model_name"]} for jid, j in jobs.items()}

@app.delete("/jobs")
def clear_jobs():
    jobs.clear()
    stop_flags.clear()
    return {"cleared": True}

@app.get("/notifications")
def get_notifications():
    if not os.path.exists(NOTIF_LOG):
        return {"recent": []}
    with open(NOTIF_LOG) as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    summaries = [b.split("\n")[0] for b in blocks[-20:]]
    return {"recent": summaries}
