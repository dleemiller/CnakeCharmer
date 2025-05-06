# backend/main.py

import os
import time
import json
import asyncio
import logging
import subprocess
import uuid

from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse, HTMLResponse
import socketio

# ── Logging ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cython-api")

# ── Paths & In‐Memory State ───────────────────────────────────────────────────────
JOBS_DIR = os.getenv("JOBS_DIR", "/app/jobs")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/results")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Track job metadata in memory
JOB_STATUS: dict[str, dict] = {}

# ── FastAPI App ─────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="Cython Analyzer API")


@fastapi_app.post("/api/submit")
async def submit_code(submission: dict):
    code = submission.get("code", "").strip()
    if not code:
        raise HTTPException(400, "Empty code submission")

    job_id = str(uuid.uuid4())
    temp = f"/tmp/job-{job_id}"
    os.makedirs(temp, exist_ok=True)
    with open(f"{temp}/code.pyx", "w") as f:
        f.write(code)

    tarball = os.path.join(JOBS_DIR, f"{job_id}.tar.gz")
    subprocess.run(["tar", "czf", tarball, "-C", temp, "."], check=True)
    subprocess.run(["rm", "-rf", temp])

    status = {"job_id": job_id, "status": "submitted", "timestamp": time.time()}
    JOB_STATUS[job_id] = status
    logger.info(f"Job submitted: {job_id}")
    return status


@fastapi_app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        result.setdefault("detailed_analysis", {"message": "Not available"})
        return result

    # fallback
    st = JOB_STATUS.get(job_id, {"status": "unknown"})
    return {
        **st,
        "yellow_lines": 0,
        "red_lines": 0,
        "cython_lint": 0,
        "pep8_issues": 0,
        "html_files": [],
        "detailed_analysis": {"message": "Pending..."},
    }


@fastapi_app.get("/api/html/{filename}")
async def get_html(filename: str):
    p = os.path.join(RESULTS_DIR, "html", filename)
    if not os.path.exists(p):
        raise HTTPException(404, "HTML file not found")
    return FileResponse(p, media_type="text/html")


@fastapi_app.get("/api/analysis/{job_id}")
async def get_full_analysis(job_id: str):
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Analysis not found")
    with open(path) as f:
        data = json.load(f)
    return HTMLResponse(content=f"<pre>{json.dumps(data, indent=2)}</pre>")


@fastapi_app.get("/api/status")
async def system_status():
    pending = [f for f in os.listdir(JOBS_DIR) if f.endswith(".tar.gz")]
    completed = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    return {
        "status": "running",
        "pending_jobs": pending,
        "completed_jobs": completed,
        "in_memory_status": JOB_STATUS,
    }


# ── Socket.IO Setup ──────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*", logger=True, engineio_logger=True
)


@sio.event
async def connect(sid, environ):
    logger.info(f"{sid} connected")
    sio.enter_room(sid, "system")
    # immediately send the current system status
    await sio.emit("system_status", await system_status(), room=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"{sid} disconnected")


@sio.on("subscribe")
async def subscribe(sid, data):
    jid = data.get("job_id")
    if jid:
        sio.enter_room(sid, jid)
        logger.info(f"{sid} subscribed to {jid}")


@sio.on("unsubscribe")
async def unsubscribe(sid, data):
    jid = data.get("job_id")
    if jid:
        sio.leave_room(sid, jid)
        logger.info(f"{sid} unsubscribed from {jid}")


# ── Background Monitoring ────────────────────────────────────────────────────────
async def monitor_results():
    logger.info("Result monitor started")
    last_seen: dict[str, float] = {}
    last_system_hash = ""

    while True:
        # 1️⃣ Check individual job files for updates
        for fn in os.listdir(RESULTS_DIR):
            if not fn.endswith(".json"):
                continue
            jid = fn[:-5]
            path = os.path.join(RESULTS_DIR, fn)
            mtime = os.path.getmtime(path)
            if last_seen.get(jid, 0) < mtime:
                last_seen[jid] = mtime
                with open(path) as f:
                    result = json.load(f)
                new_status = result.get("status", "completed")
                JOB_STATUS[jid]["status"] = new_status

                payload = {
                    "type": "job_update",
                    "job_id": jid,
                    "status": new_status,
                    "result": result,
                }
                await sio.emit("job_update", payload, room=jid)
                logger.info(f"Emitted job_update for {jid}")

        # 2️⃣ Compute system snapshot and emit only on change
        sys_state = {
            "status": "running",
            "pending_jobs": [f for f in os.listdir(JOBS_DIR) if f.endswith(".tar.gz")],
            "completed_jobs": [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
            "in_memory_status": JOB_STATUS,
        }
        sys_hash = str(hash(json.dumps(sys_state, sort_keys=True)))
        if sys_hash != last_system_hash:
            last_system_hash = sys_hash
            await sio.emit("system_status", sys_state, room="system")
            logger.info("Emitted system_status (changed)")

        await asyncio.sleep(1)


@fastapi_app.on_event("startup")
async def on_startup():
    asyncio.create_task(monitor_results())
    logger.info("Startup complete, monitoring results")


# ── Combine FastAPI + Socket.IO into one ASGI App ───────────────────────────────
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
