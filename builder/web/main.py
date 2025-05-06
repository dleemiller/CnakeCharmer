#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import socketio
import os
import subprocess
import uuid
import json
import threading
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("cython-api")

# Set up Socket.io server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=['*'],
    logger=True,
    engineio_logger=True
)

# Create FastAPI app
app = FastAPI(title="Cython Analyzer", description="API for analyzing Cython code")

# Create Socket.io ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
JOBS_DIR = "/app/jobs"
RESULTS_DIR = "/app/results"
JOB_STATUS = {}  # In-memory job tracking

# Track job subscriptions
job_subscriptions = {}  # job_id -> set of sid
client_subscriptions = {}  # sid -> set of job_id

# Socket.io event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    client_subscriptions[sid] = set()
    # Send initial system status
    try:
        status = await get_system_status()
        await sio.emit('system_status', status, room=sid)
    except Exception as e:
        logger.error(f"Error sending initial status: {e}")


@sio.event
async def subscribe(sid, data):
    job_id = data.get('job_id')
    if not job_id:
        return
    
    logger.info(f"Client {sid} subscribing to job {job_id}")
    
    # Add to job subscriptions
    if job_id not in job_subscriptions:
        job_subscriptions[job_id] = set()
    job_subscriptions[job_id].add(sid)
    
    # Add to client subscriptions
    client_subscriptions[sid].add(job_id)
    
    # Send initial job status if available
    try:
        result = await get_job_result(job_id)
        if result:
            await sio.emit('job_update', {
                'job_id': job_id,
                'status': result.get('status', 'unknown'),
                'result': result
            }, room=sid)
    except Exception as e:
        logger.error(f"Error sending initial job status: {e}")


@sio.event
async def unsubscribe(sid, data):
    job_id = data.get('job_id')
    if not job_id:
        return
    
    logger.info(f"Client {sid} unsubscribing from job {job_id}")
    
    # Remove from job subscriptions
    if job_id in job_subscriptions:
        job_subscriptions[job_id].discard(sid)
        if not job_subscriptions[job_id]:
            del job_subscriptions[job_id]
    
    # Remove from client subscriptions
    if sid in client_subscriptions:
        client_subscriptions[sid].discard(job_id)


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    
    # Clean up subscriptions
    if sid in client_subscriptions:
        job_ids = list(client_subscriptions[sid])
        for job_id in job_ids:
            if job_id in job_subscriptions:
                job_subscriptions[job_id].discard(sid)
                if not job_subscriptions[job_id]:
                    del job_subscriptions[job_id]
        
        del client_subscriptions[sid]


# Models
class CodeSubmission(BaseModel):
    code: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    timestamp: Optional[float] = None


class ManualAnalysisLine(BaseModel):
    file: str
    line_num: int
    content: str


class ManualAnalysis(BaseModel):
    yellow_lines: List[Dict[str, Any]] = []
    red_lines: List[Dict[str, Any]] = []


class AnalysisResult(BaseModel):
    job_id: str
    yellow_lines: int = 0
    red_lines: int = 0
    cython_lint: int = 0
    pep8_issues: int = 0
    timestamp: Optional[float] = None
    error: Optional[str] = None
    status: str = "unknown"
    html_files: List[str] = []
    manual_analysis: Optional[Dict[str, Any]] = None
    detailed_analysis: Optional[Dict[str, Any]] = None


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Routes
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


# Helper function to get job result
async def get_job_result(job_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
            
            # Ensure detailed_analysis is properly included
            if "detailed_analysis" not in result:
                result["detailed_analysis"] = {
                    "message": "Detailed analysis data not available. Please run the analysis again."
                }
            
            return result
        except Exception as e:
            logger.error(f"Error reading result file: {e}", exc_info=True)
            return None
    else:
        status_info = JOB_STATUS.get(job_id, {"status": "unknown"})
        return {
            "job_id": job_id,
            "status": status_info.get("status", "unknown"),
            "yellow_lines": 0,
            "red_lines": 0,
            "cython_lint": 0,
            "pep8_issues": 0,
            "html_files": [],
            "detailed_analysis": {"message": "No analysis results available yet."},
        }


@app.post("/api/submit", response_model=JobStatus)
async def submit_code(submission: CodeSubmission):
    if not submission.code.strip():
        raise HTTPException(status_code=400, detail="Empty code submission")

    logger.info("Received code submission")
    job_id = str(uuid.uuid4())
    logger.info(f"Generated job ID: {job_id}")

    # Create temp directory and save code
    temp_dir = f"/tmp/job-{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Write code to file
        with open(f"{temp_dir}/code.pyx", "w") as f:
            f.write(submission.code)

        # Create tarball
        tarball_path = os.path.join(JOBS_DIR, f"{job_id}.tar.gz")
        logger.info(f"Creating tarball at {tarball_path}")
        subprocess.run(["tar", "czf", tarball_path, "-C", temp_dir, "."], check=True)

        # Track job
        status = {"job_id": job_id, "status": "submitted", "timestamp": time.time()}
        JOB_STATUS[job_id] = status
        logger.info(f"Job {job_id} submitted successfully")

        return status

    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            subprocess.run(["rm", "-rf", temp_dir])


@app.get("/api/results/{job_id}", response_model=AnalysisResult)
async def get_results(job_id: str):
    logger.info(f"Fetching results for job {job_id}")
    result = await get_job_result(job_id)
    
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail=f"No results found for job {job_id}")


@app.get("/api/html/{filename}")
async def get_html_file(filename: str):
    """Return an HTML annotation file"""
    html_dir = os.path.join(RESULTS_DIR, "html")
    file_path = os.path.join(html_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="HTML file not found")

    return FileResponse(file_path, media_type="text/html")


@app.get("/api/analysis/{job_id}")
async def get_code_analysis(job_id: str):
    """Return detailed code analysis results"""
    result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")

    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        with open(result_path, "r") as f:
            result = json.load(f)

        # Get analysis data
        manual_analysis = result.get("manual_analysis", {})
        yellow_lines = manual_analysis.get("yellow_lines", [])
        red_lines = manual_analysis.get("red_lines", [])
        detailed_analysis = result.get("detailed_analysis", {})

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cython Analysis for {job_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .yellow {{ background-color: #FFFF99; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .red {{ background-color: #FFCCCC; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .excellent {{ background-color: #CCFFCC; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .good {{ background-color: #E6FFCC; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .line-number {{ color: #999; margin-right: 10px; }}
                .file-name {{ font-weight: bold; margin-right: 10px; }}
                pre {{ margin: 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .reason {{ font-style: italic; color: #666; }}
                .score-bar {{ height: 20px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Cython Analysis for Job {job_id}</h1>
            
            <h2>Summary</h2>
            <p>Yellow Lines: {len(yellow_lines)}</p>
            <p>Red Lines: {len(red_lines)}</p>
            
            <h2>Yellow Lines (Python operations that could be optimized)</h2>
        """

        if yellow_lines:
            for line in yellow_lines:
                file_name = line.get("file", "unknown")
                line_num = line.get("line_num", 0)
                content = line.get("content", "")
                reason = line.get("reason", "")
                html_content += f"""
                <div class="yellow">
                    <span class="file-name">{file_name}</span>
                    <span class="line-number">Line {line_num}</span>
                    <pre>{content}</pre>
                    <div class="reason">Reason: {reason}</div>
                </div>
                """
        else:
            html_content += "<p>No yellow lines detected.</p>"

        html_content += """
            <h2>Red Lines (Python operations that are difficult to optimize)</h2>
        """

        if red_lines:
            for line in red_lines:
                file_name = line.get("file", "unknown")
                line_num = line.get("line_num", 0)
                content = line.get("content", "")
                reason = line.get("reason", "")
                html_content += f"""
                <div class="red">
                    <span class="file-name">{file_name}</span>
                    <span class="line-number">Line {line_num}</span>
                    <pre>{content}</pre>
                    <div class="reason">Reason: {reason}</div>
                </div>
                """
        else:
            html_content += "<p>No red lines detected.</p>"

        html_content += """
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error generating analysis HTML: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating analysis HTML: {str(e)}"
        )


@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    logger.info("Fetching system status")
    
    try:
        # Get pending jobs
        pending_jobs = [
            f for f in os.listdir(JOBS_DIR) if f.endswith(".tar.gz")
        ]
        
        # Get completed jobs
        completed_jobs = [
            f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")
        ]
        
        # Get archived jobs (if any)
        archived_dir = os.path.join(RESULTS_DIR, "archived")
        archived_jobs = []
        if os.path.exists(archived_dir):
            archived_jobs = [
                f for f in os.listdir(archived_dir) if f.endswith(".json")
            ]
        
        # Return status
        return {
            "status": "running",
            "pending_jobs": pending_jobs,
            "completed_jobs": completed_jobs,
            "archived_jobs": archived_jobs,
            "in_memory_status": JOB_STATUS,
        }
    
    except Exception as e:
        logger.error(f"Error fetching system status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "pending_jobs": [],
            "completed_jobs": [],
            "archived_jobs": [],
        }


# Monitor results and broadcast updates
async def monitor_results():
    logger.info("Starting result monitor")
    last_check = time.time()
    processed_files = set()
    
    while True:
        try:
            current_time = time.time()
            
            # Get all result files modified after the last check
            result_files = []
            for filename in os.listdir(RESULTS_DIR):
                if filename.endswith(".json"):
                    file_path = os.path.join(RESULTS_DIR, filename)
                    mod_time = os.path.getmtime(file_path)
                    if mod_time > last_check and file_path not in processed_files:
                        result_files.append((filename, file_path, mod_time))
            
            # Process new/updated results
            for filename, file_path, mod_time in result_files:
                job_id = filename.replace(".json", "")
                logger.info(f"Processing updated result for job {job_id}")
                
                try:
                    with open(file_path, "r") as f:
                        result = json.load(f)
                    
                    # Update in-memory status
                    if job_id in JOB_STATUS:
                        JOB_STATUS[job_id]["status"] = result.get("status", "unknown")
                    
                    # Broadcast update via Socket.io
                    if job_id in job_subscriptions and job_subscriptions[job_id]:
                        logger.info(f"Broadcasting update for job {job_id} to {len(job_subscriptions[job_id])} subscribers")
                        message = {
                            "job_id": job_id,
                            "status": result.get("status", "unknown"),
                            "result": result
                        }
                        for sid in job_subscriptions[job_id]:
                            await sio.emit('job_update', message, room=sid)
                    
                    processed_files.add(file_path)
                
                except Exception as e:
                    logger.error(f"Error processing result file {file_path}: {e}", exc_info=True)
            
            # Get system status and broadcast to all
            system_status = await get_system_status()
            await sio.emit('system_status', system_status)
            
            # Update last check time
            last_check = current_time
            
            # Sleep for a while
            await asyncio.sleep(2)  # Check every 2 seconds
        
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait a bit longer on error


@app.on_event("startup")
async def startup_event():
    # Create necessary directories
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "html"), exist_ok=True)
    
    # Start result monitor as a background task
    asyncio.create_task(monitor_results())
    logger.info("Cython Analyzer API started")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
