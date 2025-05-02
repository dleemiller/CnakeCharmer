#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
import uuid
import json
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("cython-api")

# FastAPI app
app = FastAPI(title="Cython Analyzer", description="API for analyzing Cython code")

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
    result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")

    if os.path.exists(result_path):
        logger.info(f"Result file found for job {job_id}")
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
            logger.info(
                f"Result for {job_id}: Status={result.get('status', 'unknown')}"
            )

            # Ensure detailed_analysis is properly included
            if "detailed_analysis" not in result:
                result["detailed_analysis"] = {
                    "message": "Detailed analysis data not available. Please run the analysis again."
                }

            return result
        except Exception as e:
            logger.error(f"Error reading result file: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error reading result file: {str(e)}"
            )
    else:
        logger.info(
            f"No result file found for job {job_id}, returning status from memory"
        )
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
            <h1>Cython Analysis Results</h1>
            <p>Job ID: {job_id}</p>
            <p>Yellow Lines: {result.get("yellow_lines", 0)}</p>
            <p>Red Lines: {result.get("red_lines", 0)}</p>
            
            <h2>Score Distribution</h2>
        """

        # Add score distribution if available
        if detailed_analysis:
            for file_name, file_analysis in detailed_analysis.items():
                html_content += f"<h3>File: {file_name}</h3>"

                # Score distribution
                score_distribution = file_analysis.get("score_distribution", {})
                if score_distribution:
                    html_content += "<table><tr><th>Category</th><th>Count</th><th>Percentage</th><th>Visualization</th></tr>"

                    categories = [
                        ("excellent", "Excellent (80-255)", "#CCFFCC"),
                        ("good", "Good (40-79)", "#E6FFCC"),
                        ("yellow", "Yellow (20-39)", "#FFFF99"),
                        ("red", "Red (0-19)", "#FFCCCC"),
                    ]

                    for key, label, color in categories:
                        if key in score_distribution:
                            count = score_distribution[key].get("count", 0)
                            percentage = score_distribution[key].get("percentage", 0)
                            html_content += f"""
                            <tr>
                                <td>{label}</td>
                                <td>{count}</td>
                                <td>{percentage}%</td>
                                <td><div class="score-bar" style="width: {percentage}%; background-color: {color};"></div></td>
                            </tr>
                            """

                    html_content += "</table>"

                # Efficiency metrics
                efficiency_metrics = file_analysis.get("efficiency_metrics", {})
                if efficiency_metrics:
                    html_content += "<h2>Efficiency Metrics</h2>"
                    html_content += "<table><tr><th>Metric</th><th>Value</th></tr>"

                    if "avg_score" in efficiency_metrics:
                        html_content += f"""
                        <tr>
                            <td>Average Score</td>
                            <td>{efficiency_metrics["avg_score"]:.2f}</td>
                        </tr>
                        """

                    if "python_api_heavy_lines" in efficiency_metrics:
                        html_content += f"""
                        <tr>
                            <td>Python API Heavy Lines</td>
                            <td>{efficiency_metrics["python_api_heavy_lines"]}</td>
                        </tr>
                        """

                    if "exception_handling_lines" in efficiency_metrics:
                        html_content += f"""
                        <tr>
                            <td>Exception Handling Lines</td>
                            <td>{efficiency_metrics["exception_handling_lines"]}</td>
                        </tr>
                        """

                    html_content += "</table>"

                # Red line reasons
                red_line_reasons = file_analysis.get("red_line_reasons", [])
                if red_line_reasons:
                    html_content += "<h2>Red Line Reasons (Very Inefficient)</h2>"
                    html_content += (
                        "<table><tr><th>Line</th><th>Content</th><th>Reason</th></tr>"
                    )

                    for reason in red_line_reasons:
                        html_content += f"""
                        <tr class="red">
                            <td>{reason.get("line", "")}</td>
                            <td><pre>{reason.get("content", "")}</pre></td>
                            <td class="reason">{reason.get("reason", "")}</td>
                        </tr>
                        """

                    html_content += "</table>"

                # Yellow line reasons
                yellow_line_reasons = file_analysis.get("yellow_line_reasons", [])
                if yellow_line_reasons:
                    html_content += (
                        "<h2>Yellow Line Reasons (Potentially Inefficient)</h2>"
                    )
                    html_content += (
                        "<table><tr><th>Line</th><th>Content</th><th>Reason</th></tr>"
                    )

                    for reason in yellow_line_reasons:
                        html_content += f"""
                        <tr class="yellow">
                            <td>{reason.get("line", "")}</td>
                            <td><pre>{reason.get("content", "")}</pre></td>
                            <td class="reason">{reason.get("reason", "")}</td>
                        </tr>
                        """

                    html_content += "</table>"

        html_content += """
            <h2>Cython HTML Annotations</h2>
            <ul>
        """

        for html_file in result.get("html_files", []):
            html_content += f'<li><a href="/api/html/{html_file}" target="_blank">{html_file}</a></li>'

        html_content += """
            </ul>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error generating analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating analysis: {str(e)}"
        )


@app.get("/api/status")
async def get_system_status():
    """Return the status of the whole system including jobs and results"""
    job_files = [f for f in os.listdir(JOBS_DIR) if f.endswith(".tar.gz")]
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]

    archive_dir = os.path.join(JOBS_DIR, "archive")
    archived_jobs = []
    if os.path.exists(archive_dir):
        archived_jobs = [f for f in os.listdir(archive_dir) if f.endswith(".tar.gz")]

    return {
        "status": "running",
        "pending_jobs": job_files,
        "completed_jobs": result_files,
        "archived_jobs": archived_jobs,
        "in_memory_status": JOB_STATUS,
    }


def monitor_results():
    """Background task to monitor for new results"""
    logger.info("Starting result monitoring thread")
    while True:
        try:
            for result_file in os.listdir(RESULTS_DIR):
                if result_file.endswith(".json"):
                    job_id = result_file.replace(".json", "")

                    if job_id in JOB_STATUS:
                        # Update status from the result file
                        try:
                            with open(os.path.join(RESULTS_DIR, result_file), "r") as f:
                                result = json.load(f)
                                new_status = result.get("status", "completed")
                                if JOB_STATUS[job_id].get("status") != new_status:
                                    JOB_STATUS[job_id]["status"] = new_status
                                    logger.info(
                                        f"Updated job {job_id} status to {new_status}"
                                    )
                        except Exception as e:
                            logger.error(
                                f"Error reading result file {result_file}: {e}"
                            )

            # Clean up old statuses (over 1 hour)
            current_time = time.time()
            old_jobs = []
            for job_id, status in JOB_STATUS.items():
                if current_time - status.get("timestamp", 0) > 3600:  # 1 hour
                    old_jobs.append(job_id)

            for job_id in old_jobs:
                del JOB_STATUS[job_id]
                logger.info(f"Removed old job status for {job_id}")

        except Exception as e:
            logger.error(f"Error in monitor thread: {e}", exc_info=True)

        time.sleep(5)  # Check every 5 seconds


@app.on_event("startup")
async def startup_event():
    # Create necessary directories
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "html"), exist_ok=True)
    archive_dir = os.path.join(JOBS_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    logger.info(
        f"API server started. Jobs directory: {JOBS_DIR}, Results directory: {RESULTS_DIR}"
    )

    # Start result monitoring in background
    monitor_thread = threading.Thread(target=monitor_results, daemon=True)
    monitor_thread.start()
    logger.info("Background monitoring thread started")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
