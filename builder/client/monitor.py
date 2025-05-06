#!/usr/bin/env python3
import os
import time
import subprocess
import json
import uuid
import sys
import argparse
import asyncio
import websockets
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("cython-monitor")

# Default directories
DEFAULT_WATCH_DIR = "./generated_code"
DEFAULT_JOBS_DIR = "./jobs"
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_WS_URL = "ws://localhost:8000/ws/"

async def notify_websocket(job_id: str, status: str, result: dict = None):
    """Notify the WebSocket server about job status changes"""
    try:
        async with websockets.connect(f"{DEFAULT_WS_URL}{job_id}") as websocket:
            message = {
                "type": "job_update",
                "job_id": job_id,
                "status": status,
                "result": result
            }
            await websocket.send(json.dumps(message))
    except Exception as e:
        logger.error(f"Failed to notify WebSocket for job {job_id}: {e}")

def setup_directories(watch_dir, jobs_dir, results_dir):
    """Create necessary directories if they don't exist"""
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "processed"), exist_ok=True)
    logger.info(f"Watching directory: {watch_dir}")
    logger.info(f"Jobs directory: {jobs_dir}")
    logger.info(f"Results directory: {results_dir}")

async def process_file(file_path, jobs_dir):
    """Process a Cython file and submit it as a job"""
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/job-{job_id}"

    # Create temp directory and copy file
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(temp_dir, file_name)
    subprocess.run(["cp", file_path, dest_path])

    # Create tarball
    tarball_path = os.path.join(jobs_dir, f"{job_id}.tar.gz")
    subprocess.run(["tar", "czf", tarball_path, "-C", temp_dir, "."])

    # Clean up temp directory
    subprocess.run(["rm", "-rf", temp_dir])

    logger.info(f"Submitted {file_path} as job {job_id}")
    
    # Notify WebSocket about new job
    await notify_websocket(job_id, "submitted")
    
    return job_id

async def check_file_changes(watch_dir, jobs_dir, last_modified_times):
    """Check for modified Cython files and process them"""
    for root, _, files in os.walk(watch_dir):
        for file in files:
            if file.endswith(".pyx"):
                file_path = os.path.join(root, file)
                modified_time = os.path.getmtime(file_path)

                # If file is new or modified
                if (
                    file_path not in last_modified_times
                    or last_modified_times[file_path] < modified_time
                ):
                    last_modified_times[file_path] = modified_time
                    await process_file(file_path, jobs_dir)

    return last_modified_times

async def monitor_results(results_dir):
    """Check for and process new results"""
    new_results = []
    processed_result_hashes = {}  # Track processed results by their content hash
    
    for result_file in os.listdir(results_dir):
        if result_file.endswith(".json"):
            result_path = os.path.join(results_dir, result_file)
            processed_path = os.path.join(results_dir, "processed", result_file)

            # Skip already processed results
            if os.path.exists(processed_path):
                continue

            # Read the result file
            with open(result_path, "r") as f:
                try:
                    result = json.load(f)
                    job_id = result.get("job_id")
                    
                    # Generate a hash of the result content
                    result_str = json.dumps(result, sort_keys=True)
                    result_hash = hash(result_str)
                    
                    # Skip if we've already processed this exact result
                    if job_id in processed_result_hashes and processed_result_hashes[job_id] == result_hash:
                        logger.debug(f"Skipping already processed result for job {job_id}")
                        continue
                    
                    # Store the processed result hash
                    processed_result_hashes[job_id] = result_hash
                    
                    logger.info(f"\nResult for job {job_id}:")
                    logger.info(f"  Yellow lines: {result.get('yellow_lines', 0)}")
                    logger.info(f"  Red lines: {result.get('red_lines', 0)}")
                    logger.info(f"  Linting issues: {result.get('cython_lint', 0)}")
                    logger.info(f"  PEP8 issues: {result.get('pep8_issues', 0)}")

                    # Move to processed
                    os.rename(result_path, processed_path)
                    new_results.append(result)
                    
                    # Notify WebSocket about completed job
                    await notify_websocket(job_id, "completed", result)
                    logger.info(f"Sent WebSocket notification for completed job {job_id}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Error parsing result file {result_file}")

    return new_results

async def main():
    parser = argparse.ArgumentParser(
        description="Monitor directory for Cython files and submit to worker"
    )
    parser.add_argument(
        "--watch-dir",
        default=DEFAULT_WATCH_DIR,
        help="Directory to watch for .pyx files",
    )
    parser.add_argument(
        "--jobs-dir", default=DEFAULT_JOBS_DIR, help="Directory for job submissions"
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory for results"
    )
    parser.add_argument(
        "--interval", type=int, default=2, help="Polling interval in seconds"
    )
    args = parser.parse_args()

    # Setup
    setup_directories(args.watch_dir, args.jobs_dir, args.results_dir)

    # Main monitoring loop
    last_modified_times = {}

    logger.info("Monitoring for Cython file changes... (Ctrl+C to exit)")

    try:
        while True:
            # Check for file changes
            last_modified_times = await check_file_changes(
                args.watch_dir, args.jobs_dir, last_modified_times
            )

            # Check for new results
            await monitor_results(args.results_dir)

            # Wait before checking again
            await asyncio.sleep(args.interval)
            sys.stdout.write(".")
            sys.stdout.flush()

    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped.")


if __name__ == "__main__":
    asyncio.run(main())
