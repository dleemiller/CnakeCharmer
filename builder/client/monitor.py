#!/usr/bin/env python3
import os
import time
import subprocess
import json
import uuid
import sys
import argparse

# Default directories
DEFAULT_WATCH_DIR = "./generated_code"
DEFAULT_JOBS_DIR = "./jobs"
DEFAULT_RESULTS_DIR = "./results"


def setup_directories(watch_dir, jobs_dir, results_dir):
    """Create necessary directories if they don't exist"""
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "processed"), exist_ok=True)
    print(f"Watching directory: {watch_dir}")
    print(f"Jobs directory: {jobs_dir}")
    print(f"Results directory: {results_dir}")


def process_file(file_path, jobs_dir):
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

    print(f"Submitted {file_path} as job {job_id}")
    return job_id


def check_file_changes(watch_dir, jobs_dir, last_modified_times):
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
                    process_file(file_path, jobs_dir)

    return last_modified_times


def monitor_results(results_dir):
    """Check for and process new results"""
    new_results = []
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
                    print(f"\nResult for job {result.get('job_id')}:")
                    print(f"  Yellow lines: {result.get('yellow_lines', 0)}")
                    print(f"  Red lines: {result.get('red_lines', 0)}")
                    print(f"  Linting issues: {result.get('cython_lint', 0)}")
                    print(f"  PEP8 issues: {result.get('pep8_issues', 0)}")

                    # Move to processed
                    os.rename(result_path, processed_path)
                    new_results.append(result)
                except json.JSONDecodeError:
                    print(f"Error parsing result file {result_file}")

    return new_results


def main():
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

    print("Monitoring for Cython file changes... (Ctrl+C to exit)")

    try:
        while True:
            # Check for file changes
            last_modified_times = check_file_changes(
                args.watch_dir, args.jobs_dir, last_modified_times
            )

            # Check for new results
            monitor_results(args.results_dir)

            # Wait before checking again
            time.sleep(args.interval)
            sys.stdout.write(".")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
