# Cython Analysis Dashboard - Implementation Guide

This guide provides step-by-step instructions for setting up and running the improved Cython Analysis Dashboard system.

## Overview

The Cython Analysis Dashboard is a web-based tool that helps developers analyze and optimize their Cython code by:

1. Detecting inefficient Python operations in Cython code
2. Categorizing issues by severity (yellow and red lines)
3. Providing detailed metrics and visualizations
4. Generating annotated HTML output for line-by-line analysis
5. Tracking job history and analysis results

## System Components

The system consists of three main components:

1. **Web Interface**: A FastAPI-based web server that provides the frontend and API endpoints
2. **Worker Service**: A background service that processes Cython code analysis jobs
3. **Shared Storage**: Directories for job submissions and analysis results

## Prerequisites

- Docker and Docker Compose
- Basic knowledge of Python, Cython, and web technologies

## Project Structure

Create the following directory structure:

```
cython-analyzer/
├── docker-compose.yml
├── jobs/                 # Shared directory for job submissions
├── results/              # Shared directory for analysis results
│   └── html/             # Sub-directory for HTML annotations
├── web/
│   ├── Dockerfile
│   ├── main.py         # FastAPI server implementation
│   └── static/
│       └── index.html    # Dashboard frontend
└── worker/
    ├── Dockerfile
    └── worker.py         # Cython analysis worker script
```

## Implementation Steps

### 1. Set Up the Project Structure

```bash
mkdir -p cython-analyzer/{jobs,results/html,web/static,worker}
cd cython-analyzer
```

### 2. Create the Web Interface

#### 2.1. Create the FastAPI Server (web/main.py)

Copy the provided server code into `web/main.py`. This implements:
- API endpoints for job submission and result retrieval
- Static file serving for the dashboard frontend
- Background monitoring of job results

#### 2.2. Create the Dashboard Frontend (web/static/index.html)

Copy the provided HTML code into `web/static/index.html`. This implements:
- Code editor with example loading
- Real-time job status updates
- Detailed analysis visualization
- Job history tracking

#### 2.3. Create the Web Dockerfile (web/Dockerfile)

```dockerfile
FROM python:3.12-slim

# Install FastAPI and dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-multipart

# Create app directory
WORKDIR /app

# Create directories for jobs and results
RUN mkdir -p /app/jobs /app/results /app/results/html /app/static

# Copy application files
COPY main.py .
COPY static/index.html ./static/

# Expose port
EXPOSE 8000

# Run the app
CMD ["python3", "main.py"]
```

### 3. Create the Worker Service

#### 3.1. Create the Worker Script (worker/worker.py)

Copy the provided worker code into `worker/worker.py`. This implements:
- Monitoring for new job submissions
- Cython compilation and annotation
- Code efficiency analysis
- Result generation and storage

#### 3.2. Create the Worker Dockerfile (worker/Dockerfile)

```dockerfile
FROM python:3.12-slim

# Install system build dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

# Install Cython + linters + wheel + setuptools
RUN pip install --no-cache-dir \
      cython \
      cython-lint \
      flake8 \
      wheel \
      setuptools \
      numpy 

# Create directories for jobs and results
RUN mkdir -p /app/jobs /app/results /app/results/html

# Copy worker script
WORKDIR /app
COPY worker.py .

# Set entrypoint
ENTRYPOINT ["python3", "worker.py"]
```

### 4. Create the Docker Compose Configuration

Create a `docker-compose.yml` file in the project root:

```yaml
version: '3'

services:
  cython-worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    volumes:
      - ./jobs:/app/jobs
      - ./results:/app/results
    restart: always
    
  web-interface:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./jobs:/app/jobs
      - ./results:/app/results
    depends_on:
      - cython-worker
```

## Running the System

1. Start the system with Docker Compose:

```bash
docker-compose up -d
```

2. Access the dashboard at http://localhost:8000

3. Monitor the logs:

```bash
docker-compose logs -f
```

## Using the Dashboard

1. Enter or paste Cython code in the editor, or load one of the provided examples.

2. Click "Analyze Code" to submit the code for analysis.

3. Monitor the analysis progress in real-time.

4. Once completed, view the results in the Summary tab:
   - Yellow Lines: Python operations that could be optimized
   - Red Lines: Python operations that are difficult to optimize
   - Cython Lint Issues: Style and best practice issues
   - PEP8 Issues: Python style guide compliance issues

5. Click on HTML Annotation links to view detailed line-by-line analysis.

6. Explore the Detailed Analysis tab for:
   - Score distribution across efficiency categories
   - Efficiency metrics (average score, API heavy lines, etc.)
   - Specific reasons for yellow and red line classifications

7. Use the Job History tab to view previously submitted jobs and their results.

## Troubleshooting

### Server Issues

- Check if the FastAPI server is running:
  ```bash
  docker-compose ps
  ```

- View server logs:
  ```bash
  docker-compose logs web-interface
  ```

### Worker Issues

- Check if the worker service is running:
  ```bash
  docker-compose ps
  ```

- View worker logs:
  ```bash
  docker-compose logs cython-worker
  ```

### File Permission Issues

If you encounter permission problems with the shared volumes:

```bash
chmod -R 777 jobs/ results/
```

## Extending the System

### Adding New Analysis Metrics

To add new analysis metrics:

1. Modify the worker's `parse_cython_html` function to extract additional metrics
2. Update the server's result model to include the new metrics
3. Add UI elements in the dashboard to display the new metrics

### Supporting Additional File Types

To support additional file types (e.g., .py files with Cython annotations):

1. Modify the worker's file detection logic in the `compile_and_annotate` function
2. Update the submission API to accept different file extensions

## Conclusion

This Cython Analysis Dashboard provides a powerful way to analyze and optimize Cython code. By visualizing inefficient operations and providing detailed metrics, it helps developers write more performant Cython code.

For further assistance or to report issues, please open a GitHub issue in the project repository.