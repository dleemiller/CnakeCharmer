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

The system consists of five main components:

1. **Web Interface**: A FastAPI-based web server that provides the API endpoints
2. **Socket.io Server**: A Node.js Express server that provides real-time communication
3. **Web Client**: A React-based frontend application for the dashboard
4. **Worker Service**: A background service that processes Cython code analysis jobs
5. **Monitor Service**: A service that monitors for new code files and submits them for analysis
6. **Shared Storage**: Directories for job submissions and analysis results

## Prerequisites

- Docker and Docker Compose
- Basic knowledge of Python, Cython, and web technologies

## Project Structure

The project has the following structure:

```
builder/
├── docker-compose.yml
├── jobs/                 # Shared directory for job submissions
├── results/              # Shared directory for analysis results
│   └── html/             # Sub-directory for HTML annotations
├── web/                  # FastAPI backend
│   ├── Dockerfile
│   ├── main.py           # FastAPI server implementation
│   ├── socket_server.js  # Express Socket.io server
│   ├── Dockerfile.node   # Dockerfile for Socket.io server 
│   ├── package.json      # Node.js dependencies for Socket.io server
│   └── static/
├── web-client/           # React frontend
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── contexts/
│   │   │   ├── JobContext.tsx
│   │   │   └── SocketContext.tsx  # Socket.io context
│   │   ├── types/
│   │   └── api/
│   └── public/
├── client/               # Monitor service
│   ├── Dockerfile
│   └── monitor.py
├── worker/               # Analysis worker
│   ├── Dockerfile
│   └── worker.py
└── pyproject.toml        # Python project configuration
```

## Implementation Details

### Web Interface (FastAPI)

The FastAPI server (`web/main.py`) provides:
- API endpoints for job submission and result retrieval
- Static file serving for the dashboard frontend

Key features:
- Job history tracking
- System status monitoring
- RESTful API for job management

### Socket.io Server (Express.js)

The Socket.io server (`web/socket_server.js`) provides:
- Real-time communication between the backend and frontend
- Job status updates via Socket.io
- System status broadcasts
- Subscription-based job updates

Key features:
- Low-latency real-time updates
- Event-based communication model
- Client reconnection with backoff
- Efficient subscription management

### Web Client (React)

The React frontend (`web-client/`) provides:
- Modern, responsive UI for code submission and analysis
- Real-time job status updates
- Detailed analysis visualization
- Job history tracking

Key features:
- Code editor with syntax highlighting
- Real-time Socket.io updates
- Interactive analysis results
- Job history management

### Monitor Service

The monitor service (`client/monitor.py`) provides:
- Directory monitoring for new Cython files
- Automatic job submission
- WebSocket notifications for job status

### Worker Service

The worker service (`worker/worker.py`) provides:
- Cython code analysis
- Efficiency scoring
- HTML annotation generation
- Result file generation

## Running the System

1. Start the system with Docker Compose:

```bash
docker-compose up -d
```

2. Access the dashboard at http://localhost:3000

3. Monitor the logs:

```bash
docker-compose logs -f
```

## Using the Dashboard

1. Enter or paste Cython code in the editor, or load one of the provided examples.

2. Click "Analyze Code" to submit the code for analysis.

3. Monitor the analysis progress in real-time:
   - Job status updates via WebSocket
   - Real-time progress indicators
   - Automatic result loading

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
  docker-compose logs web
  ```

### Socket.io Server Issues

- Check if the Socket.io server is running:
  ```bash
  docker-compose ps socketio
  ```

- View Socket.io server logs:
  ```bash
  docker-compose logs socketio
  ```

- Check connection issues:
  - Verify that the Socket.io server is running on port 8001
  - Check that the web client is able to connect to the Socket.io server
  - Inspect browser console for connection errors

### Worker Issues

- Check if the worker service is running:
  ```bash
  docker-compose ps
  ```

- View worker logs:
  ```bash
  docker-compose logs worker
  ```

### Monitor Issues

- Check if the monitor service is running:
  ```bash
  docker-compose ps
  ```

- View monitor logs:
  ```bash
  docker-compose logs monitor
  ```

### Web Client Issues

- Check if the web client is running:
  ```bash
  docker-compose ps
  ```

- View web client logs:
  ```bash
  docker-compose logs web-client
  ```

### File Permission Issues

If you encounter permission problems with the shared volumes:

```bash
chmod -R 777 jobs/ results/
```

## Extending the System

### Adding New Analysis Metrics

To add new analysis metrics:

1. Modify the worker's analysis functions to extract additional metrics
2. Update the server's result model to include the new metrics
3. Add UI elements in the dashboard to display the new metrics

### Supporting Additional File Types

To support additional file types (e.g., .py files with Cython annotations):

1. Modify the worker's file detection logic
2. Update the submission API to accept different file extensions

## Conclusion

This Cython Analysis Dashboard provides a powerful way to analyze and optimize Cython code. By visualizing inefficient operations and providing detailed metrics, it helps developers write more performant Cython code.

For further assistance or to report issues, please open a GitHub issue in the project repository.