FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create and activate virtual environment
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy pyproject.toml first to leverage Docker cache
COPY pyproject.toml .
RUN uv pip install -e .

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/generated_code /app/jobs /app/results /app/results/processed

# Command to run the monitor
CMD ["python", "monitor.py"] 