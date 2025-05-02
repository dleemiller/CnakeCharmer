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
      numpy \
      beautifulsoup4
# Create directories for jobs and results
RUN mkdir -p /app/jobs /app/results /app/results/html

# Copy worker script
WORKDIR /app
COPY worker.py .

# Set entrypoint
ENTRYPOINT ["python", "worker.py"]