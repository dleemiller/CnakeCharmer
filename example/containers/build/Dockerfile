# containers/build/Dockerfile

FROM python:3.10-slim

# Install a compiler and required tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Cython.
RUN pip install --no-cache-dir cython

# Use /build as our working directory.
WORKDIR /build

# Copy in our build scripts.
COPY setup.py /build/
COPY build.sh /build/
RUN chmod +x /build/build.sh

CMD ["bash"]
