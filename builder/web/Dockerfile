FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy pyproject.toml for package installation
COPY pyproject.toml .
RUN pip install -e .

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with Socket.io integration
CMD ["uvicorn", "main:socket_app", "--host", "0.0.0.0", "--port", "8000"]