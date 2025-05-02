FROM python:3.12-slim

# Install FastAPI and dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pydantic

# Create app directory
WORKDIR /app

# Copy application files
COPY main.py main.py
COPY static ./static

# Create directories for jobs and results
RUN mkdir -p /app/jobs /app/results /app/results/html

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]