# backend/Dockerfile
FROM python:3.10-slim

# 1. Set workdir
WORKDIR /app

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Create persistent dirs
RUN mkdir -p /app/jobs /app/results

# 5. Expose and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
