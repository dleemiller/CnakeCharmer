FROM python:3.12

# Set working directory inside the container
WORKDIR /app

# Copy the whole repository
COPY . /app

# Ensure the correct package path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
#RUN pip install --no-cache-dir .
RUN pip install .

# Verify Python module installation
RUN python -c "import cnake_charmer.generate.ephemeral_runner.core" || echo 'Module not found!'

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "cnake_charmer.generate.fastapi_service.main:app", "--host", "0.0.0.0", "--port", "8000"]