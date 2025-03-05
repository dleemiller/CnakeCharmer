# containers/run/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the run script.
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["bash"]
