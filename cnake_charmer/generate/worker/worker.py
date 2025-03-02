# worker.py (file structure example)

from celery import Celery
import os

# Get configuration from environment variables
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# Create Celery app
app = Celery('cnake_charmer',
             broker=broker_url,
             backend=result_backend,
             include=['cnake_charmer.generate.fastapi_service.tasks'])

# Optional configuration
app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    worker_prefetch_multiplier=1,  # Disable prefetching for fair task distribution
    task_acks_late=True,  # Tasks are acknowledged after execution
    broker_connection_retry_on_startup=True,  # Retry broker connection on startup
)

if __name__ == '__main__':
    app.start()