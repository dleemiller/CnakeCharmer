# worker/celery.py
import os
import logging
from celery import Celery


logger = logging.getLogger(__name__)

# Configure Celery
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

app = Celery('cnake_charmer',
             broker=broker_url,
             backend=result_backend,
             include=['worker.tasks.generate_tasks',
                      'worker.tasks.build_tasks',
                      'worker.tasks.analyze_tasks'])

# Configure Celery
app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,  # Disable prefetching
    task_acks_late=True,  # Tasks are acknowledged after execution
    broker_connection_retry_on_startup=True,  # Retry broker connection on startup
)

# Optional: Configure task routing
app.conf.task_routes = {
    'worker.tasks.generate_tasks.*': {'queue': 'generate'},
    'worker.tasks.build_tasks.*': {'queue': 'build'},
    'worker.tasks.analyze_tasks.*': {'queue': 'analyze'},
}

# Optional: Define periodic tasks
# app.conf.beat_schedule = {}


if __name__ == '__main__':
    app.start()