from celery import Celery
import cnake_charmer.generate.fastapi_service.tasks 

celery_app = Celery(
    "cnake_charmer",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
    include=["cnake_charmer.generate.fastapi_service.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

if __name__ == "__main__":
    celery_app.start()