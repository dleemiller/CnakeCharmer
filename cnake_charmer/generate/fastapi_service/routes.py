from fastapi import APIRouter, BackgroundTasks
from cnake_charmer.generate.fastapi_service.tasks import generate_code_task, check_build_status_task, execute_task
from cnake_charmer.generate.worker.worker import app

router = APIRouter()

@router.post("/generate")
async def generate_code(prompt: str, background_tasks: BackgroundTasks):
    """
    Generate Python & Cython code from a prompt.
    """
    task = generate_code_task.delay(prompt)
    return {"task_id": task.id, "status": "Processing"}

@router.get("/build/{task_id}")
async def check_build_status(task_id: str):
    """
    Check the build status of a generated task.
    """
    result = app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}

@router.get("/execute/{task_id}")
async def execute_code(task_id: str):
    """
    Execute compiled Python & Cython code and check equivalency.
    """
    task = execute_task.delay(task_id)
    return {"task_id": task.id, "status": "Processing"}

@router.get("/logs/{task_id}")
async def fetch_logs(task_id: str):
    """
    Fetch logs for a specific task.
    """
    result = app.AsyncResult(task_id)
    return {"task_id": task_id, "logs": str(result.result)}