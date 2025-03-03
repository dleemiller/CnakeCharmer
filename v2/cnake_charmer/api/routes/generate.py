# api/routes/generate.py (updated)
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from core.models import CodeGenerationRequest
from core.enums import LanguageType, AnalysisType, RequestStatus
from repositories.request_repo import CodeRequestRepository
from api.models import (
    CodeGenerationRequestModel, 
    CodeGenerationResponseModel,
    CodeGenerationStatusModel,
    CodeGenerationResultModel
)
from api.dependencies import get_request_repository
from worker.tasks.generate_tasks import generate_code_task


router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/", response_model=CodeGenerationResponseModel)
async def generate_code(
    request: CodeGenerationRequestModel,
    request_repo: CodeRequestRepository = Depends(get_request_repository)
):
    """
    Generate code based on a request.
    
    The code generation will run in the background, and the request ID will be returned immediately.
    Use the /status/{request_id} endpoint to check the status of the request, and the
    /results/{request_id} endpoint to get the results when the request is completed.
    """
    # Prepare the request data for the Celery task
    request_data = {
        "prompt": request.prompt,
        "target_languages": request.target_languages,
        "source_language": request.source_language,
        "source_code": request.source_code,
        "equivalency_check": request.equivalency_check,
        "optimization_level": request.optimization_level,
        "analysis_types": request.analysis_types,
        "max_attempts": request.max_attempts,
        "build": True  # Automatically build after generation
    }
    
    # Submit the task to Celery
    task = generate_code_task.delay(request_data)
    
    # The task ID can be used to check status
    return CodeGenerationResponseModel(request_id=task.id)