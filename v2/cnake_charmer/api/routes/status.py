# api/routes/status.py
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException

from core.enums import RequestStatus
from services.code_generator import CodeGeneratorService
from repositories.request_repo import CodeRequestRepository
from api.models import CodeGenerationStatusModel
from api.dependencies import get_code_generator_service, get_request_repository


router = APIRouter(prefix="/status", tags=["status"])


@router.get("/{request_id}", response_model=CodeGenerationStatusModel)
async def get_status(
    request_id: str,
    request_repo: CodeRequestRepository = Depends(get_request_repository)
):
    """
    Get the status of a code generation request.
    """
    # Get request status
    status = request_repo.get_status(request_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    # TODO: Calculate progress and ETA
    # For now, just return the status
    
    return CodeGenerationStatusModel(
        request_id=request_id,
        status=status.value
    )