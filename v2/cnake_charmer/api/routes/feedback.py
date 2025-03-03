# api/routes/feedback.py
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException

from core.enums import LanguageType
from services.feedback_service import FeedbackService
from repositories.code_repo import GeneratedCodeRepository
from api.dependencies import get_feedback_service, get_code_repository


router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    request_id: str
    language: str
    feedback_type: str
    message: str
    source: str = "user"


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    improved_code: Optional[str] = None
    message: Optional[str] = None


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_service: FeedbackService = Depends(get_feedback_service),
    code_repo: GeneratedCodeRepository = Depends(get_code_repository)
):
    """
    Submit feedback on generated code and get improved code.
    """
    # Get original code
    try:
        language = LanguageType(request.language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")
    
    code = code_repo.get_latest_code(request.request_id, language)
    
    if not code:
        raise HTTPException(status_code=404, detail=f"No {language} code found for request {request.request_id}")
    
    # Process feedback
    feedback = {
        "type": request.feedback_type,
        "message": request.message,
        "source": request.source
    }
    
    try:
        improved_code = await feedback_service.process_feedback(
            code,
            language,
            feedback,
            []  # No history for now
        )
        
        # Save the improved code as a new version
        code_repo.save_code(request.request_id, language, improved_code)
        
        return FeedbackResponse(
            success=True,
            improved_code=improved_code
        )
    
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message=str(e)
        )