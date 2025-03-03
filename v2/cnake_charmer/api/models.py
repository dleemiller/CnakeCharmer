# api/models.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from core.enums import LanguageType, AnalysisType, RequestStatus


class CodeGenerationRequestModel(BaseModel):
    """API model for code generation requests."""
    prompt: str = Field(..., description="Natural language description of the code to generate")
    target_languages: List[str] = Field(..., description="Languages to generate code in")
    source_language: Optional[str] = Field(None, description="Source language if translating existing code")
    source_code: Optional[str] = Field(None, description="Source code if translating existing code")
    equivalency_check: bool = Field(True, description="Whether to check equivalence between implementations")
    optimization_level: int = Field(1, description="Level of optimization (0-3)")
    analysis_types: Optional[List[str]] = Field(None, description="Types of analysis to perform")
    max_attempts: int = Field(3, description="Maximum number of retry attempts")


class CodeGenerationResponseModel(BaseModel):
    """API model for code generation responses."""
    request_id: str = Field(..., description="Unique identifier for the request")


class CodeGenerationStatusModel(BaseModel):
    """API model for code generation status."""
    request_id: str = Field(..., description="Unique identifier for the request")
    status: str = Field(..., description="Current status of the request")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    eta: Optional[float] = Field(None, description="Estimated time remaining in seconds")


class CodeGenerationResultModel(BaseModel):
    """API model for code generation results."""
    request_id: str = Field(..., description="Unique identifier for the request")
    status: str = Field(..., description="Current status of the request")
    generated_code: Dict[str, str] = Field(..., description="Generated code for each language")
    build_results: Optional[Dict[str, Any]] = Field(None, description="Build results for each language")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Analysis results for each language")
    equivalency_result: Optional[Any] = Field(None, description="Result of equivalency checking")
    error_messages: Optional[List[str]] = Field(None, description="Error messages if any")