# api/routes/analyze.py
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException

from core.enums import LanguageType, AnalysisType
from services.analysis_service import AnalysisService
from analyzers.static_analyzer import StaticCodeAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from api.dependencies import get_static_analyzer, get_performance_analyzer


router = APIRouter(prefix="/analyze", tags=["analyze"])


class DirectAnalysisRequest(BaseModel):
    """Request model for direct code analysis."""
    code: str
    language: str
    analysis_types: List[str] = ["static"]
    options: Dict = {}


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    results: Optional[Dict] = None
    message: Optional[str] = None


@router.post("/", response_model=AnalysisResponse)
async def analyze_code(
    request: DirectAnalysisRequest,
    static_analyzer: StaticCodeAnalyzer = Depends(get_static_analyzer),
    performance_analyzer: PerformanceAnalyzer = Depends(get_performance_analyzer)
):
    """
    Analyze code directly without storing it.
    """
    try:
        language = LanguageType(request.language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")
    
    try:
        analysis_types = [AnalysisType(a_type) for a_type in request.analysis_types]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported analysis type in {request.analysis_types}")
    
    results = {}
    
    try:
        # Perform requested analyses
        for analysis_type in analysis_types:
            if analysis_type == AnalysisType.STATIC and language in static_analyzer.supported_languages():
                result = static_analyzer.analyze(request.code, language, request.options)
                results["static"] = {
                    "score": result.score,
                    "details": result.details,
                    "suggestions": [
                        {
                            "line": s.line,
                            "message": s.message,
                            "severity": s.severity,
                            "code": s.code,
                            "replacement": s.replacement
                        }
                        for s in result.suggestions
                    ]
                }
            
            elif analysis_type == AnalysisType.PERFORMANCE and language in performance_analyzer.supported_languages():
                result = performance_analyzer.analyze(request.code, language, request.options)
                results["performance"] = {
                    "score": result.score,
                    "details": result.details,
                    "suggestions": [
                        {
                            "line": s.line,
                            "message": s.message,
                            "severity": s.severity,
                            "code": s.code,
                            "replacement": s.replacement
                        }
                        for s in result.suggestions
                    ]
                }
        
        return AnalysisResponse(
            success=bool(results),
            results=results,
            message=None if results else "No analysis performed"
        )
    
    except Exception as e:
        return AnalysisResponse(
            success=False,
            message=str(e)
        )