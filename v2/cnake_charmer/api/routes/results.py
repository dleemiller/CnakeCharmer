# api/routes/results.py
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException

from core.enums import RequestStatus
from services.code_generator import CodeGeneratorService
from api.models import CodeGenerationResultModel
from api.dependencies import get_code_generator_service


router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{request_id}", response_model=CodeGenerationResultModel)
async def get_results(
    request_id: str,
    code_generator: CodeGeneratorService = Depends(get_code_generator_service)
):
    """
    Get the results of a code generation request.
    """
    # Get the results
    result = await code_generator.get_result(request_id)
    
    if result.status == RequestStatus.FAILED and "Request not found" in result.error_messages:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    # Convert domain model to API model
    return CodeGenerationResultModel(
        request_id=result.request_id,
        status=result.status.value,
        generated_code={lang.value: code for lang, code in result.generated_code.items()},
        build_results={lang.value: build_result for lang, build_result in result.build_results.items()} 
            if result.build_results else None,
        analysis_results={lang.value: {analysis.value: result 
                                      for analysis, result in analysis_results.items()}
                         for lang, analysis_results in result.analysis_results.items()} 
            if result.analysis_results else None,
        equivalency_result=result.equivalency_result,
        error_messages=result.error_messages
    )