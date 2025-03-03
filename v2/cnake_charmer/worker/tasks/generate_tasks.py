# worker/tasks/generate_tasks.py
import logging
from typing import Dict, List, Optional, Any

from worker.celery import app
from core.models import CodeGenerationRequest
from core.enums import LanguageType, AnalysisType, RequestStatus
from services.code_generator import CodeGeneratorService
from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository
from api.dependencies import get_code_generator_service


logger = logging.getLogger(__name__)


@app.task(name='generate.code')
def generate_code_task(request_data: Dict) -> Dict:
    """
    Task to generate code.
    
    Args:
        request_data: Request data dictionary
        
    Returns:
        Dictionary with request_id and status
    """
    logger.info(f"Generating code for request: {request_data.get('prompt', '')[:50]}...")
    
    try:
        # Convert dictionary to CodeGenerationRequest
        request = CodeGenerationRequest(
            prompt=request_data.get('prompt', ''),
            target_languages=[LanguageType(lang) for lang in request_data.get('target_languages', [])],
            source_language=LanguageType(request_data.get('source_language')) if request_data.get('source_language') else None,
            source_code=request_data.get('source_code'),
            equivalency_check=request_data.get('equivalency_check', True),
            optimization_level=request_data.get('optimization_level', 1),
            analysis_types=[AnalysisType(analysis) for analysis in request_data.get('analysis_types', [])] 
                if request_data.get('analysis_types') else None,
            max_attempts=request_data.get('max_attempts', 3)
        )
        
        # Get code generator service
        code_generator = get_code_generator_service()
        
        # Generate code
        request_id = code_generator.generate_code(request)
        
        # Chain build tasks if needed
        if request_data.get('build', False):
            from worker.tasks.build_tasks import build_code_task
            build_code_task.delay(request_id)
        
        return {
            'request_id': request_id,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }