# worker/tasks/build_tasks.py
import logging
from typing import Dict, List, Optional, Any

from worker.celery import app
from core.enums import RequestStatus, LanguageType
from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository
from builders.python_builder import PythonBuilder
from builders.cython_builder import CythonBuilder


logger = logging.getLogger(__name__)


@app.task(name='build.code')
def build_code_task(request_id: str) -> Dict:
    """
    Task to build generated code.
    
    Args:
        request_id: Request ID
        
    Returns:
        Dictionary with build status
    """
    logger.info(f"Building code for request: {request_id}")
    
    try:
        # Get repositories
        request_repo = CodeRequestRepository()
        code_repo = GeneratedCodeRepository()
        
        # Get request
        request = request_repo.get_by_id(request_id)
        if not request:
            return {
                'status': 'error',
                'error': f"Request {request_id} not found"
            }
        
        # Get generated code
        generated_code = code_repo.get_all_latest_code(request_id)
        if not generated_code:
            return {
                'status': 'error',
                'error': f"No generated code found for request {request_id}"
            }
        
        # Initialize builders
        builders = {
            LanguageType.PYTHON: PythonBuilder(),
            LanguageType.CYTHON: CythonBuilder()
        }
        
        # Build each implementation
        build_results = {}
        
        for lang, code in generated_code.items():
            if lang not in builders:
                logger.warning(f"No builder available for language {lang}")
                continue
            
            logger.info(f"Building {lang} implementation")
            
            build_result = builders[lang].build(code, lang, {})
            
            build_results[lang] = {
                'success': build_result.success,
                'output': build_result.output,
                'error': build_result.error,
                'artifact_path': build_result.artifact_path,
                'build_time': build_result.build_time
            }
        
        # TODO: Store build results in database
        
        # Check if we should run equivalency check
        if request.equivalency_check and len(build_results) >= 2:
            from worker.tasks.analyze_tasks import check_equivalence_task
            check_equivalence_task.delay(request_id)
        
        return {
            'status': 'success',
            'build_results': build_results
        }
    
    except Exception as e:
        logger.error(f"Error building code: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }