# worker/tasks/analyze_tasks.py
import logging
from typing import Dict, List, Optional, Any

from worker.celery import app
from core.enums import RequestStatus, LanguageType, AnalysisType
from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository
from builders.python_builder import PythonBuilder
from builders.cython_builder import CythonBuilder
from analyzers.static_analyzer import StaticCodeAnalyzer
from equivalency.checker import SimpleEquivalencyChecker


logger = logging.getLogger(__name__)


@app.task(name='analyze.code')
def analyze_code_task(request_id: str) -> Dict:
    """
    Task to analyze generated code.
    
    Args:
        request_id: Request ID
        
    Returns:
        Dictionary with analysis status
    """
    logger.info(f"Analyzing code for request: {request_id}")
    
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
        
        # Initialize analyzers
        analyzers = {
            AnalysisType.STATIC: StaticCodeAnalyzer()
        }
        
        # Analyze each implementation
        analysis_results = {}
        
        for lang, code in generated_code.items():
            lang_results = {}
            
            for analysis_type, analyzer in analyzers.items():
                if lang not in analyzer.supported_languages():
                    logger.warning(f"Language {lang} not supported by {analysis_type} analyzer")
                    continue
                
                logger.info(f"Running {analysis_type} analysis on {lang} implementation")
                
                result = analyzer.analyze(code, lang, {})
                
                lang_results[analysis_type] = {
                    'score': result.score,
                    'details': result.details,
                    'suggestions': [
                        {
                            'line': s.line,
                            'message': s.message,
                            'severity': s.severity,
                            'code': s.code,
                            'replacement': s.replacement
                        }
                        for s in result.suggestions
                    ]
                }
            
            if lang_results:
                analysis_results[lang] = lang_results
        
        # TODO: Store analysis results in database
        
        return {
            'status': 'success',
            'analysis_results': analysis_results
        }
    
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@app.task(name='analyze.equivalence')
def check_equivalence_task(request_id: str) -> Dict:
    """
    Task to check equivalence between implementations.
    
    Args:
        request_id: Request ID
        
    Returns:
        Dictionary with equivalence check status
    """
    logger.info(f"Checking equivalence for request: {request_id}")
    
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
        
        if len(generated_code) < 2:
            return {
                'status': 'warning',
                'message': f"At least two implementations are required for equivalence check"
            }
        
        # Initialize builders
        builders = {
            LanguageType.PYTHON: PythonBuilder(),
            LanguageType.CYTHON: CythonBuilder()
        }
        
        # Initialize equivalence checker
        checker = SimpleEquivalencyChecker(builders)
        
        # Generate test cases using the first implementation
        first_lang = next(iter(generated_code.keys()))
        test_cases = checker.generate_test_cases(
            generated_code[first_lang],
            first_lang,
            count=5
        )
        
        if not test_cases:
            return {
                'status': 'warning',
                'message': f"No test cases could be generated for equivalence check"
            }
        
        # Check equivalence
        result = checker.check_equivalence(generated_code, test_cases)
        
        # TODO: Store equivalence results in database
        
        return {
            'status': 'success',
            'equivalent': result.equivalent,
            'test_cases': result.test_cases,
            'differences': result.differences
        }
    
    except Exception as e:
        logger.error(f"Error checking equivalence: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }