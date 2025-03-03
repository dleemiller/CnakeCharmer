# api/dependencies.py (updated)
from typing import Generator

from core.enums import LanguageType

from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository
from repositories.build_repo import BuildResultRepository
from repositories.analysis_repo import AnalysisResultRepository
from repositories.equivalency_repo import EquivalencyResultRepository

from services.code_generator import CodeGeneratorService
from services.feedback_service import FeedbackService

from builders.python_builder import PythonBuilder
from builders.cython_builder import CythonBuilder

from analyzers.static_analyzer import StaticCodeAnalyzer

from equivalency.checker import SimpleEquivalencyChecker


# Repository instances
_request_repo = None
_code_repo = None
_build_repo = None
_analysis_repo = None
_equivalency_repo = None

# Service instances
_code_generator = None
_feedback_service = None

# Builder instances
_python_builder = None
_cython_builder = None

# Analyzer instances
_static_analyzer = None

# Checker instances
_equivalency_checker = None


def get_request_repository() -> CodeRequestRepository:
    """Get the code request repository."""
    global _request_repo
    if _request_repo is None:
        _request_repo = CodeRequestRepository()
    return _request_repo


def get_code_repository() -> GeneratedCodeRepository:
    """Get the generated code repository."""
    global _code_repo
    if _code_repo is None:
        _code_repo = GeneratedCodeRepository()
    return _code_repo


def get_build_repository() -> BuildResultRepository:
    """Get the build result repository."""
    global _build_repo
    if _build_repo is None:
        _build_repo = BuildResultRepository()
    return _build_repo


def get_analysis_repository() -> AnalysisResultRepository:
    """Get the analysis result repository."""
    global _analysis_repo
    if _analysis_repo is None:
        _analysis_repo = AnalysisResultRepository()
    return _analysis_repo


def get_equivalency_repository() -> EquivalencyResultRepository:
    """Get the equivalency result repository."""
    global _equivalency_repo
    if _equivalency_repo is None:
        _equivalency_repo = EquivalencyResultRepository()
    return _equivalency_repo


def get_python_builder() -> PythonBuilder:
    """Get the Python builder."""
    global _python_builder
    if _python_builder is None:
        _python_builder = PythonBuilder()
    return _python_builder


def get_cython_builder() -> CythonBuilder:
    """Get the Cython builder."""
    global _cython_builder
    if _cython_builder is None:
        _cython_builder = CythonBuilder()
    return _cython_builder


def get_static_analyzer() -> StaticCodeAnalyzer:
    """Get the static code analyzer."""
    global _static_analyzer
    if _static_analyzer is None:
        _static_analyzer = StaticCodeAnalyzer()
    return _static_analyzer


def get_equivalency_checker() -> SimpleEquivalencyChecker:
    """Get the equivalency checker."""
    global _equivalency_checker
    if _equivalency_checker is None:
        builders = {
            LanguageType.PYTHON: get_python_builder(),
            LanguageType.CYTHON: get_cython_builder()
        }
        _equivalency_checker = SimpleEquivalencyChecker(builders)
    return _equivalency_checker


def get_code_generator_service() -> CodeGeneratorService:
    """Get the code generator service."""
    global _code_generator
    if _code_generator is None:
        request_repo = get_request_repository()
        code_repo = get_code_repository()
        _code_generator = CodeGeneratorService(request_repo, code_repo)
    return _code_generator


def get_feedback_service() -> FeedbackService:
    """Get the feedback service."""
    global _feedback_service
    if _feedback_service is None:
        request_repo = get_request_repository()
        code_repo = get_code_repository()
        _feedback_service = FeedbackService(request_repo, code_repo)
    return _feedback_service