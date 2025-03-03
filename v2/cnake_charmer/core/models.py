# core/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.enums import LanguageType, AnalysisType, RequestStatus


@dataclass
class CodeGenerationRequest:
    """Request to generate code."""
    prompt: str
    target_languages: List[LanguageType]
    source_language: Optional[LanguageType] = None
    source_code: Optional[str] = None
    equivalency_check: bool = True
    optimization_level: int = 1
    analysis_types: List[AnalysisType] = field(default_factory=list)
    max_attempts: int = 3

    def __post_init__(self):
        if not self.analysis_types:
            self.analysis_types = [AnalysisType.STATIC, AnalysisType.PERFORMANCE]


@dataclass
class BuildResult:
    """Result of building code."""
    success: bool
    output: str
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    build_time: float = 0.0


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Suggestion:
    """Suggestion for code improvement."""
    line: int
    message: str
    severity: str
    code: Optional[str] = None
    replacement: Optional[str] = None


@dataclass
class EquivalencyResult:
    """Result of equivalency checking."""
    equivalent: bool
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    differences: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None


@dataclass
class CodeGenerationResult:
    """Result of code generation."""
    request_id: str
    status: RequestStatus
    generated_code: Dict[LanguageType, str] = field(default_factory=dict)
    build_results: Dict[LanguageType, BuildResult] = field(default_factory=dict)
    analysis_results: Dict[LanguageType, Dict[AnalysisType, AnalysisResult]] = field(default_factory=dict)
    equivalency_result: Optional[EquivalencyResult] = None
    error_messages: List[str] = field(default_factory=list)
    feedback_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)