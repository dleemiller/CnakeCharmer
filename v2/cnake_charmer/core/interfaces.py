# core/interfaces.py
from typing import Dict, List, Optional, Protocol, TypeVar, Generic, Any

from core.models import (
    BuildResult, ExecutionResult, AnalysisResult, Suggestion, 
    EquivalencyResult, CodeGenerationRequest, CodeGenerationResult
)
from core.enums import LanguageType, AnalysisType


class CodeBuilder(Protocol):
    """Protocol for code builders."""
    
    def build(self, code: str, language: LanguageType, options: Dict) -> BuildResult:
        """Build the provided code in the specified language."""
        ...
    
    def run(self, build_artifact: str, inputs: Dict) -> ExecutionResult:
        """Run a built artifact with the given inputs."""
        ...
    
    def supported_languages(self) -> List[LanguageType]:
        """Return list of supported languages."""
        ...
    
    def cleanup(self, artifact_path: str) -> None:
        """Clean up any temporary files or resources."""
        ...


class CodeAnalyzer(Protocol):
    """Protocol for code analyzers."""
    
    def analyze(self, code: str, language: LanguageType, options: Dict) -> AnalysisResult:
        """Analyze the provided code."""
        ...
    
    def get_score(self) -> float:
        """Return a normalized score (0.0-1.0) for the last analysis."""
        ...
    
    def get_improvement_suggestions(self) -> List[Suggestion]:
        """Return suggestions for code improvement."""
        ...
    
    def analyzer_type(self) -> AnalysisType:
        """Return the type of analyzer."""
        ...
    
    def supported_languages(self) -> List[LanguageType]:
        """Return list of supported languages."""
        ...


class EquivalencyChecker(Protocol):
    """Protocol for equivalency checkers."""
    
    def check_equivalence(
        self, 
        implementations: Dict[LanguageType, str], 
        test_cases: List[Dict]
    ) -> EquivalencyResult:
        """Check if implementations are functionally equivalent."""
        ...
    
    def generate_test_cases(
        self, 
        code: str, 
        language: LanguageType, 
        count: int
    ) -> List[Dict]:
        """Generate test cases for equivalency checking."""
        ...
    
    def supported_language_pairs(self) -> List[tuple[LanguageType, LanguageType]]:
        """Return list of supported language pairs for equivalency checking."""
        ...


class FeedbackProcessor(Protocol):
    """Protocol for feedback processors."""
    
    def process_feedback(
        self, 
        code: str,
        language: LanguageType,
        feedback: Dict,
        generation_history: List[Dict]
    ) -> str:
        """Process feedback and generate improved code."""
        ...
    
    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if we should retry after an error."""
        ...
    
    def generate_improvement_prompt(
        self, 
        original_code: str,
        analysis_results: Dict,
        error_messages: List[str]
    ) -> str:
        """Generate a prompt for improving code based on feedback."""
        ...


T = TypeVar('T')

class Repository(Generic[T], Protocol):
    """Base repository protocol."""
    
    def create(self, entity: T) -> str:
        """Create a new entity and return its ID."""
        ...
    
    def get_by_id(self, id: str) -> Optional[T]:
        """Get an entity by its ID."""
        ...
    
    def update(self, entity: T) -> bool:
        """Update an existing entity."""
        ...
    
    def delete(self, id: str) -> bool:
        """Delete an entity by its ID."""
        ...