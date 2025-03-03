# tests/unit/test_models.py
"""
Unit tests for the core models.
"""
import pytest

from core.models import (
    CodeGenerationRequest, 
    BuildResult, 
    AnalysisResult, 
    EquivalencyResult
)
from core.enums import LanguageType, AnalysisType, RequestStatus


def test_code_generation_request():
    """Test CodeGenerationRequest model."""
    # Test creation
    request = CodeGenerationRequest(
        prompt="Generate a function to calculate Fibonacci numbers",
        target_languages=[LanguageType.PYTHON, LanguageType.CYTHON]
    )
    
    # Check defaults
    assert request.source_language is None
    assert request.source_code is None
    assert request.equivalency_check is True
    assert request.optimization_level == 1
    assert AnalysisType.STATIC in request.analysis_types
    assert AnalysisType.PERFORMANCE in request.analysis_types
    assert request.max_attempts == 3


def test_build_result():
    """Test BuildResult model."""
    # Test successful build
    result = BuildResult(
        success=True,
        output="Build successful",
        artifact_path="/tmp/build123",
        build_time=1.5
    )
    
    assert result.success is True
    assert result.output == "Build successful"
    assert result.error is None
    assert result.artifact_path == "/tmp/build123"
    assert result.build_time == 1.5
    
    # Test failed build
    result = BuildResult(
        success=False,
        output="",
        error="Compilation error",
        build_time=0.5
    )
    
    assert result.success is False
    assert result.output == ""
    assert result.error == "Compilation error"
    assert result.artifact_path is None
    assert result.build_time == 0.5


def test_analysis_result():
    """Test AnalysisResult model."""
    result = AnalysisResult(
        score=0.85,
        details={"complexity": 5, "documentation_ratio": 0.3},
        suggestions=[
            {"line": 10, "message": "Line too long", "severity": "warning"}
        ]
    )
    
    assert result.score == 0.85
    assert result.details["complexity"] == 5
    assert result.details["documentation_ratio"] == 0.3
    assert len(result.suggestions) == 1
    assert result.suggestions[0]["line"] == 10


def test_equivalency_result():
    """Test EquivalencyResult model."""
    # Test equivalent implementations
    result = EquivalencyResult(
        equivalent=True,
        test_cases=[{"input": 5, "expected": 120}],
        differences=None,
        error=None
    )
    
    assert result.equivalent is True
    assert len(result.test_cases) == 1
    assert result.differences is None
    assert result.error is None
    
    # Test non-equivalent implementations
    result = EquivalencyResult(
        equivalent=False,
        test_cases=[{"input": 5, "expected": 120}],
        differences=[
            {"test_case_id": 0, "languages": ["python", "cython"], "values": [120, 119]}
        ],
        error=None
    )
    
    assert result.equivalent is False
    assert len(result.test_cases) == 1
    assert len(result.differences) == 1
    assert result.differences[0]["languages"] == ["python", "cython"]