# analyzers/performance_analyzer.py
import time
import tempfile
import os
import subprocess
import logging
import statistics
from typing import Dict, List, Optional, Any

from core.models import AnalysisResult, Suggestion
from core.enums import LanguageType, AnalysisType
from builders.base import BaseBuilder


class PerformanceAnalyzer:
    """Analyzer for code performance."""
    
    def __init__(self, builders: Dict[LanguageType, BaseBuilder], runs: int = 5):
        """
        Initialize performance analyzer.
        
        Args:
            builders: Dictionary mapping languages to their respective builders
            runs: Number of times to run each test
        """
        self.builders = builders
        self.runs = runs
        self.logger = logging.getLogger(__name__)
    
    def analyzer_type(self) -> AnalysisType:
        """Get the analyzer type."""
        return AnalysisType.PERFORMANCE
    
    def supported_languages(self) -> List[LanguageType]:
        """Get supported languages."""
        return list(self.builders.keys())
    
    def analyze(self, code: str, language: LanguageType, options: Dict) -> AnalysisResult:
        """
        Analyze code performance.
        
        Args:
            code: Code to analyze
            language: Language of the code
            options: Analysis options
            
        Returns:
            Analysis result
        """
        if language not in self.supported_languages():
            self.logger.warning(f"Language {language} not supported for performance analysis")
            return AnalysisResult(
                score=0.0,
                details={
                    "error": f"Language {language} not supported for performance analysis"
                }
            )
        
        # Build the code
        builder = self.builders[language]
        build_result = builder.build(code, language, {})
        
        if not build_result.success:
            self.logger.warning(f"Failed to build {language} code: {build_result.error}")
            return AnalysisResult(
                score=0.0,
                details={
                    "error": f"Failed to build code: {build_result.error}"
                }
            )
        
        try:
            # Create test inputs
            test_inputs = self._generate_test_inputs()
            
            # Run performance tests
            results = []
            
            for test_input in test_inputs:
                test_name = test_input.pop("name")
                
                # Run multiple times and calculate average and standard deviation
                execution_times = []
                
                for _ in range(self.runs):
                    execution_result = builder.run(build_result.artifact_path, test_input)
                    
                    if not execution_result.success:
                        self.logger.warning(f"Failed to run test {test_name}: {execution_result.error}")
                        continue
                    
                    execution_times.append(execution_result.execution_time)
                
                if not execution_times:
                    continue
                
                # Calculate statistics
                avg_time = statistics.mean(execution_times)
                std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
                
                results.append({
                    "test_name": test_name,
                    "avg_execution_time": avg_time,
                    "std_dev": std_dev,
                    "runs": len(execution_times)
                })
            
            # Calculate overall score based on performance
            if not results:
                return AnalysisResult(
                    score=0.0,
                    details={
                        "error": "No successful test runs"
                    }
                )
            
            # Normalize results (lower is better for execution time)
            # For now, we'll use a simple scoring scheme based on execution time
            # A better approach would be to compare against a baseline
            score = 1.0 - min(1.0, sum(r["avg_execution_time"] for r in results) / 5.0)
            
            # Generate suggestions
            suggestions = []
            
            # Check for high standard deviation
            for result in results:
                if result["std_dev"] > 0.1 * result["avg_execution_time"]:
                    suggestions.append(Suggestion(
                        line=1,
                        message=f"High variability in execution time for test {result['test_name']}",
                        severity="warning"
                    ))
            
            # Language-specific suggestions
            if language == LanguageType.CYTHON and score < 0.8:
                suggestions.append(Suggestion(
                    line=1,
                    message="Consider using C types and memoryviews for better performance",
                    severity="suggestion"
                ))
                
                suggestions.append(Suggestion(
                    line=1,
                    message="Add Cython directives like boundscheck=False and wraparound=False",
                    severity="suggestion"
                ))
            
            return AnalysisResult(
                score=score,
                details={
                    "test_results": results
                },
                suggestions=suggestions
            )
        
        finally:
            # Clean up artifacts
            builder.cleanup(build_result.artifact_path)
    
    def _generate_test_inputs(self) -> List[Dict]:
        """
        Generate test inputs for performance analysis.
        
        Returns:
            List of test inputs
        """
        return [
            {"name": "empty", "input_data": []},
            {"name": "small", "input_data": list(range(10))},
            {"name": "medium", "input_data": list(range(100))},
            {"name": "large", "input_data": list(range(1000))},
            {"name": "very_large", "input_data": list(range(10000))}
        ]
    
    def get_score(self) -> float:
        """
        Get the score from the last analysis.
        
        Returns:
            Last analysis score or 0.0 if no analysis has been performed
        """
        # This method would typically return the last score,
        # but since we're not storing state, it's not useful here
        return 0.0
    
    def get_improvement_suggestions(self) -> List[Suggestion]:
        """
        Get improvement suggestions from the last analysis.
        
        Returns:
            List of suggestions
        """
        # This method would typically return the last suggestions,
        # but since we're not storing state, it's not useful here
        return []