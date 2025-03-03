# analyzers/static_analyzer.py
import re
import logging
from typing import Dict, List, Optional, Any

from core.models import AnalysisResult, Suggestion
from core.enums import LanguageType, AnalysisType


class StaticCodeAnalyzer:
    """Simple static code analyzer."""
    
    def __init__(self):
        """Initialize static code analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyzer_type(self) -> AnalysisType:
        """Get the analyzer type."""
        return AnalysisType.STATIC
    
    def supported_languages(self) -> List[LanguageType]:
        """Get supported languages."""
        return [
            LanguageType.PYTHON,
            LanguageType.CYTHON
        ]
    
    def analyze(self, code: str, language: LanguageType, options: Dict) -> AnalysisResult:
        """
        Analyze code.
        
        Args:
            code: Code to analyze
            language: Language of the code
            options: Analysis options
            
        Returns:
            Analysis result
        """
        if language not in self.supported_languages():
            self.logger.warning(f"Language {language} not supported for static analysis")
            return AnalysisResult(
                score=0.0,
                details={
                    "error": f"Language {language} not supported for static analysis"
                }
            )
        
        # Split code into lines
        lines = code.splitlines()
        
        # Initialize metrics
        metrics = {
            "line_count": len(lines),
            "empty_lines": 0,
            "comment_lines": 0,
            "docstring_lines": 0,
            "code_lines": 0,
            "function_count": 0,
            "class_count": 0,
            "complexity": 0.0,
            "issues": []
        }
        
        # Analysis patterns
        patterns = {
            "empty_line": re.compile(r'^\s*$'),
            "comment": re.compile(r'^\s*#'),
            "function_def": re.compile(r'^\s*def\s+(\w+)\s*\('),
            "class_def": re.compile(r'^\s*class\s+(\w+)'),
            "complexity_factor": re.compile(r'(if|for|while|except|with|def|class|and|or|elif|else)'),
            "long_line": re.compile(r'^.{80,}$'),
            "todo": re.compile(r'#\s*TODO'),
            "docstring_start": re.compile(r'^\s*[\'"]\'\'\'|^\s*["\']{3}'),
            "docstring_end": re.compile(r'.*[\'"]\'\'\'|.*["\']{3}'),
        }
        
        if language == LanguageType.CYTHON:
            # Add Cython-specific patterns
            patterns.update({
                "cython_directive": re.compile(r'^\s*#\s*cython:'),
                "c_type_decl": re.compile(r'^\s*cdef\s+'),
                "gil_annotation": re.compile(r'nogil'),
                "memoryview": re.compile(r'[:,]\s*\w+[:]\s*'),
            })
            
            # Add Cython-specific metrics
            metrics.update({
                "cython_directives": 0,
                "c_type_declarations": 0,
                "gil_annotations": 0,
                "memoryview_usage": 0,
            })
        
        # Track if we're in a docstring
        in_docstring = False
        
        # Analyze each line
        for i, line in enumerate(lines):
            # Check if we're in a docstring
            if not in_docstring and patterns["docstring_start"].match(line):
                in_docstring = True
                metrics["docstring_lines"] += 1
                
                # If the docstring ends on the same line
                if patterns["docstring_end"].match(line) and line.count('"""') >= 2 or line.count("'''") >= 2:
                    in_docstring = False
                continue
            
            if in_docstring:
                metrics["docstring_lines"] += 1
                if patterns["docstring_end"].match(line):
                    in_docstring = False
                continue
            
            # Check for empty lines
            if patterns["empty_line"].match(line):
                metrics["empty_lines"] += 1
                continue
            
            # Check for comments
            if patterns["comment"].match(line):
                metrics["comment_lines"] += 1
                
                # Check for TODOs
                if patterns["todo"].search(line):
                    metrics["issues"].append({
                        "line": i + 1,
                        "message": "TODO comment found",
                        "severity": "info"
                    })
                continue
            
            # Count code lines
            metrics["code_lines"] += 1
            
            # Check for function definitions
            if patterns["function_def"].match(line):
                metrics["function_count"] += 1
            
            # Check for class definitions
            if patterns["class_def"].match(line):
                metrics["class_count"] += 1
            
            # Calculate complexity
            complexity_factors = len(patterns["complexity_factor"].findall(line))
            metrics["complexity"] += complexity_factors
            
            # Check for long lines
            if patterns["long_line"].match(line):
                metrics["issues"].append({
                    "line": i + 1,
                    "message": "Line too long (> 80 characters)",
                    "severity": "warning"
                })
            
            # Cython-specific checks
            if language == LanguageType.CYTHON:
                if patterns["cython_directive"].match(line):
                    metrics["cython_directives"] += 1
                
                if patterns["c_type_decl"].match(line):
                    metrics["c_type_declarations"] += 1
                
                if patterns["gil_annotation"].search(line):
                    metrics["gil_annotations"] += 1
                
                if patterns["memoryview"].search(line):
                    metrics["memoryview_usage"] += 1
        
        # Calculate documentation ratio
        doc_lines = metrics["comment_lines"] + metrics["docstring_lines"]
        metrics["documentation_ratio"] = doc_lines / max(1, metrics["line_count"])
        
        # Calculate complexity per line
        metrics["complexity_per_line"] = metrics["complexity"] / max(1, metrics["code_lines"])
        
        # Generate suggestions
        suggestions = []
        
        for issue in metrics["issues"]:
            suggestions.append(Suggestion(
                line=issue["line"],
                message=issue["message"],
                severity=issue["severity"]
            ))
        
        # Add language-specific suggestions
        if language == LanguageType.CYTHON and metrics["cython_directives"] == 0:
            suggestions.append(Suggestion(
                line=1,
                message="No Cython directives found. Consider adding '# cython: boundscheck=False' for optimization",
                severity="suggestion"
            ))
        
        if language == LanguageType.CYTHON and metrics["c_type_declarations"] == 0:
            suggestions.append(Suggestion(
                line=1,
                message="No C type declarations found. Consider using 'cdef' for variables to improve performance",
                severity="suggestion"
            ))
        
        # Calculate overall score
        score = self._calculate_score(metrics, language)
        
        return AnalysisResult(
            score=score,
            details=metrics,
            suggestions=suggestions
        )
    
    def _calculate_score(self, metrics: Dict, language: LanguageType) -> float:
        """
        Calculate an overall score for the code quality.
        
        Args:
            metrics: Code metrics
            language: Code language
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Base score components
        components = {
            "docs_score": min(1.0, metrics["documentation_ratio"] / 0.3),  # Aim for ~30% documentation
            "complexity_score": max(0.0, 1.0 - metrics["complexity_per_line"] / 2.0),  # Penalize complexity
            "issues_score": max(0.0, 1.0 - len(metrics["issues"]) / max(1, metrics["line_count"]) * 5)  # Penalize issues
        }
        
        # Language-specific scoring
        if language == LanguageType.CYTHON:
            # Add Cython optimization score
            optimization_score = 0.0
            
            if metrics["cython_directives"] > 0:
                optimization_score += 0.25
            
            if metrics["c_type_declarations"] > 0:
                c_type_ratio = metrics["c_type_declarations"] / max(1, metrics["function_count"] * 3)
                optimization_score += min(0.25, c_type_ratio)
            
            if metrics["gil_annotations"] > 0:
                optimization_score += 0.25
            
            if metrics["memoryview_usage"] > 0:
                optimization_score += 0.25
            
            components["optimization_score"] = optimization_score
        
        # Calculate weighted average
        if language == LanguageType.CYTHON:
            score = (
                components["docs_score"] * 0.2 +
                components["complexity_score"] * 0.2 +
                components["issues_score"] * 0.2 +
                components["optimization_score"] * 0.4  # Weight optimization higher for Cython
            )
        else:
            score = (
                components["docs_score"] * 0.3 +
                components["complexity_score"] * 0.4 +
                components["issues_score"] * 0.3
            )
        
        return min(1.0, max(0.0, score))
    
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