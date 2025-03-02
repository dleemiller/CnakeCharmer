"""
Main analyzer module for Cython code quality assessment.
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional

from .parsing.static_parser import analyze_static_features
from .parsing.html_parser import parse_annotation_html
from .metrics import calculate_optimization_scores
from .common import CythonAnalysisResult

logger = logging.getLogger("cython_analyzer")


class CythonAnalyzer:
    """
    Analyzes Cython code quality based on static analysis and
    annotated HTML output from Cython compilation.
    """

    def __init__(self, ephemeral_runner=None, temp_dir=None):
        """
        Initialize the analyzer.

        Args:
            ephemeral_runner: Instance of ephemeral_runner for compilation
            temp_dir: Optional directory to use for temporary files
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.runner = ephemeral_runner
        self.last_metrics = {}
        logger.info(f"Initialized CythonAnalyzer with temp_dir: {self.temp_dir}")

    def analyze_code(self, code_str) -> Dict[str, Any]:
        """
        Analyze Cython code by compiling it with annotations and parsing the HTML output.

        Args:
            code_str: String containing the Cython code to analyze

        Returns:
            dict: Analysis metrics
        """
        # Step 1: Static code analysis
        metrics = analyze_static_features(code_str)

        # Step 2: Get HTML annotation via ephemeral_runner (if available)
        if self.runner:
            html_content = self._get_html_annotation(code_str)
            if html_content:
                # Step 3: Parse HTML annotation
                html_metrics = parse_annotation_html(html_content)
                # Step 4: Merge metrics
                metrics.update(html_metrics)

        # Step 5: Calculate final scores
        metrics = calculate_optimization_scores(metrics)

        # Save a copy of the metrics for later reference
        self.last_metrics = metrics

        return metrics

    def analyze_code_structured(self, code_str) -> CythonAnalysisResult:
        """
        Analyze Cython code and return a structured result.

        Args:
            code_str: String containing the Cython code to analyze

        Returns:
            CythonAnalysisResult: Structured analysis results
        """
        from .reporting import metrics_to_analysis_result

        # Get standard dictionary-based metrics
        metrics = self.analyze_code(code_str)

        # Convert to structured result
        return metrics_to_analysis_result(metrics)

    def _get_html_annotation(self, code_str):
        """
        Get HTML annotation for the code using ephemeral_runner.

        Args:
            code_str: Cython code to analyze

        Returns:
            str or None: HTML content if successful, None otherwise
        """
        try:
            # Use the runner with annotation enabled
            result = self.runner.build_and_run(code_str, annotate=True)

            if result.success and result.html_annotation:
                return result.html_annotation

            if not result.success:
                logger.error(f"Compilation failed: {result.error_message}")
            elif not result.html_annotation:
                logger.error(
                    "Compilation succeeded but no HTML annotation was generated"
                )

            return None

        except Exception as e:
            logger.error(f"Error getting HTML annotation: {str(e)}")
            return None

    def get_last_metrics(self):
        """Get the results of the last analysis."""
        return self.last_metrics
