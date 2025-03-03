# tests/unit/test_static_analyzer.py
"""
Unit tests for the static code analyzer.
"""
import pytest

from core.enums import LanguageType
from analyzers.static_analyzer import StaticCodeAnalyzer


def test_static_analyzer_python():
    """Test static analyzer with Python code."""
    analyzer = StaticCodeAnalyzer()
    
    # Test empty code
    result = analyzer.analyze("", LanguageType.PYTHON, {})
    assert result.score == 0.0
    
    # Test well-documented code
    good_code = """
def factorial(n):
    \"\"\"
    Calculate the factorial of a number.
    
    Args:
        n: The number to calculate factorial for
        
    Returns:
        The factorial of n
    \"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    """
    
    result = analyzer.analyze(good_code, LanguageType.PYTHON, {})
    assert result.score > 0.7  # Good score
    
    # Test poorly documented code
    bad_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    """
    
    result = analyzer.analyze(bad_code, LanguageType.PYTHON, {})
    assert result.score < 0.5  # Poor score


def test_static_analyzer_cython():
    """Test static analyzer with Cython code."""
    analyzer = StaticCodeAnalyzer()
    
    # Test good Cython code
    good_code = """
# cython: boundscheck=False
# cython: wraparound=False

cdef double factorial(int n) nogil:
    \"\"\"
    Calculate the factorial of a number efficiently.
    
    Args:
        n: The number to calculate factorial for
        
    Returns:
        The factorial of n
    \"\"\"
    cdef double result = 1
    cdef int i
    
    for i in range(2, n + 1):
        result *= i
        
    return result
    """
    
    result = analyzer.analyze(good_code, LanguageType.CYTHON, {})
    assert result.score > 0.8  # Very good score
    
    # Test poor Cython code (no C types)
    bad_code = """
def factorial(n):
    \"\"\"Calculate factorial.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    """
    
    result = analyzer.analyze(bad_code, LanguageType.CYTHON, {})
    assert result.score < 0.6  # Lower score due to lack of Cython features