# equivalency/test_generator.py
import ast
import random
import logging
import inspect
from typing import Dict, List, Optional, Any

from core.enums import LanguageType


class TestCaseGenerator:
    """Generator for test cases."""
    
    def __init__(self):
        """Initialize test case generator."""
        self.logger = logging.getLogger(__name__)
    
    def generate_test_cases(self, code: str, language: LanguageType, count: int = 5) -> List[Dict]:
        """
        Generate test cases for code.
        
        Args:
            code: Code to generate test cases for
            language: Language of the code
            count: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        if language == LanguageType.PYTHON:
            return self._generate_python_test_cases(code, count)
        elif language == LanguageType.CYTHON:
            return self._generate_cython_test_cases(code, count)
        else:
            self.logger.warning(f"Test case generation not implemented for {language}")
            return []
    
    def _generate_python_test_cases(self, code: str, count: int) -> List[Dict]:
        """
        Generate test cases for Python code.
        
        Args:
            code: Python code
            count: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        test_cases = []
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Find function definitions
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                self.logger.warning("No functions found in the code")
                return test_cases
            
            # Find the 'main' function or the first function
            main_func = next((f for f in functions if f.name == 'main'), functions[0])
            
            # Extract arguments
            args = [arg.arg for arg in main_func.args.args]
            
            # Generate test cases based on argument names and types
            for i in range(count):
                test_case = {}
                
                for arg in args:
                    # Generate values based on argument name
                    if arg in ['data', 'input_data', 'values', 'numbers', 'array', 'list']:
                        test_case[arg] = [random.randint(-100, 100) for _ in range(random.randint(5, 20))]
                    elif arg in ['n', 'num', 'count', 'size', 'length']:
                        test_case[arg] = random.randint(1, 100)
                    elif arg in ['x', 'y', 'value', 'number']:
                        test_case[arg] = random.randint(-100, 100)
                    elif arg in ['text', 'string', 'str', 'name']:
                        test_case[arg] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 15)))
                    elif arg in ['flag', 'enabled', 'active']:
                        test_case[arg] = random.choice([True, False])
                    else:
                        # Fallback to integers
                        test_case[arg] = random.randint(1, 100)
                
                test_cases.append(test_case)
            
            return test_cases
        
        except Exception as e:
            self.logger.error(f"Error generating test cases: {e}")
            return []
    
    def _generate_cython_test_cases(self, code: str, count: int) -> List[Dict]:
        """
        Generate test cases for Cython code.
        
        Args:
            code: Cython code
            count: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        # For Cython, we use the same approach as Python for now
        return self._generate_python_test_cases(code, count)