"""
Cython Analyzer Module

This module provides tools for analyzing Cython code quality through HTML annotation,
parsing the results, and extracting performance metrics.
"""

import os
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional, Union
from bs4 import BeautifulSoup
import sys
import re

# Configure logger
logger = logging.getLogger("cython_analyzer")

class CythonAnnotationAnalyzer:
    """
    Analyzes Cython code quality based on the annotated HTML output from Cython compilation.
    
    This analyzer compiles Cython code with annotation enabled, parses the HTML output,
    and extracts metrics about Python interaction, C-level operations, and optimization
    opportunities.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            temp_dir: Optional directory to use for temporary files. If None, creates one.
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        logger.info(f"Initialized CythonAnnotationAnalyzer with temp_dir: {self.temp_dir}")
        self.last_metrics = {}
    
    def analyze_code(self, code_str: str) -> Dict[str, Any]:
        """
        Analyze Cython code by compiling it with annotations and parsing the HTML output.
        
        Args:
            code_str: String containing the Cython code to analyze
            
        Returns:
            dict: Analysis metrics including optimization scores and line categorization
        """
        # Step 1: Pre-analyze the code itself for static metrics
        static_metrics = self._analyze_code_static(code_str)
        
        # Step 2: Write code to a temporary file
        pyx_path = os.path.join(self.temp_dir, "analyze_code.pyx")
        with open(pyx_path, "w") as f:
            f.write(code_str)
        
        logger.info(f"Wrote {len(code_str)} bytes of Cython code to {pyx_path}")
        
        # Step 3: Compile with annotation
        html_path = self._compile_with_annotation(pyx_path)
        if not html_path or not os.path.exists(html_path):
            logger.error("Failed to generate HTML annotation")
            # Even if annotation fails, return static metrics
            return {**static_metrics, "error": "Failed to generate HTML annotation"}
        
        # Step 4: Parse the HTML annotation
        html_metrics = self._parse_annotation_html(html_path)
        
        # Step 5: Combine static and HTML metrics
        metrics = {**static_metrics, **html_metrics}
        
        # Step 6: Calculate derived metrics
        if "total_lines" in metrics and metrics["total_lines"] > 0:
            metrics["c_ratio"] = metrics["c_lines"] / metrics["total_lines"]
            metrics["python_ratio"] = metrics["python_lines"] / metrics["total_lines"]
        else:
            metrics["c_ratio"] = metrics.get("c_ratio", 0)
            metrics["python_ratio"] = metrics.get("python_ratio", 0)
        
        # Calculate optimization score from both static and HTML analysis
        self._calculate_optimization_score(metrics)
        
        # Log the results
        log_items = [f"{k}={v}" for k, v in metrics.items() 
                    if k not in ["line_categories", "line_contents", "code_features"]]
        logger.info(f"Analysis complete: {', '.join(log_items)}")
        
        # Save a copy of the metrics for later reference
        self.last_metrics = metrics
        
        return metrics
    
    def _analyze_code_static(self, code_str: str) -> Dict[str, Any]:
        """
        Perform static analysis of the code without compilation.
        
        Args:
            code_str: The Cython code to analyze
            
        Returns:
            dict: Static analysis metrics
        """
        lines = code_str.strip().split('\n')
        metrics = {
            "total_lines": len(lines),
            "line_contents": {},  # Store the content of each line
            "code_features": {
                "cdef_vars": 0,      # Number of cdef variable declarations
                "cdef_funcs": 0,     # Number of cdef function declarations
                "cpdef_funcs": 0,    # Number of cpdef function declarations
                "memoryviews": 0,    # Number of memoryview type declarations
                "typed_args": 0,     # Number of typed function arguments
                "nogil": 0,          # Number of nogil blocks or functions
                "prange": 0,         # Number of parallel range loops
                "directives": 0,     # Number of cython directives
            },
            "static_c_ratio": 0.0,   # Estimated C-to-Python ratio from static analysis
        }
        
        # Store line contents for reference
        for i, line in enumerate(lines):
            metrics["line_contents"][i+1] = line.strip()
        
        # Count features using regex patterns
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Cython directives
            if "# cython:" in stripped:
                metrics["code_features"]["directives"] += 1
            
            # cdef variables
            if re.search(r'cdef\s+(?!class|(?:class\s+)?(?:struct|enum|union))', stripped):
                metrics["code_features"]["cdef_vars"] += 1
            
            # cdef functions
            if re.search(r'cdef\s+\w+[\s\w\[\],\*]*\s+\w+\s*\(', stripped):
                metrics["code_features"]["cdef_funcs"] += 1
            
            # cpdef functions
            if re.search(r'cpdef\s+\w+[\s\w\[\],\*]*\s+\w+\s*\(', stripped):
                metrics["code_features"]["cpdef_funcs"] += 1
            
            # Memoryviews
            if "[:]" in stripped or "[:,:]" in stripped:
                metrics["code_features"]["memoryviews"] += 1
            
            # Typed arguments (in function declarations)
            if re.search(r'\(\s*\w+\s*:\s*\w+', stripped) or re.search(r',\s*\w+\s*:\s*\w+', stripped):
                metrics["code_features"]["typed_args"] += 1
            
            # nogil blocks or functions
            if "nogil" in stripped:
                metrics["code_features"]["nogil"] += 1
            
            # Parallel range (prange)
            if "prange" in stripped:
                metrics["code_features"]["prange"] += 1
        
        # Calculate feature density
        c_features_count = sum(metrics["code_features"].values())
        metrics["feature_density"] = c_features_count / max(1, len(lines))
        
        # Estimate C-to-Python ratio based on feature density
        # This is just an estimate, the actual ratio will come from HTML analysis
        if metrics["feature_density"] > 0.3:
            metrics["static_c_ratio"] = 0.7  # High density of Cython features
        elif metrics["feature_density"] > 0.1:
            metrics["static_c_ratio"] = 0.5  # Medium density
        else:
            metrics["static_c_ratio"] = 0.3  # Low density
        
        logger.info(f"Static analysis: {c_features_count} Cython features found, density: {metrics['feature_density']:.3f}")
        return metrics
    
    def _compile_with_annotation(self, pyx_path: str) -> Optional[str]:
        """
        Compile a .pyx file with annotation enabled to generate HTML output.
        
        Args:
            pyx_path: Path to the .pyx file to compile
            
        Returns:
            str: Path to the generated HTML file, or None if compilation failed
        """
        # Create a minimal setup.py for compilation
        setup_py_path = os.path.join(self.temp_dir, "setup.py")
        module_name = os.path.splitext(os.path.basename(pyx_path))[0]
        
        with open(setup_py_path, "w") as f:
            f.write(f"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

# Try to get NumPy include directory if NumPy is imported in the code
numpy_include = []
try:
    import numpy
    numpy_include = [numpy.get_include()]
except ImportError:
    pass

extensions = [
    Extension(
        "{module_name}",
        ["{os.path.basename(pyx_path)}"],
        include_dirs=numpy_include
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={{'language_level': "3"}},
        annotate=True
    )
)
""")
        
        # Create and set up a virtual environment for compilation
        venv_dir = os.path.join(self.temp_dir, "venv")
        self._setup_venv(venv_dir)
        
        # Run the setup.py to compile with annotation
        logger.info(f"Compiling {pyx_path} with annotation")
        try:
            # Determine python executable
            if sys.platform == "win32":
                python_exe = os.path.join(venv_dir, "Scripts", "python")
            else:
                python_exe = os.path.join(venv_dir, "bin", "python")
                
            output = subprocess.check_output(
                [python_exe, setup_py_path, "build_ext", "--inplace"],
                cwd=self.temp_dir,
                stderr=subprocess.STDOUT,
                text=True
            )
            logger.debug(f"Compilation output: {output}")
            
            # Find the HTML file - try multiple possible locations
            # Cython can generate HTML files with different naming patterns:
            # 1. module_name.html (most common)
            # 2. path/to/file.pyx.html (less common)
            
            module_name = os.path.splitext(os.path.basename(pyx_path))[0]
            possible_paths = [
                os.path.join(self.temp_dir, f"{module_name}.html"),  # module_name.html
                f"{pyx_path}.html",  # file.pyx.html
                os.path.join(os.path.dirname(pyx_path), f"{module_name}.html")  # dir/module_name.html
            ]
            
            # Log all the places we're looking
            logger.debug(f"Looking for HTML annotation files at: {possible_paths}")
            
            # Check each possible location
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Generated annotation HTML found at {path}")
                    return path
            
            # If we get here, we haven't found an HTML file
            logger.error(f"HTML annotation not found in any expected location")
            
            # List all files in the directory to help debug
            dir_contents = os.listdir(self.temp_dir)
            html_files = [f for f in dir_contents if f.endswith('.html')]
            if html_files:
                logger.info(f"Found these HTML files in directory: {html_files}")
                # Return the first HTML file found
                found_path = os.path.join(self.temp_dir, html_files[0])
                logger.info(f"Using HTML file: {found_path}")
                return found_path
                
            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed: {e.output}")
            return None
    
    def _setup_venv(self, venv_dir: str) -> bool:
        """
        Set up a virtual environment with the necessary dependencies.
        
        Args:
            venv_dir: Path to create the virtual environment
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Create the virtual environment
            import venv
            venv.create(venv_dir, with_pip=True)
            
            # Determine pip executable
            if sys.platform == "win32":
                pip_exe = os.path.join(venv_dir, "Scripts", "pip")
            else:
                pip_exe = os.path.join(venv_dir, "bin", "pip")
            
            # Install setuptools, wheel, and Cython
            subprocess.check_call(
                [pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Install Cython and NumPy (if needed)
            subprocess.check_call(
                [pip_exe, "install", "cython", "numpy"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            return True
        except Exception as e:
            logger.error(f"Error setting up virtual environment: {str(e)}")
            return False
    
    def _parse_annotation_html(self, html_path: str) -> Dict[str, Any]:
        """
        Parse the Cython-generated HTML annotation to extract optimization metrics.
        
        Args:
            html_path: Path to the HTML annotation file
            
        Returns:
            dict: Metrics including Python interaction, C operations, etc.
        """
        try:
            with open(html_path, "r") as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize metrics
            metrics = {
                "python_lines": 0,
                "c_lines": 0,
                "py_object_interactions": 0,
                "gil_operations": 0,
                "vectorizable_loops": 0,
                "unoptimized_math": 0,
                "line_categories": {}  # Will store line number -> category mappings
            }
            
            # Process each line
            processed_lines = 0
            for line in soup.find_all("div", class_="line"):
                try:
                    line_num = int(line.get("id", "0").replace("line", ""))
                    processed_lines += 1
                    
                    # Skip empty lines
                    if not line.text.strip():
                        continue
                    
                    # Categories based on Cython's color coding
                    category = None
                    
                    # Check for Python interaction (yellow background)
                    if line.find("span", style=lambda s: s and "background-color: #FFFF00" in s):
                        metrics["python_lines"] += 1
                        metrics["py_object_interactions"] += 1
                        category = "python_interaction"
                    
                    # Check for GIL acquisition (pink background)
                    elif line.find("span", style=lambda s: s and "background-color: #FFABAB" in s):
                        metrics["gil_operations"] += 1
                        category = "gil_acquisition"
                    
                    # Check for pure C operations (no special highlighting)
                    elif not line.find("span", style=lambda s: s and "background-color:" in s):
                        metrics["c_lines"] += 1
                        category = "c_operation"
                    
                    # Check for loops that could be vectorized
                    if "for" in line.text and "range" in line.text and not "prange" in line.text:
                        metrics["vectorizable_loops"] += 1
                        category = category or "vectorizable_loop"
                    
                    # Check for unoptimized math operations
                    if any(op in line.text for op in ["+", "-", "*", "/"]) and "double" in line.text:
                        if not line.find("span", style=lambda s: s and "color: #0000FF" in s):
                            metrics["unoptimized_math"] += 1
                            category = category or "unoptimized_math"
                    
                    # Store the category for this line
                    if category:
                        metrics["line_categories"][line_num] = category
                except Exception as e:
                    logger.warning(f"Error processing line in HTML: {str(e)}")
            
            # If we didn't parse any lines from the HTML but we have static metrics,
            # use the static metrics to estimate
            if processed_lines == 0:
                logger.warning("No lines processed from HTML annotation, using estimates from static analysis")
                # We'll calculate these in the main analyze_code method
                pass
            else:
                metrics["total_lines"] = processed_lines
            
            logger.info(f"Parsed annotation HTML, processed {processed_lines} lines")
            return metrics
            
        except Exception as e:
            logger.error(f"Error parsing HTML annotation: {str(e)}")
            return {"error": f"Error parsing HTML annotation: {str(e)}"}
    
    def _calculate_optimization_score(self, metrics: Dict[str, Any]) -> None:
        """
        Calculate the overall optimization score based on analysis metrics.
        
        Args:
            metrics: The metrics dictionary to update with the optimization score
        """
        # Set defaults for missing metrics
        c_ratio = metrics.get("c_ratio", metrics.get("static_c_ratio", 0))
        python_ratio = metrics.get("python_ratio", 1.0 - c_ratio)
        gil_ops = metrics.get("gil_operations", 0)
        vec_loops = metrics.get("vectorizable_loops", 0)
        unopt_math = metrics.get("unoptimized_math", 0)
        total_lines = metrics.get("total_lines", 1)
        
        # Calculate feature quality score from code features
        code_features = metrics.get("code_features", {})
        feature_score = 0.0
        
        # Award points for each optimization feature
        if code_features.get("nogil", 0) > 0:
            feature_score += 0.2
        
        if code_features.get("prange", 0) > 0:
            feature_score += 0.2
        
        if code_features.get("memoryviews", 0) > 0:
            feature_score += 0.2
        
        if code_features.get("directives", 0) > 0:
            feature_score += 0.1
        
        if code_features.get("cdef_vars", 0) / max(1, total_lines) > 0.1:
            feature_score += 0.2
        
        if code_features.get("cdef_funcs", 0) + code_features.get("cpdef_funcs", 0) > 0:
            feature_score += 0.1
        
        # Normalize feature score to 0-1 range
        feature_score = min(1.0, feature_score)
        
        # Overall optimization score (weighted components)
        optimization_score = (
            # Reward high C ratio (40% weight)
            (c_ratio * 0.4) + 
            # Penalize Python interaction (20% weight) 
            ((1.0 - python_ratio) * 0.2) + 
            # Penalize GIL operations (10% weight)
            ((1.0 - min(1.0, gil_ops / max(1, total_lines))) * 0.1) +
            # Penalize unvectorized loops (10% weight)
            ((1.0 - min(1.0, vec_loops / max(1, total_lines))) * 0.1) +
            # Reward good Cython features (20% weight)
            (feature_score * 0.2)
        )
        
        # Store all component scores for detailed reporting
        metrics["component_scores"] = {
            "c_ratio_score": c_ratio,
            "python_interaction_score": 1.0 - python_ratio,
            "gil_operations_score": 1.0 - min(1.0, gil_ops / max(1, total_lines)),
            "vectorizable_loops_score": 1.0 - min(1.0, vec_loops / max(1, total_lines)),
            "feature_score": feature_score
        }
        
        metrics["optimization_score"] = optimization_score
        logger.info(f"Calculated optimization score: {optimization_score:.2f}")

def analyze_cython_code(code_str: str, temp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze Cython code without creating an analyzer instance.
    
    Args:
        code_str: String containing the Cython code to analyze
        temp_dir: Optional directory to use for temporary files
        
    Returns:
        dict: Analysis metrics
    """
    analyzer = CythonAnnotationAnalyzer(temp_dir=temp_dir)
    return analyzer.analyze_code(code_str)

def is_cython_code(code_str: str) -> bool:
    """
    Check if a code string appears to be Cython.
    
    Args:
        code_str: The code string to check
        
    Returns:
        bool: True if the code contains Cython-specific elements
    """
    cython_indicators = ["cdef", "cpdef", "cimport", "nogil", "# cython:"]
    return any(indicator in code_str for indicator in cython_indicators)

def get_optimization_hints(metrics: Dict[str, Any]) -> Dict[int, str]:
    """
    Generate optimization hints based on the analysis metrics.
    
    Args:
        metrics: Analysis metrics from CythonAnnotationAnalyzer
        
    Returns:
        dict: Line number -> optimization hint mappings
    """
    hints = {}
    line_categories = metrics.get("line_categories", {})
    
    for line_num, category in line_categories.items():
        if category == "python_interaction":
            hints[line_num] = "Consider using typed variables to avoid Python object interaction"
        elif category == "gil_acquisition":
            hints[line_num] = "This operation requires the GIL. Consider using 'nogil' where possible"
        elif category == "vectorizable_loop":
            hints[line_num] = "This loop could be vectorized using 'prange' from cython.parallel"
        elif category == "unoptimized_math":
            hints[line_num] = "Use typed variables for math operations to enable C-level performance"
    
    return hints

def get_optimization_report(metrics: Dict[str, Any]) -> str:
    """
    Generate a detailed report of the optimization metrics.
    
    Args:
        metrics: Analysis metrics from CythonAnnotationAnalyzer
        
    Returns:
        str: Detailed optimization report
    """
    if "error" in metrics:
        return f"Analysis failed: {metrics['error']}"
    
    report = ["Cython Optimization Analysis Report"]
    report.append("=" * 40)
    report.append("")
    
    # Basic metrics
    report.append(f"Total lines: {metrics.get('total_lines', 0)}")
    report.append(f"C code lines: {metrics.get('c_lines', 0)} ({metrics.get('c_ratio', 0):.2%})")
    report.append(f"Python interaction lines: {metrics.get('python_lines', 0)} ({metrics.get('python_ratio', 0):.2%})")
    report.append(f"GIL operations: {metrics.get('gil_operations', 0)}")
    report.append(f"Vectorizable loops: {metrics.get('vectorizable_loops', 0)}")
    report.append(f"Unoptimized math operations: {metrics.get('unoptimized_math', 0)}")
    report.append("")
    
    # Cython features
    code_features = metrics.get('code_features', {})
    if code_features:
        report.append("Cython Features:")
        report.append(f"- C variable declarations (cdef): {code_features.get('cdef_vars', 0)}")
        report.append(f"- C functions (cdef): {code_features.get('cdef_funcs', 0)}")
        report.append(f"- Python-accessible C functions (cpdef): {code_features.get('cpdef_funcs', 0)}")
        report.append(f"- Memory views: {code_features.get('memoryviews', 0)}")
        report.append(f"- Type-annotated arguments: {code_features.get('typed_args', 0)}")
        report.append(f"- No-GIL sections: {code_features.get('nogil', 0)}")
        report.append(f"- Parallel loops (prange): {code_features.get('prange', 0)}")
        report.append(f"- Cython directives: {code_features.get('directives', 0)}")
        report.append("")
    
    # Component scores
    component_scores = metrics.get('component_scores', {})
    if component_scores:
        report.append("Optimization Score Components:")
        report.append(f"- C ratio score: {component_scores.get('c_ratio_score', 0):.2f} (40% weight)")
        report.append(f"- Python interaction score: {component_scores.get('python_interaction_score', 0):.2f} (20% weight)")
        report.append(f"- GIL operations score: {component_scores.get('gil_operations_score', 0):.2f} (10% weight)")
        report.append(f"- Vectorizable loops score: {component_scores.get('vectorizable_loops_score', 0):.2f} (10% weight)")
        report.append(f"- Feature usage score: {component_scores.get('feature_score', 0):.2f} (20% weight)")
        report.append("")
    
    # Final score
    report.append(f"Overall optimization score: {metrics.get('optimization_score', 0):.2f}")
    
    # Optimization suggestions
    line_categories = metrics.get('line_categories', {})
    if line_categories:
        report.append("\nOptimization Suggestions:")
        python_lines = [line for line, cat in line_categories.items() if cat == "python_interaction"]
        gil_lines = [line for line, cat in line_categories.items() if cat == "gil_acquisition"]
        loop_lines = [line for line, cat in line_categories.items() if cat == "vectorizable_loop"]
        math_lines = [line for line, cat in line_categories.items() if cat == "unoptimized_math"]
        
        if python_lines:
            report.append(f"- Lines with Python interaction (use C types): {python_lines[:5]}")
            
        if gil_lines:
            report.append(f"- Lines with GIL acquisition (use nogil): {gil_lines[:5]}")
            
        if loop_lines:
            report.append(f"- Loops that could be vectorized (use prange): {loop_lines[:5]}")
            
        if math_lines:
            report.append(f"- Unoptimized math operations (use C types): {math_lines[:5]}")
    
    return "\n".join(report)