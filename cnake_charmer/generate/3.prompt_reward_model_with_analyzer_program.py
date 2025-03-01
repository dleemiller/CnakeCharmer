"""
Integration Example

This example demonstrates how to use the modular components together
with or without the DSPy framework, using markdown files for instructions.
"""

import os
import logging
import tempfile
import json
from typing import Dict, Any, Optional
import re
from dotenv import load_dotenv
import sys

# Configure logging - more detailed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("integration_example")

# Set up file logging
try:
    file_handler = logging.FileHandler("cython_evaluator.log")
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    logger.info("File logging initialized to cython_evaluator.log")
except Exception as e:
    logger.warning(f"Could not initialize file logging: {str(e)}")

# Load environment variables for API keys
load_dotenv()

# Import our modular components
try:
    from code_runner import CodeRunner, run_code
    from cython_analyzer import CythonAnnotationAnalyzer, analyze_cython_code
    from reward_system import RewardSystem, create_default_reward_system
except ImportError as e:
    logger.warning(f"Error importing module: {str(e)}")
    logger.warning("Make sure you're running from the correct directory (project root)")

# Try to import DSPy components (optional)
try:
    import dspy
    HAS_DSPY = True
except ImportError:
    logger.warning("DSPy not found, DSPy-specific functionality will be disabled")
    HAS_DSPY = False

# Example Cython code for testing
EXAMPLE_CYTHON_CODE = '''
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np

def dot_product(double[:] a, double[:] b):
    """
    Compute the dot product of two vectors efficiently.
    
    Args:
        a: First vector as a memoryview
        b: Second vector as a memoryview
        
    Returns:
        float: The dot product of the vectors
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must be the same length")
        
    cdef int i
    cdef int n = a.shape[0]
    cdef double result = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
        
    return result
'''

# Function to load instructions from a markdown file
def load_instructions_from_file(filename: str) -> str:
    """
    Load instructions from a markdown file in the prompt folder.
    
    Args:
        filename: The name of the markdown file (e.g., 'prompt1.md')
        
    Returns:
        str: The contents of the markdown file
    """
    from pathlib import Path
    
    try:
        # Find the project root by looking for directories that should be there
        # Start with the current file's directory and go up until we find the project root
        current_path = Path(__file__).resolve().parent
        
        # Go up directories until we find the 'prompt' folder or reach the filesystem root
        project_root = current_path
        while project_root != project_root.parent:
            if (project_root / 'prompt').exists():
                break
            # Go up one directory
            project_root = project_root.parent
            
            # Safety check to avoid infinite loop
            if project_root == project_root.parent:
                logger.error("Could not find project root containing 'prompt' directory")
                raise ValueError("Project structure not as expected - couldn't locate prompt directory")
        
        # Now that we have the project root, look for the file in various locations
        prompt_dir = project_root / 'prompt'
        
        # Possible locations for the file
        possible_paths = [
            prompt_dir / filename,                      # Direct in prompt folder
            prompt_dir / 'cython_notes' / filename,     # In cython_notes subfolder
            Path(filename)                              # Direct path if absolute
        ]
        
        # Find the first path that exists
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
                
        # If we didn't find it, try one more approach - maybe it's just the basename we need
        if file_path is None:
            # Search for any file with the same name in the prompt directory and its subdirectories
            for path in prompt_dir.glob(f"**/{filename}"):
                if path.exists():
                    file_path = path
                    break
        
        if file_path is None:
            logger.error(f"Could not find file {filename} in prompt directory or subdirectories")
            raise FileNotFoundError(f"File {filename} not found in prompt directory structure")
            
        logger.info(f"Loading instructions from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"Successfully loaded {len(content)} bytes from {filename}")
        return content
    except Exception as e:
        logger.error(f"Error loading instructions from {filename}: {str(e)}")
        raise ValueError(f"Failed to load instructions from {filename}: {str(e)}")

class CythonEvaluator:
    """
    A standalone evaluator for Cython code that uses our modular components.
    """
    
    def __init__(self):
        self.code_runner = CodeRunner()
        self.analyzer = CythonAnnotationAnalyzer(temp_dir=tempfile.mkdtemp())
        self.reward_system = create_default_reward_system()
        
        # Register an annotation-based scoring function
        self.reward_system.register_scoring_function(
            self._annotation_score, weight=0.5, name="annotation"
        )
    
    def evaluate_code(self, code_str: str, prompt: str = "") -> Dict[str, Any]:
        """
        Evaluate Cython code by running it and analyzing its quality.
        
        Args:
            code_str: The Cython code to evaluate
            prompt: Optional prompt that generated the code
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating Cython code ({len(code_str)} bytes)")
        
        # Run the code
        logger.info("Running code in isolated environment")
        run_result = self.code_runner.run_code(code_str)
        
        # Store results
        evaluation_results = {
            "execution": run_result,
            "code_size": len(code_str),
            "language": "cython" if self.code_runner._is_cython_code(code_str) else "python"
        }
        
        # Only analyze if it's actually Cython code
        is_cython = self.code_runner._is_cython_code(code_str)
        if not is_cython:
            logger.warning("Code does not appear to be Cython, skipping Cython-specific analysis")
            evaluation_results["error"] = "Code does not contain Cython features"
            return evaluation_results
        
        # Only analyze with annotation if compilation succeeded
        try:
            logger.info("Analyzing code with Cython annotation")
            analysis = self.analyzer.analyze_code(code_str)
            evaluation_results["analysis"] = analysis
            
            if "error" in analysis:
                logger.error(f"Analysis failed: {analysis['error']}")
                evaluation_results["analysis_error"] = analysis["error"]
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            evaluation_results["analysis_error"] = str(e)
        
        # Calculate reward score
        try:
            logger.info("Calculating reward score")
            inputs = {"prompt": prompt}
            outputs = {
                "generated_code": code_str,
                "execution": run_result
            }
            reward = self.reward_system.calculate_reward(inputs, outputs)
            evaluation_results["reward"] = reward
            evaluation_results["detailed_scores"] = self.reward_system.get_detailed_scores()
            evaluation_results["score_explanation"] = self.reward_system.get_score_explanation()
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            evaluation_results["reward_error"] = str(e)
        
        logger.info(f"Evaluation complete. Reward score: {evaluation_results.get('reward', 0.0):.2f}")
        return evaluation_results
    
    def _annotation_score(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> float:
            """
            Calculate a score based on Cython annotation analysis.
            
            Args:
                inputs: The inputs to the code generator
                outputs: The outputs from the code generator
                
            Returns:
                float: Annotation-based score between 0 and 1
            """
            code = outputs.get("generated_code", "")
            if not code or not self.code_runner._is_cython_code(code):
                logger.warning("No valid Cython code found for annotation scoring")
                return 0.0
            
            try:
                # Get metrics from analysis or last metrics
                metrics = outputs.get("analysis", {})
                if not metrics:
                    metrics = getattr(self.analyzer, "last_metrics", {})
                
                if not metrics or "error" in metrics:
                    logger.warning(f"No valid metrics for annotation scoring: {metrics.get('error', 'Unknown error')}")
                    return 0.0
                
                # Use the optimization score components for detailed logging
                component_scores = metrics.get("component_scores", {})
                if component_scores:
                    # Log detailed component scores for transparency
                    c_ratio_score = component_scores.get("c_ratio_score", 0.0)
                    py_interaction_score = component_scores.get("python_interaction_score", 0.0)
                    gil_score = component_scores.get("gil_operations_score", 0.0)
                    vectorizable_score = component_scores.get("vectorizable_loops_score", 0.0)
                    feature_score = component_scores.get("feature_score", 0.0)
                    
                    logger.info(f"Annotation score components: c_ratio={c_ratio_score:.2f}, "
                              f"python_interaction={py_interaction_score:.2f}, "
                              f"gil_operations={gil_score:.2f}, "
                              f"vectorizable_loops={vectorizable_score:.2f}, "
                              f"feature_score={feature_score:.2f}")
                else:
                    # Fall back to basic metrics
                    c_ratio = metrics.get("c_ratio", 0.0)
                    python_ratio = metrics.get("python_ratio", 0.0)
                    gil_ops = metrics.get("gil_operations", 0)
                    
                    logger.info(f"Annotation score details: c_ratio={c_ratio:.2f}, python_ratio={python_ratio:.2f}, "
                              f"gil_operations={gil_ops}, optimization_score={metrics.get('optimization_score', 0):.2f}")
                
                # The final optimization score
                optimization_score = metrics.get("optimization_score", 0.0)
                
                # Return the optimization score directly
                return optimization_score
            except Exception as e:
                logger.error(f"Error in annotation scoring: {str(e)}")
                return 0.0
        
    def format_report(self, evaluation: Dict[str, Any]) -> str:
        """
        Format the evaluation results as a detailed report.
        
        Args:
            evaluation: The evaluation results from evaluate_code()
            
        Returns:
            str: A formatted report
        """
        report = ["=== Cython Code Evaluation Report ===\n"]
        
        # Basic code info
        report.append(f"Code Size: {evaluation.get('code_size', 0)} bytes")
        report.append(f"Language: {evaluation.get('language', 'unknown')}")
        
        # Execution results
        execution = evaluation.get("execution", {})
        success = execution.get("success", False)
        report.append(f"\n-- Execution Status: {'SUCCESS' if success else 'FAILED'} --")
        
        if not success:
            stderr = execution.get("stderr", "")
            report.append(f"Error Message: {stderr[:500]}" + ("..." if len(stderr) > 500 else ""))
        else:
            stdout = execution.get("stdout", "")
            if stdout:
                report.append("Output:")
                report.append(stdout[:200] + ("..." if len(stdout) > 200 else ""))
        
        # Reward score and breakdown
        report.append("\n-- Reward Score --")
        if "reward" in evaluation:
            report.append(f"Overall Score: {evaluation['reward']:.2f}")
            
            # Add detailed score breakdown if available
            if "score_explanation" in evaluation:
                report.append("\nScore Breakdown:")
                report.append(evaluation["score_explanation"])
        else:
            report.append("Score calculation failed")
            if "reward_error" in evaluation:
                report.append(f"Error: {evaluation['reward_error']}")
        
        # Analysis results
        report.append("\n-- Cython Analysis --")
        analysis = evaluation.get("analysis", {})
        if "error" in analysis:
            report.append(f"Analysis failed: {analysis['error']}")
        elif "analysis_error" in evaluation:
            report.append(f"Analysis failed: {evaluation['analysis_error']}")
        else:
            # Get the detailed optimization report if available
            if "analysis" in evaluation:
                from cython_analyzer import get_optimization_report
                optimization_report = get_optimization_report(analysis)
                report.append(optimization_report)
            else:
                # Fallback to basic metrics
                c_ratio = analysis.get("c_ratio", 0.0)
                python_ratio = analysis.get("python_ratio", 0.0)
                gil_ops = analysis.get("gil_operations", 0)
                optimization = analysis.get("optimization_score", 0.0)
                
                report.append(f"C Ratio: {c_ratio:.2f} (higher is better)")
                report.append(f"Python Interaction Ratio: {python_ratio:.2f} (lower is better)")
                report.append(f"GIL Operations: {gil_ops} (fewer is better)")
                report.append(f"Optimization Score: {optimization:.2f} (higher is better)")
                
                # Show Cython features if available
                code_features = analysis.get("code_features", {})
                if code_features:
                    report.append("\nCython Features:")
                    for feature, count in code_features.items():
                        report.append(f"- {feature}: {count}")
                
                # Add component scores if available
                component_scores = analysis.get("component_scores", {})
                if component_scores:
                    report.append("\nOptimization Score Components:")
                    for component, score in component_scores.items():
                        report.append(f"- {component}: {score:.2f}")
                
                # Add suggestions for improvement
                if "line_categories" in analysis:
                    categories = analysis["line_categories"]
                    if categories:
                        report.append("\nSuggestions for Improvement:")
                        python_lines = [line for line, cat in categories.items() if cat == "python_interaction"]
                        gil_lines = [line for line, cat in categories.items() if cat == "gil_acquisition"]
                        loop_lines = [line for line, cat in categories.items() if cat == "vectorizable_loop"]
                        
                        if python_lines:
                            report.append(f"- Lines with Python interaction (consider using C types): {python_lines[:5]}")
                        if gil_lines:
                            report.append(f"- Lines with GIL acquisition (consider nogil): {gil_lines[:5]}")
                        if loop_lines:
                            report.append(f"- Loops that could be vectorized (consider prange): {loop_lines[:5]}")
        
        return "\n".join(report)

# If DSPy is available, create an integrated DSPy module
if HAS_DSPY:
    from dspy.signatures import Signature, InputField, OutputField
    from dspy.primitives import Module
    
    class OptimizedCythonGenerator(Module):
        """
        A DSPy module for generating and evaluating optimized Cython code.
        """
        
        def __init__(self, signature=None, instructions_file=None):
            super().__init__()
            
            # Create a signature if none provided
            if signature is None:
                signature = Signature({
                    "prompt": InputField(
                        prefix="User Prompt:",
                        desc="The user request describing what code to generate",
                        format=str
                    ),
                    "generated_code": OutputField(
                        prefix="Code:",
                        desc="The code snippet that solves the user request",
                        format=str
                    ),
                })
                
            self.signature = signature
            
            # Load instructions from file if provided
            instructions = None
            if instructions_file:
                try:
                    instructions = load_instructions_from_file(instructions_file)
                    logger.info(f"Loaded instructions from {instructions_file}")
                except Exception as e:
                    logger.error(f"Failed to load instructions from {instructions_file}: {str(e)}")
                    logger.info("Falling back to default instructions")
            
            # Default instructions if no file was provided or loading failed
            if instructions is None:
                instructions = (
                    "You are given `prompt` describing a user request. "
                    "Generate code in Cython that solves the request. "
                    "Your response must be enclosed in triple backticks (```), with NO language indicator. "
                    "Include ONLY the code itself, no additional commentary before or after the code block.\n\n"
                    
                    "Code quality requirements:\n"
                    "1. Follow PEP 8 style guidelines (proper spacing, naming conventions)\n"
                    "2. Include Google-style docstrings for all functions, classes, and modules\n"
                    "3. Add appropriate comments for complex logic\n\n"
                    
                    "For Cython optimization:\n"
                    "- Add comment-based directives at the top of the file:\n"
                    "  # cython: boundscheck=False\n"
                    "  # cython: wraparound=False\n"
                    "- Use cdef for variables, especially in loops\n"
                    "- Use memoryviews (e.g., double[:]) for array operations\n"
                    "- Add proper C type declarations for all functions and variables\n"
                    "- Use nogil where possible to avoid the Python GIL\n"
                )
            
            # Initialize the code generation chain
            self.generate_chain = dspy.ChainOfThought(
                Signature(
                    {
                        "prompt": signature.fields["prompt"],
                        "generated_code": signature.fields["generated_code"]
                    },
                    instructions=instructions
                )
            )
            
            # Initialize the evaluator
            self.evaluator = CythonEvaluator()
        
        def forward(self, prompt, **kwargs):
            """
            Generate, run, and evaluate Cython code based on a prompt.
            
            Args:
                prompt: The user prompt describing what code to generate
                **kwargs: Additional arguments
                
            Returns:
                dict: Result containing generated code and evaluation metrics
            """
            logger.info(f"Generating code for prompt: {prompt}")
            
            # Generate the code
            generation = self.generate_chain(prompt=prompt)
            code = generation.generated_code if hasattr(generation, 'generated_code') else ""
            
            # Extract code from backticks if present
            match = re.search(r"```(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
                logger.info("Extracted code from backticks")
            
            # Evaluate the generated code
            evaluation = self.evaluator.evaluate_code(code, prompt)
            
            # Generate a detailed report
            report = self.evaluator.format_report(evaluation)
            
            # Return combined results
            return {
                "generated_code": code,
                "evaluation": evaluation,
                "reward": evaluation.get("reward", 0.0),
                "report": report
            }


# Helper to format and print evaluation reports
def print_evaluation_report(title, evaluation):
    """Print a nicely formatted evaluation report"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)
    
    if isinstance(evaluation, dict) and "report" in evaluation:
        print(evaluation["report"])
    elif isinstance(evaluation, dict) and "evaluation" in evaluation:
        evaluator = CythonEvaluator()
        print(evaluator.format_report(evaluation["evaluation"]))
    else:
        print("Execution success:", evaluation.get("execution", {}).get("success", False))
        print(f"Reward score: {evaluation.get('reward', 0.0):.2f}")
        
        if "analysis" in evaluation:
            analysis = evaluation["analysis"]
            print("\nAnalysis metrics:")
            print(f"- C ratio: {analysis.get('c_ratio', 0):.2f}")
            print(f"- Python ratio: {analysis.get('python_ratio', 0):.2f}")
            print(f"- GIL operations: {analysis.get('gil_operations', 0)}")
            print(f"- Optimization score: {analysis.get('optimization_score', 0):.2f}")
        
        if "score_explanation" in evaluation:
            print("\nScore explanation:")
            print(evaluation["score_explanation"])
        
        if "execution" in evaluation and evaluation["execution"].get("stderr"):
            print("\nExecution errors:")
            print(evaluation["execution"]["stderr"][:500])
    
    print("=" * 80 + "\n")


# Example usage of the standalone evaluator
def standalone_example():
    """Run the standalone evaluator example"""
    print("Running standalone Cython evaluator example...")
    evaluator = CythonEvaluator()
    results = evaluator.evaluate_code(EXAMPLE_CYTHON_CODE)
    
    # Print detailed report
    print_evaluation_report("Standalone Cython Evaluator Example", results)


# Example usage with DSPy if available
def dspy_example(instructions_file=None):
    """
    Run the DSPy integration example if DSPy is available
    
    Args:
        instructions_file: Optional path to a markdown file with instructions
    """
    if not HAS_DSPY:
        print("DSPy is not installed. Skipping DSPy example.")
        return
    
    print("Running DSPy integration example...")
    
    lm = dspy.LM(model="openrouter/anthropic/claude-3.7-sonnet", cache=False, max_tokens=2500)
    dspy.configure(lm=lm)
    
    # Create and use the generator with the specified instructions file
    generator = OptimizedCythonGenerator(instructions_file=instructions_file)
    result = generator(prompt="Write an efficient Cython function for fizzbuzz.")
    
    # Print detailed report
    print_evaluation_report("DSPy Integration Example", result)


if __name__ == "__main__":
    # Print environment information
    print(f"Python version: {sys.version}")
    print(f"Running from: {os.getcwd()}")
    print(f"DSPy available: {HAS_DSPY}")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Cython code generation and evaluation examples')
    parser.add_argument('--standalone', action='store_true', help='Run only the standalone example')
    parser.add_argument('--dspy', action='store_true', help='Run only the DSPy example (if available)')
    parser.add_argument('--instructions', type=str, help='Path to a markdown file with instructions')
    parser.add_argument('--list-prompts', action='store_true', help='List available prompt files')
    
    args = parser.parse_args()
    
    # List available prompt files if requested
    if args.list_prompts:
        from pathlib import Path
        
        # Find project root
        current_path = Path(__file__).resolve().parent
        project_root = current_path
        while project_root != project_root.parent:
            if (project_root / 'prompt').exists():
                break
            project_root = project_root.parent
        
        prompt_dir = project_root / 'prompt'
        if prompt_dir.exists():
            print(f"\nAvailable prompt files in {prompt_dir}:")
            for md_file in sorted(prompt_dir.glob("*.md")):
                print(f"  - {md_file.name}")
            
            cython_notes_dir = prompt_dir / 'cython_notes'
            if cython_notes_dir.exists():
                print(f"\nAvailable prompt files in {cython_notes_dir}:")
                for md_file in sorted(cython_notes_dir.glob("*.md")):
                    print(f"  - cython_notes/{md_file.name}")
        else:
            print(f"Prompt directory not found at {prompt_dir}")
        
        sys.exit(0)
    
    # Determine which examples to run
    run_standalone = args.standalone or (not args.standalone and not args.dspy)
    run_dspy = args.dspy or (not args.standalone and not args.dspy)
    
    # Run the selected examples
    if run_standalone:
        standalone_example()
    
    if run_dspy:
        dspy_example(instructions_file=args.instructions)