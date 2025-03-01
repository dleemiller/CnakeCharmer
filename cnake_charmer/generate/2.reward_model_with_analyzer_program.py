import re
import logging
import traceback
import dspy
from dspy.signatures import Signature, InputField, OutputField
from dspy.primitives import Module
from program import EphemeralCodeGenerator
# Configure logger for our integrated solution
logger = logging.getLogger("RefinableCodeGenerator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class RefinableCodeGenerator(Module):
    """
    A wrapper around EphemeralCodeGenerator that makes it compatible with the Refine module.
    This allows using Refine's iterative improvement capabilities with EphemeralCodeGenerator's
    code generation and testing functionality.
    """
    def __init__(self, signature=None, max_iters=1):
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
        
        # Create an instance of the original EphemeralCodeGenerator
        # We set max_iters=1 since Refine will handle iterations
        self.code_generator = EphemeralCodeGenerator(signature=signature, max_iters=max_iters)
        
    def forward(self, prompt, hint_=None, **kwargs):
        """
        Generate code based on prompt and optional hint from Refine's feedback.
        
        Args:
            prompt: The original code generation prompt
            hint_: Feedback from Refine's previous iterations
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with generated_code and error fields
        """
        # Log the incoming request
        request_id = id(prompt)
        logger.info(f"Request {request_id}: Processing prompt: {prompt[:50]}...")
        
        # Incorporate hint if available
        if hint_ and hint_ != "N/A":
            logger.info(f"Request {request_id}: Using hint: {hint_[:100]}...")
            enhanced_prompt = f"{prompt}\n\nAdditional guidance based on previous attempt:\n{hint_}"
        else:
            enhanced_prompt = prompt
            
        try:
            # Use the existing EphemeralCodeGenerator
            result = self.code_generator.forward(prompt=enhanced_prompt)
            
            # Log the result status
            if result.get("error"):
                logger.warning(f"Request {request_id}: Code generation produced error: {result['error'][:100]}...")
            else:
                logger.info(f"Request {request_id}: Code generation successful")
                
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id}: Unexpected error during code generation: {str(e)}")
            logger.debug(f"Request {request_id}: Error details: {traceback.format_exc()}")
            return {"generated_code": "", "error": f"Internal error: {str(e)}"}


def code_quality_reward(inputs, outputs):
    """
    An enhanced reward function that evaluates code quality with special focus
    on Cython utilization when Cython is expected, and reports maximum possible
    scores and percentages achieved.
    
    Args:
        inputs: The inputs to the code generator
        outputs: The outputs from the code generator
        
    Returns:
        float: Reward value (higher is better)
    """
    # Base score for successful code
    base_score = 1.0
    
    # Define maximum scores for each category
    max_scores = {
        'base': 1.0,
        'cython': 2.0,      # Maximum for exceptional Cython
        'docs': 1.4,        # Module docstring (0.2) + full coverage (1.0) + ideal comments (0.2)
        'style': 0.4,       # Line length (0.2) + indentation (0.1) + naming (0.1) 
        'opt': 1.0          # Maximum for all optimizations
    }
    
    # If there was an error in compilation/execution, give a negative reward
    if outputs.get("error"):
        error_text = str(outputs["error"]).lower()
        
        # Distinguish between different types of errors
        if "syntax error" in error_text or "parse error" in error_text:
            logger.info("Reward: -1.0 (Syntax/Parse Error)")
            return -1.0
        elif "import error" in error_text or "no module named" in error_text:
            logger.info("Reward: -0.8 (Import Error)")
            return -0.8
        elif "type error" in error_text or "attribute error" in error_text:
            logger.info("Reward: -0.7 (Type/Attribute Error)")
            return -0.7
        elif "runtime error" in error_text or "exception" in error_text:
            logger.info("Reward: -0.6 (Runtime Error)")
            return -0.6
        else:
            logger.info("Reward: -0.5 (Other Error)")
            return -0.5
    
    # Code ran successfully - evaluate quality
    code = outputs["generated_code"]
    code_lines = code.split('\n')
    total_lines = len(code_lines)
    
    # Initialize score categories
    documentation_score = 0.0
    style_score = 0.0
    optimization_score = 0.0
    cython_score = 0.0
    
    # Determine if the prompt suggests Cython is expected
    prompt = inputs.get("prompt", "").lower()
    cython_expected = any(keyword in prompt for keyword in 
                         ["cython", "fast", "performance", "efficient", "speed", "optimize"])
    
    # Detect actual Cython usage levels
    is_cython_file = any(indicator in code for indicator in 
                       ["# cython:", "cimport", "cpdef", "cdef class"])
    
    # More granular detection of Cython feature usage
    cython_features = {
        'cdef_vars': sum(1 for line in code_lines if re.search(r'cdef\s+(?!class)', line)),
        'cdef_class': sum(1 for line in code_lines if 'cdef class' in line),
        'cpdef_funcs': sum(1 for line in code_lines if 'cpdef' in line),
        'memoryviews': sum(1 for line in code_lines if '[:]' in line or re.search(r'[a-z]+\[:', line)),
        'typed_args': sum(1 for line in code_lines if ':' in line and any(
                      t in line for t in ['int', 'float', 'double', 'long', 'bint'])),
        'c_types': sum(1 for line in code_lines if any(
                    t in line for t in ['int ', 'float ', 'double ', 'bint ', 'unsigned '])),
        'nogil': sum(1 for line in code_lines if 'nogil' in line),
        'directives': sum(1 for line in code_lines if '# cython:' in line)
    }
    
    # Count total cython features used
    total_cython_features = sum(cython_features.values())
    
    #==========================================================================
    # 1. CYTHON UTILIZATION SCORING (if Cython is expected)
    #==========================================================================
    
    if cython_expected:
        # Base check - is it even a Cython file?
        if is_cython_file:
            cython_score += 0.5
            
            # Calculate Cython feature density (features per SLOC)
            feature_density = total_cython_features / max(1, (total_lines - cython_features['directives']))
            
            # Score based on feature density (more features = better Cython utilization)
            if feature_density >= 0.5:  # At least one Cython feature every 2 lines
                cython_score += 1.0
            elif feature_density >= 0.25:  # At least one Cython feature every 4 lines
                cython_score += 0.5
            else:
                cython_score += 0.2
                
            # Specific feature bonuses (regardless of density)
            if cython_features['cdef_vars'] > 0:
                cython_score += 0.3  # Using cdef for variables
                
            if cython_features['memoryviews'] > 0:
                cython_score += 0.4  # Using memoryviews (important optimization)
                
            if cython_features['c_types'] > 2:
                cython_score += 0.3  # Good use of C types
                
            # Context-aware nogil evaluation
            # Detect if the algorithm is amenable to nogil
            has_list_ops = any('list' in line or '[' in line and ']' in line for line in code_lines)
            has_dict_ops = any('dict' in line or '{' in line and '}' in line for line in code_lines)
            has_py_objects = has_list_ops or has_dict_ops or 'str' in code
            
            # If algorithm works with pure C types and doesn't need Python objects
            if not has_py_objects and cython_features['c_types'] > 0:
                # Then nogil would be appropriate
                if cython_features['nogil'] > 0:
                    cython_score += 0.3
            # If algorithm needs Python objects, nogil isn't expected
        else:
            # Cython was expected but not delivered - significant penalty
            cython_score -= 1.0
            logger.info("Cython was expected but not provided (-1.0)")
    
    #==========================================================================
    # 2. DOCUMENTATION SCORING
    #==========================================================================
    
    # Check for module-level docstring
    has_module_docstring = False
    for i, line in enumerate(code_lines[:10]):  # Check first 10 lines
        if '"""' in line or "'''" in line:
            has_module_docstring = True
            break
    
    if has_module_docstring:
        documentation_score += 0.2
    
    # Count Google-style docstrings in functions and classes
    docstring_count = 0
    function_count = 0
    
    for i, line in enumerate(code_lines):
        # Match function or class definitions
        if re.match(r'^\s*(def|cdef|cpdef|class)', line):
            function_count += 1
            
            # Look for docstring in next lines
            for j in range(i+1, min(i+10, total_lines)):
                if '"""' in code_lines[j] or "'''" in code_lines[j]:
                    docstring_text = ""
                    # Find the end of the docstring
                    for k in range(j, min(j+20, total_lines)):
                        docstring_text += code_lines[k]
                        if ('"""' in code_lines[k] or "'''" in code_lines[k]) and k > j:
                            break
                    
                    # Check if it looks like a Google-style docstring
                    if any(section in docstring_text for section in 
                          ["Args:", "Returns:", "Yields:", "Raises:", "Attributes:", "Example:"]):
                        docstring_count += 1
                    else:
                        # Simple docstring (less points)
                        docstring_count += 0.5
                    break
    
    # Calculate docstring coverage and award points
    if function_count > 0:
        docstring_ratio = docstring_count / function_count
        # Heavily weight docstring coverage (up to 1.0 points)
        documentation_score += docstring_ratio * 1.0
    
    # Check for inline comments (good practice but don't overdo)
    comment_lines = sum(1 for line in code_lines if line.strip().startswith('#') and not 
                       any(directive in line for directive in ["cython:", "distutils:"]))
    comment_ratio = comment_lines / max(total_lines, 1)
    
    # Ideal comment ratio is between 10% and 25% of code lines
    if 0.1 <= comment_ratio <= 0.25:
        documentation_score += 0.2
    elif comment_ratio > 0 and comment_ratio < 0.1:
        documentation_score += 0.1  # Some comments, but could use more
    
    #==========================================================================
    # 3. CODE STYLE SCORING
    #==========================================================================
    
    # Check line length (PEP 8 recommends <= 79 chars)
    long_lines = sum(1 for line in code_lines if len(line) > 79)
    if long_lines == 0:
        style_score += 0.2
    elif long_lines / total_lines < 0.1:  # Less than 10% of lines are too long
        style_score += 0.1
    
    # Check for consistent indentation
    indentation_pattern = re.compile(r'^(\s*)\S')
    indentation_types = set()
    for line in code_lines:
        match = indentation_pattern.match(line)
        if match and match.group(1):
            indentation_types.add(match.group(1))
    
    # Fewer indentation types is better (ideally just multiples of spaces or tabs)
    if len(indentation_types) <= 2:  # Allow for zero indentation and one indentation level
        style_score += 0.1
    
    # Check for proper function/variable naming (snake_case for Python)
    camel_case_pattern = re.compile(r'[a-z][a-z0-9]*[A-Z]')
    camel_case_names = sum(1 for line in code_lines if camel_case_pattern.search(line))
    
    if camel_case_names == 0:  # Proper snake_case throughout
        style_score += 0.1
    
    #==========================================================================
    # 4. OPTIMIZATION SCORING
    #==========================================================================
    
    # Evaluate algorithm-specific optimizations
    if is_cython_file:
        # Cython optimizations
        directive_score = 0.0
        
        # Core optimizations that are almost always beneficial
        if "# cython: boundscheck=False" in code:
            directive_score += 0.15
        
        if "# cython: wraparound=False" in code:
            directive_score += 0.15
            
        # Additional optimizations that may be context-dependent
        if "# cython: cdivision=True" in code and any('/' in line for line in code_lines):
            directive_score += 0.1  # Only reward if division is used
            
        if "# cython: initializedcheck=False" in code:
            directive_score += 0.1
            
        optimization_score += directive_score
        
        # Check for algorithm-specific optimizations
        has_loops = any('for' in line for line in code_lines)
        has_math = any(op in code for op in ['+', '-', '*', '/', 'sqrt(', 'pow(', 'sin(', 'cos('])
        
        if has_loops and has_math:
            # This would be a good candidate for parallel execution
            if 'prange' in code:
                optimization_score += 0.3
                
            # Using external C libraries for math
            if 'libc.math' in code or 'cmath' in code:
                optimization_score += 0.2
    else:
        # Python optimizations
        if any(algo in code for algo in ['set(', 'dict(', 'defaultdict', 'Counter']):
            optimization_score += 0.1  # Efficient data structures
            
        if any(lib in code for lib in ['numpy', 'pandas', 'numba']):
            optimization_score += 0.3  # Using performance libraries
            
        list_comprehensions = sum(1 for line in code_lines if '[' in line and 'for' in line)
        if list_comprehensions > 0:
            optimization_score += 0.1  # Using list comprehensions
    
    # Calculate weights based on whether Cython was expected
    weights = {
        'base': 1.0,
        'cython': 0.0,  # Default weight for cython score
        'docs': 0.3,    # 30% weight for documentation
        'style': 0.1,   # 10% weight for code style
        'opt': 0.1      # 10% weight for optimizations
    }
    
    # Adjust weights based on whether Cython was expected
    if cython_expected:
        weights['cython'] = 0.5  # 50% weight for Cython utilization when expected
    
    # Calculate category scores (weighted)
    weighted_scores = {
        'base': base_score,
        'cython': cython_score * weights['cython'],
        'docs': documentation_score * weights['docs'],
        'style': style_score * weights['style'],
        'opt': optimization_score * weights['opt']
    }
    
    # Calculate final score
    final_score = sum(weighted_scores.values())
    
    # Calculate percentages of maximum possible for each category
    percentages = {
        'base': 100 * base_score / max_scores['base'],
        'cython': 100 * cython_score / max_scores['cython'] if max_scores['cython'] > 0 else 0,
        'docs': 100 * documentation_score / max_scores['docs'],
        'style': 100 * style_score / max_scores['style'],
        'opt': 100 * optimization_score / max_scores['opt']
    }
    
    # Calculate the total maximum possible score (weighted)
    max_total = max_scores['base'] + (max_scores['cython'] * weights['cython']) + \
               (max_scores['docs'] * weights['docs']) + (max_scores['style'] * weights['style']) + \
               (max_scores['opt'] * weights['opt'])
    
    total_percentage = 100 * final_score / max_total if max_total > 0 else 0
    
    # Log detailed information about Cython features for debugging
    if is_cython_file:
        logger.debug(f"Cython features detected: {cython_features}")
        logger.debug(f"Cython feature density: {total_cython_features/max(1, total_lines):.2f} features per line")
    
    # Log detailed scoring breakdown with percentages
    logger.info(
        f"Reward breakdown: base={base_score:.2f}/{max_scores['base']:.2f} ({percentages['base']:.0f}%), "
        f"cython={cython_score:.2f}/{max_scores['cython']:.2f} ({percentages['cython']:.0f}%), "
        f"docs={documentation_score:.2f}/{max_scores['docs']:.2f} ({percentages['docs']:.0f}%), "
        f"style={style_score:.2f}/{max_scores['style']:.2f} ({percentages['style']:.0f}%), "
        f"opt={optimization_score:.2f}/{max_scores['opt']:.2f} ({percentages['opt']:.0f}%)"
    )
    logger.info(f"Final reward: {final_score:.2f}/{max_total:.2f} ({total_percentage:.0f}%)")
    
    return final_score
  
def create_refinable_code_generator(max_iters=1, refine_iterations=3, threshold=1.0):
    """
    Creates a Refine module with our RefinableCodeGenerator.
    
    Args:
        max_iters: Maximum iterations for inner EphemeralCodeGenerator
        refine_iterations: Number of refinement attempts in Refine
        threshold: Reward threshold for stopping refinement
        
    Returns:
        Configured Refine module with RefinableCodeGenerator
    """
    # Create the signature for our code generator
    code_signature = Signature({
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
    
    # Create the wrapped code generator
    generator = RefinableCodeGenerator(signature=code_signature, max_iters=max_iters)
    
    # Create the Refine module
    refinable_code_gen = dspy.Refine(
        module=generator,
        N=refine_iterations,
        reward_fn=code_quality_reward,
        threshold=threshold
    )
    
    return refinable_code_gen


# Example usage
if __name__ == "__main__":
    # Set up DSPy with your preferred LM
    lm = dspy.LM(model="openrouter/anthropic/claude-3.7-sonnet:thinking", cache=False, max_tokens=16000)
    dspy.configure(lm=lm)
    
    # Create the refinable code generator
    # We set a high threshold to encourage high-quality Cython code
    refinable_gen = create_refinable_code_generator(
        max_iters=1,        # Let Refine handle iterations
        refine_iterations=3, # Number of refinement attempts
        threshold=2.0       # High enough that we want optimized code
    )
    
    # Example prompt for generating Cython code
    prompt = "Write an efficient cython class for a BFS algorithm."
    logger.info(f"Generating code for prompt: {prompt}")
    
    # Use the Refine module to generate and refine code 
    result = refinable_gen(prompt=prompt)
    
    # Check the final result
    if not result.get("error"):
        logger.info("Final code generation successful!")
        logger.info(f"Generated code:\n{result['generated_code']}")
    else:
        logger.error(f"Final code generation failed with error: {result['error']}")
        logger.info(f"Last code attempt:\n{result['generated_code']}")