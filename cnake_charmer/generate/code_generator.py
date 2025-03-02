import dspy
import os
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("code_generator")

class GenerateEquivalentCode(dspy.Module):
    """
    DSPy module to generate equivalent Python and Cython code.
    """

    def __init__(self):
        super().__init__()
        self.signature = dspy.Signature({
            "prompt_id": dspy.InputField(desc="The identifier for the prompt"),
            "language": dspy.InputField(desc="Target language (python or cython)"),
            "base_code": dspy.InputField(desc="Existing code if any"),
            "python_code": dspy.OutputField(desc="Generated Python code"),
            "cython_code": dspy.OutputField(desc="Generated Cython code"),
        })

    def forward(self, prompt_id, language, base_code=""):
        """ Calls the configured language model to generate code. """
        # Return the prediction from dspy.Predict
        prediction = dspy.Predict(self.signature)(
            prompt_id=prompt_id,
            language=language,
            base_code=base_code
        )
        return prediction

class CodeGenerator:
    """Handles code generation for Python and Cython using dspy."""

    def __init__(self, api_key=None, model=None):
        """
        Initialize with a DSPy model (defaults to Claude-3.7 Sonnet via OpenRouter).
        
        Args:
            api_key: API key for OpenRouter (defaults to OPENROUTER_API_KEY env var)
            model: Optional model instance to use
        """
        # Get API key from argument or environment
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenRouter API key provided - API calls will likely fail")
        
        if model:
            self.model = model
        else:
            # Configure the LM with API key if available
            try:
                self.model = dspy.LM(
                    model="openrouter/anthropic/claude-3.7-sonnet", 
                    cache=False, 
                    max_tokens=2500,
                    api_key=self.api_key
                )
                dspy.configure(lm=self.model)
                logger.info("Successfully configured DSPy with OpenRouter/Claude")
            except Exception as e:
                logger.error(f"Failed to initialize DSPy model: {e}")
                # Don't raise - let it fail at generation time instead
                
        self.task = GenerateEquivalentCode()

    def generate_equivalent_code(self, prompt_id: str, language: str, base_code: Optional[str] = None) -> Dict[str, str]:
        """
        Generates equivalent Python and Cython code from a prompt.

        Args:
            prompt_id (str): The identifier for the prompt.
            language (str): 'python' or 'cython'.
            base_code (Optional[str]): Existing code, if any.

        Returns:
            Dict[str, str]: { "python": generated_python_code, "cython": generated_cython_code }
        """
        try:
            # Ensure API key is provided
            if not self.api_key:
                raise ValueError("OpenRouter API key is not set. Set OPENROUTER_API_KEY environment variable.")
            
            logger.info(f"Generating code for prompt ID: {prompt_id}")
            
            # Use dspy.Predict with the signature
            response = dspy.Predict(self.task.signature)(
                prompt_id=prompt_id,
                language=language,
                base_code=base_code or ""
            )
            
            return {
                "python": response.python_code,
                "cython": response.cython_code
            }
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "python": f"# Error generating code: {str(e)}",
                "cython": f"# Error generating code: {str(e)}"
            }