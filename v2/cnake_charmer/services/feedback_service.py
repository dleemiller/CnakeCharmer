# services/feedback_service.py
import logging
from typing import Dict, List, Optional, Any

from core.models import CodeGenerationRequest
from core.enums import LanguageType, RequestStatus
from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository


class FeedbackService:
    """Service for processing feedback and improving code."""
    
    def __init__(self, request_repo: CodeRequestRepository, code_repo: GeneratedCodeRepository):
        """
        Initialize feedback service.
        
        Args:
            request_repo: Repository for code generation requests
            code_repo: Repository for generated code
        """
        self.request_repo = request_repo
        self.code_repo = code_repo
        self.logger = logging.getLogger(__name__)
    
    def process_feedback(
        self, 
        code: str,
        language: LanguageType,
        feedback: Dict,
        generation_history: List[Dict]
    ) -> str:
        """
        Process feedback and generate improved code.
        
        Args:
            code: Original code
            language: Code language
            feedback: Feedback data
            generation_history: History of previous generations
            
        Returns:
            Improved code
        """
        self.logger.info(f"Processing feedback for {language} code")
        
        # Extract feedback details
        feedback_type = feedback.get("type", "general")
        feedback_message = feedback.get("message", "")
        feedback_source = feedback.get("source", "user")
        
        # Generate improvement prompt based on feedback
        improvement_prompt = self.generate_improvement_prompt(
            code,
            {"feedback": feedback},
            [feedback_message]
        )
        
        # TODO: Call LLM to generate improved code
        # For now, just return the original code with a comment
        improved_code = f"""
# Original code improved based on feedback: {feedback_message}
# Feedback type: {feedback_type}
# Feedback source: {feedback_source}

{code}
"""
        
        return improved_code
    
    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        """
        Determine if we should retry after an error.
        
        Args:
            error: The error that occurred
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            
        Returns:
            True if we should retry, False otherwise
        """
        # Don't retry if we've reached the maximum attempts
        if attempt >= max_attempts:
            return False
        
        # Retry for most errors
        retryable_errors = [
            "SyntaxError",
            "TimeoutError",
            "ConnectionError",
            "BuildError"
        ]
        
        # Check if error type is in the list
        error_type = type(error).__name__
        return error_type in retryable_errors
    
    def generate_improvement_prompt(
        self, 
        original_code: str,
        analysis_results: Dict,
        error_messages: List[str]
    ) -> str:
        """
        Generate a prompt for improving code based on feedback.
        
        Args:
            original_code: Original code
            analysis_results: Analysis results
            error_messages: Error messages
            
        Returns:
            Improvement prompt
        """
        # Extract relevant information from analysis results
        suggestions = []
        if "feedback" in analysis_results:
            suggestions.append(analysis_results["feedback"].get("message", ""))
        
        # Create a prompt that includes the original code and feedback
        prompt = f"""
Please improve the following code based on these issues:
{', '.join(error_messages)}

Additional suggestions:
{', '.join(suggestions)}

Original code:
{original_code}
Copy
Please provide an improved version that addresses these issues.
"""
        
        return prompt 