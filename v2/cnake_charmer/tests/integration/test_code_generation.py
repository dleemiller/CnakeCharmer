# tests/integration/test_code_generation.py
"""
Integration tests for code generation.
"""
import pytest
import asyncio

from core.models import CodeGenerationRequest
from core.enums import LanguageType, RequestStatus


@pytest.mark.asyncio
async def test_code_generation_flow(code_generator, request_repo, code_repo):
    """Test the full code generation flow."""
    # Create a request
    request = CodeGenerationRequest(
        prompt="Write a function to calculate the factorial of a number",
        target_languages=[LanguageType.PYTHON]
    )
    
    # Generate code
    request_id = await code_generator.generate_code(request)
    
    # Check that the request was created
    assert request_id is not None
    
    # Check that the status was updated
    status = request_repo.get_status(request_id)
    assert status in [RequestStatus.PROCESSING, RequestStatus.COMPLETED]
    
    # Get the generated code
    code = code_repo.get_latest_code(request_id, LanguageType.PYTHON)
    
    # Check that code was generated
    assert code is not None
    assert "def factorial" in code or "def calculate_factorial" in code