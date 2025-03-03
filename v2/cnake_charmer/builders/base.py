# builders/base.py
import os
import tempfile
import logging
from typing import Dict, List, Optional, Any

from core.models import BuildResult, ExecutionResult
from core.enums import LanguageType


class BaseBuilder:
    """Base class for code builders."""
    
    def __init__(self):
        """Initialize builder."""
        self.logger = logging.getLogger(__name__)
    
    def build(self, code: str, language: LanguageType, options: Dict) -> BuildResult:
        """
        Build the provided code.
        
        Args:
            code: The code to build
            language: The programming language
            options: Build options
            
        Returns:
            The build result
        """
        self.logger.warning(f"BaseBuilder does not implement build for {language}")
        return BuildResult(
            success=False,
            output="",
            error="Not implemented",
            build_time=0.0
        )
    
    def run(self, build_artifact: str, inputs: Dict) -> ExecutionResult:
        """
        Run a built artifact.
        
        Args:
            build_artifact: Path to the built artifact
            inputs: Execution inputs
            
        Returns:
            The execution result
        """
        self.logger.warning("BaseBuilder does not implement run")
        return ExecutionResult(
            success=False,
            output="",
            error="Not implemented",
            execution_time=0.0
        )
    
    def supported_languages(self) -> List[LanguageType]:
        """
        Get the list of supported languages.
        
        Returns:
            List of supported languages
        """
        return []
    
    def cleanup(self, artifact_path: str) -> None:
        """
        Clean up any temporary files or resources.
        
        Args:
            artifact_path: Path to the artifact
        """
        if os.path.exists(artifact_path):
            try:
                os.remove(artifact_path)
                self.logger.debug(f"Removed artifact: {artifact_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove artifact {artifact_path}: {e}")