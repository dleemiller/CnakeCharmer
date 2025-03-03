# services/build_service.py
import logging
from typing import Dict, List, Optional, Any

from core.models import BuildResult
from core.enums import LanguageType
from repositories.code_repo import GeneratedCodeRepository
from repositories.build_repo import BuildResultRepository
from builders.base import BaseBuilder


class BuildService:
    """Service for building code."""
    
    def __init__(
        self, 
        code_repo: GeneratedCodeRepository, 
        build_repo: BuildResultRepository,
        builders: Dict[LanguageType, BaseBuilder]
    ):
        """
        Initialize build service.
        
        Args:
            code_repo: Repository for generated code
            build_repo: Repository for build results
            builders: Dictionary mapping languages to their respective builders
        """
        self.code_repo = code_repo
        self.build_repo = build_repo
        self.builders = builders
        self.logger = logging.getLogger(__name__)
    
    async def build_code(self, request_id: str, language: Optional[LanguageType] = None) -> Dict[LanguageType, BuildResult]:
        """
        Build code for a request.
        
        Args:
            request_id: Request ID
            language: Specific language to build (or all if None)
            
        Returns:
            Dictionary mapping languages to build results
        """
        # Get generated code
        if language:
            code = self.code_repo.get_latest_code(request_id, language)
            if not code:
                self.logger.warning(f"No {language} code found for request {request_id}")
                return {}
            
            code_dict = {language: code}
        else:
            code_dict = self.code_repo.get_all_latest_code(request_id)
            if not code_dict:
                self.logger.warning(f"No code found for request {request_id}")
                return {}
        
        # Build each language
        build_results = {}
        
        for lang, code in code_dict.items():
            if lang not in self.builders:
                self.logger.warning(f"No builder available for language {lang}")
                continue
            
            self.logger.info(f"Building {lang} code for request {request_id}")
            
            # Build the code
            result = self.builders[lang].build(code, lang, {})
            
            # Store the result
            # TODO: Get code_id from code_repo
            # For now, we'll just use a placeholder
            code_id = 0
            self.build_repo.save_result(code_id, result)
            
            build_results[lang] = result
        
        return build_results