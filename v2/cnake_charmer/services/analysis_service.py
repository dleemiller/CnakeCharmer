# services/analysis_service.py
import logging
from typing import Dict, List, Optional, Any

from core.models import AnalysisResult
from core.enums import LanguageType, AnalysisType
from repositories.code_repo import GeneratedCodeRepository
from repositories.analysis_repo import AnalysisResultRepository
from analyzers.static_analyzer import StaticCodeAnalyzer


class AnalysisService:
    """Service for analyzing code."""
    
    def __init__(
        self, 
        code_repo: GeneratedCodeRepository, 
        analysis_repo: AnalysisResultRepository,
        analyzers: Dict[AnalysisType, StaticCodeAnalyzer]
    ):
        """
        Initialize analysis service.
        
        Args:
            code_repo: Repository for generated code
            analysis_repo: Repository for analysis results
            analyzers: Dictionary mapping analysis types to analyzers
        """
        self.code_repo = code_repo
        self.analysis_repo = analysis_repo
        self.analyzers = analyzers
        self.logger = logging.getLogger(__name__)
    
    async def analyze_code(
        self, 
        request_id: str, 
        language: Optional[LanguageType] = None,
        analysis_type: Optional[AnalysisType] = None
    ) -> Dict[LanguageType, Dict[AnalysisType, AnalysisResult]]:
        """
        Analyze code for a request.
        
        Args:
            request_id: Request ID
            language: Specific language to analyze (or all if None)
            analysis_type: Specific analysis type to run (or all if None)
            
        Returns:
            Nested dictionary mapping languages to analysis types to results
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
        
        # Determine analyzers to use
        if analysis_type:
            if analysis_type not in self.analyzers:
                self.logger.warning(f"No analyzer available for type {analysis_type}")
                return {}
            
            analyzers_dict = {analysis_type: self.analyzers[analysis_type]}
        else:
            analyzers_dict = self.analyzers
        
        # Analyze each language with each analyzer
        analysis_results = {}
        
        for lang, code in code_dict.items():
            lang_results = {}
            
            for a_type, analyzer in analyzers_dict.items():
                if lang not in analyzer.supported_languages():
                    self.logger.warning(f"Language {lang} not supported by {a_type} analyzer")
                    continue
                
                self.logger.info(f"Running {a_type} analysis on {lang} code for request {request_id}")
                
                # Analyze the code
                result = analyzer.analyze(code, lang, {})
                
                # Store the result
                # TODO: Get code_id from code_repo
                # For now, we'll just use a placeholder
                code_id = 0
                self.analysis_repo.save_result(code_id, a_type, result)
                
                lang_results[a_type] = result
            
            if lang_results:
                analysis_results[lang] = lang_results
        
        return analysis_results