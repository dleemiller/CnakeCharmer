# repositories/analysis_repo.py
import json
import logging
from typing import Dict, List, Optional, Any

from repositories.postgres import PostgresRepository
from core.models import AnalysisResult, Suggestion
from core.enums import LanguageType, AnalysisType


class AnalysisResultRepository(PostgresRepository):
    """Repository for analysis results."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository."""
        super().__init__(connection_string)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the table exists
        self._create_table()
    
    def _create_table(self):
        """Create the analysis_results table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            code_id INT NOT NULL,
            analyzer_type TEXT NOT NULL,
            score FLOAT NOT NULL,
            details JSONB,
            suggestions JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute(query)
        
        # Create index for code_id
        query = """
        CREATE INDEX IF NOT EXISTS idx_analysis_results_code_id ON analysis_results (code_id)
        """
        self.execute(query)
        
        self.logger.debug("Ensured analysis_results table exists")
    
    def save_result(self, code_id: int, analyzer_type: AnalysisType, result: AnalysisResult) -> int:
        """
        Save an analysis result.
        
        Args:
            code_id: ID of the generated code
            analyzer_type: Type of analyzer
            result: Analysis result
            
        Returns:
            ID of the saved result
        """
        query = """
        INSERT INTO analysis_results (
            code_id, analyzer_type, score, details, suggestions
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        # Convert suggestions to JSON-serializable format
        suggestions_json = []
        for suggestion in result.suggestions:
            suggestions_json.append({
                "line": suggestion.line,
                "message": suggestion.message,
                "severity": suggestion.severity,
                "code": suggestion.code,
                "replacement": suggestion.replacement
            })
        
        params = (
            code_id,
            analyzer_type.value,
            result.score,
            json.dumps(result.details),
            json.dumps(suggestions_json)
        )
        
        results = self.execute(query, params)
        result_id = results[0][0]
        
        self.logger.info(f"Saved {analyzer_type} analysis result for code {code_id}, score: {result.score}")
        
        return result_id
    
    def get_by_code_id(self, code_id: int) -> Dict[AnalysisType, AnalysisResult]:
        """
        Get all analysis results for a code ID.
        
        Args:
            code_id: ID of the generated code
            
        Returns:
            Dictionary mapping analyzer types to analysis results
        """
        query = """
        SELECT analyzer_type, score, details, suggestions
        FROM analysis_results
        WHERE code_id = %s
        ORDER BY created_at DESC
        """
        
        results = self.execute(query, (code_id,))
        
        analysis_results = {}
        
        for result in results:
            analyzer_type = AnalysisType(result[0])
            details = json.loads(result[2]) if result[2] else {}
            
            # Convert suggestions back to objects
            suggestions_json = json.loads(result[3]) if result[3] else []
            suggestions = []
            
            for suggestion_json in suggestions_json:
                suggestions.append(Suggestion(
                    line=suggestion_json["line"],
                    message=suggestion_json["message"],
                    severity=suggestion_json["severity"],
                    code=suggestion_json.get("code"),
                    replacement=suggestion_json.get("replacement")
                ))
            
            analysis_results[analyzer_type] = AnalysisResult(
                score=result[1],
                details=details,
                suggestions=suggestions
            )
        
        return analysis_results
    
    def get_for_request(self, request_id: str) -> Dict[LanguageType, Dict[AnalysisType, AnalysisResult]]:
        """
        Get all analysis results for a request ID.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Nested dictionary mapping languages to analyzer types to analysis results
        """
        query = """
        WITH latest_code_ids AS (
            SELECT gc.id, gc.language
            FROM generated_code gc
            JOIN (
                SELECT request_id, language, MAX(version) AS max_version
                FROM generated_code
                WHERE request_id = %s
                GROUP BY request_id, language
            ) latest ON gc.request_id = latest.request_id 
                AND gc.language = latest.language 
                AND gc.version = latest.max_version
        )
        SELECT ar.analyzer_type, ar.score, ar.details, ar.suggestions, lc.language
        FROM analysis_results ar
        JOIN latest_code_ids lc ON ar.code_id = lc.id
        ORDER BY ar.created_at DESC
        """
        
        results = self.execute(query, (request_id,))
        
        analysis_results = {}
        
        for result in results:
            analyzer_type = AnalysisType(result[0])
            language = LanguageType(result[4])
            details = json.loads(result[2]) if result[2] else {}
            
            # Convert suggestions back to objects
            suggestions_json = json.loads(result[3]) if result[3] else []
            suggestions = []
            
            for suggestion_json in suggestions_json:
                suggestions.append(Suggestion(
                    line=suggestion_json["line"],
                    message=suggestion_json["message"],
                    severity=suggestion_json["severity"],
                    code=suggestion_json.get("code"),
                    replacement=suggestion_json.get("replacement")
                ))
            
            if language not in analysis_results:
                analysis_results[language] = {}
            
            analysis_results[language][analyzer_type] = AnalysisResult(
                score=result[1],
                details=details,
                suggestions=suggestions
            )
        
        return analysis_results