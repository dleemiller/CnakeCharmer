# repositories/build_repo.py
import json
import logging
from typing import Dict, List, Optional, Any

from repositories.postgres import PostgresRepository
from core.models import BuildResult
from core.enums import LanguageType


class BuildResultRepository(PostgresRepository):
    """Repository for build results."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository."""
        super().__init__(connection_string)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the table exists
        self._create_table()
    
    def _create_table(self):
        """Create the build_results table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS build_results (
            id SERIAL PRIMARY KEY,
            code_id INT NOT NULL,
            success BOOLEAN NOT NULL,
            build_output TEXT,
            build_error TEXT,
            artifact_path TEXT,
            build_time FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute(query)
        
        # Create index for code_id
        query = """
        CREATE INDEX IF NOT EXISTS idx_build_results_code_id ON build_results (code_id)
        """
        self.execute(query)
        
        self.logger.debug("Ensured build_results table exists")
    
    def save_result(self, code_id: int, result: BuildResult) -> int:
        """
        Save a build result.
        
        Args:
            code_id: ID of the generated code
            result: Build result
            
        Returns:
            ID of the saved result
        """
        query = """
        INSERT INTO build_results (
            code_id, success, build_output, build_error, artifact_path, build_time
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        params = (
            code_id,
            result.success,
            result.output,
            result.error,
            result.artifact_path,
            result.build_time
        )
        
        results = self.execute(query, params)
        result_id = results[0][0]
        
        self.logger.info(f"Saved build result for code {code_id}, success: {result.success}")
        
        return result_id
    
    def get_by_code_id(self, code_id: int) -> Optional[BuildResult]:
        """
        Get the build result for a code ID.
        
        Args:
            code_id: ID of the generated code
            
        Returns:
            Build result or None if not found
        """
        query = """
        SELECT success, build_output, build_error, artifact_path, build_time
        FROM build_results
        WHERE code_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        results = self.execute(query, (code_id,))
        
        if not results:
            return None
        
        result = results[0]
        
        return BuildResult(
            success=result[0],
            output=result[1],
            error=result[2],
            artifact_path=result[3],
            build_time=result[4]
        )
    
    def get_for_request(self, request_id: str) -> Dict[LanguageType, BuildResult]:
        """
        Get all build results for a request ID.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Dictionary mapping languages to build results
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
        SELECT br.success, br.build_output, br.build_error, br.artifact_path, br.build_time, lc.language
        FROM build_results br
        JOIN latest_code_ids lc ON br.code_id = lc.id
        ORDER BY br.created_at DESC
        """
        
        results = self.execute(query, (request_id,))
        
        build_results = {}
        
        for result in results:
            language = LanguageType(result[5])
            
            build_results[language] = BuildResult(
                success=result[0],
                output=result[1],
                error=result[2],
                artifact_path=result[3],
                build_time=result[4]
            )
        
        return build_results