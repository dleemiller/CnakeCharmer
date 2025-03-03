# repositories/code_repo.py
import logging
from typing import Dict, List, Optional, Any

from repositories.postgres import PostgresRepository
from core.enums import LanguageType


class GeneratedCodeRepository(PostgresRepository):
    """Repository for generated code."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository."""
        super().__init__(connection_string)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the table exists
        self._create_table()
    
    def _create_table(self):
        """Create the generated_code table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS generated_code (
            id SERIAL PRIMARY KEY,
            request_id TEXT NOT NULL,
            language TEXT NOT NULL,
            code TEXT NOT NULL,
            version INT NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(request_id, language, version)
        )
        """
        self.execute(query)
        self.logger.debug("Ensured generated_code table exists")
    
    def save_code(self, request_id: str, language: LanguageType, code: str) -> int:
        """
        Save generated code.
        
        Args:
            request_id: The request ID
            language: The programming language
            code: The generated code
            
        Returns:
            The code ID
        """
        # Get current version number
        query = """
        SELECT MAX(version) FROM generated_code
        WHERE request_id = %s AND language = %s
        """
        
        results = self.execute(query, (request_id, language.value))
        current_version = results[0][0] if results and results[0][0] else 0
        new_version = current_version + 1
        
        # Insert new version
        query = """
        INSERT INTO generated_code (request_id, language, code, version)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """
        
        results = self.execute(query, (request_id, language.value, code, new_version))
        code_id = results[0][0]
        
        self.logger.info(f"Saved code for request {request_id}, language {language.value}, version {new_version}")
        
        return code_id
    
    def get_latest_code(self, request_id: str, language: LanguageType) -> Optional[str]:
        """
        Get the latest generated code for a request and language.
        
        Args:
            request_id: The request ID
            language: The programming language
            
        Returns:
            The generated code or None if not found
        """
        query = """
        SELECT code FROM generated_code
        WHERE request_id = %s AND language = %s
        ORDER BY version DESC
        LIMIT 1
        """
        
        results = self.execute(query, (request_id, language.value))
        
        if not results:
            return None
        
        return results[0][0]
    
    def get_all_latest_code(self, request_id: str) -> Dict[LanguageType, str]:
        """
        Get all the latest generated code for a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            A dictionary mapping languages to code
        """
        query = """
        WITH latest_versions AS (
            SELECT language, MAX(version) as max_version
            FROM generated_code
            WHERE request_id = %s
            GROUP BY language
        )
        SELECT g.language, g.code
        FROM generated_code g
        JOIN latest_versions lv ON g.language = lv.language AND g.version = lv.max_version
        WHERE g.request_id = %s
        """
        
        results = self.execute(query, (request_id, request_id))
        
        return {LanguageType(row[0]): row[1] for row in results}