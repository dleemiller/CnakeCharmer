# repositories/equivalency_repo.py
import json
import logging
from typing import Dict, List, Optional, Any

from repositories.postgres import PostgresRepository
from core.models import EquivalencyResult


class EquivalencyResultRepository(PostgresRepository):
    """Repository for equivalency results."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository."""
        super().__init__(connection_string)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the table exists
        self._create_table()
    
    def _create_table(self):
        """Create the equivalency_results table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS equivalency_results (
            id SERIAL PRIMARY KEY,
            request_id TEXT NOT NULL,
            equivalent BOOLEAN NOT NULL,
            test_cases JSONB,
            differences JSONB,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute(query)
        
        # Create index for request_id
        query = """
        CREATE INDEX IF NOT EXISTS idx_equivalency_results_request_id ON equivalency_results (request_id)
        """
        self.execute(query)
        
        self.logger.debug("Ensured equivalency_results table exists")
    
    def save_result(self, request_id: str, result: EquivalencyResult) -> int:
        """
        Save an equivalency result.
        
        Args:
            request_id: ID of the request
            result: Equivalency result
            
        Returns:
            ID of the saved result
        """
        query = """
        INSERT INTO equivalency_results (
            request_id, equivalent, test_cases, differences, error
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        params = (
            request_id,
            result.equivalent,
            json.dumps(result.test_cases) if result.test_cases else None,
            json.dumps(result.differences) if result.differences else None,
            result.error
        )
        
        results = self.execute(query, params)
        result_id = results[0][0]
        
        self.logger.info(f"Saved equivalency result for request {request_id}, equivalent: {result.equivalent}")
        
        return result_id
    
    def get_for_request(self, request_id: str) -> Optional[EquivalencyResult]:
        """
        Get the latest equivalency result for a request ID.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Equivalency result or None if not found
        """
        query = """
        SELECT equivalent, test_cases, differences, error
        FROM equivalency_results
        WHERE request_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        results = self.execute(query, (request_id,))
        
        if not results:
            return None
        
        result = results[0]
        
        return EquivalencyResult(
            equivalent=result[0],
            test_cases=json.loads(result[1]) if result[1] else [],
            differences=json.loads(result[2]) if result[2] else None,
            error=result[3]
        )