# repositories/request_repo.py
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from repositories.postgres import PostgresRepository
from core.models import CodeGenerationRequest, CodeGenerationResult
from core.enums import LanguageType, AnalysisType, RequestStatus


class CodeRequestRepository(PostgresRepository):
    """Repository for code generation requests."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository."""
        super().__init__(connection_string)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the table exists
        self._create_table()
    
    def _create_table(self):
        """Create the code_requests table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS code_requests (
            id SERIAL PRIMARY KEY,
            request_id TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL,
            source_language TEXT,
            target_languages JSONB NOT NULL,
            source_code TEXT,
            equivalency_check BOOLEAN NOT NULL DEFAULT FALSE,
            optimization_level INT NOT NULL DEFAULT 1,
            analysis_types JSONB,
            max_attempts INT NOT NULL DEFAULT 3,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute(query)
        self.logger.debug("Ensured code_requests table exists")
    
    def create(self, request: CodeGenerationRequest) -> str:
        """
        Create a new code generation request.
        
        Args:
            request: The code generation request
            
        Returns:
            The request ID
        """
        # Generate a request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{abs(hash(request.prompt)) % 10000}"
        
        query = """
        INSERT INTO code_requests (
            request_id, prompt, source_language, target_languages, source_code,
            equivalency_check, optimization_level, analysis_types, max_attempts, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING request_id
        """
        
        params = (
            request_id,
            request.prompt,
            request.source_language.value if request.source_language else None,
            json.dumps([lang.value for lang in request.target_languages]),
            request.source_code,
            request.equivalency_check,
            request.optimization_level,
            json.dumps([analysis.value for analysis in request.analysis_types]),
            request.max_attempts,
            RequestStatus.PENDING.value
        )
        
        self.execute(query, params)
        self.logger.info(f"Created code generation request: {request_id}")
        
        return request_id
    
    def get_by_id(self, request_id: str) -> Optional[CodeGenerationRequest]:
        """
        Get a code generation request by ID.
        
        Args:
            request_id: The request ID
            
        Returns:
            The code generation request or None if not found
        """
        query = """
        SELECT * FROM code_requests WHERE request_id = %s
        """
        
        results = self.execute_dict(query, (request_id,))
        
        if not results:
            return None
        
        result = results[0]
        
        return CodeGenerationRequest(
            prompt=result['prompt'],
            source_language=(
                LanguageType(result['source_language']) 
                if result['source_language'] 
                else None
            ),
            target_languages=[
                LanguageType(lang) 
                for lang in json.loads(result['target_languages'])
            ],
            source_code=result['source_code'],
            equivalency_check=result['equivalency_check'],
            optimization_level=result['optimization_level'],
            analysis_types=[
                AnalysisType(analysis) 
                for analysis in json.loads(result['analysis_types'])
            ],
            max_attempts=result['max_attempts']
        )
    
    def update_status(self, request_id: str, status: RequestStatus) -> bool:
        """
        Update the status of a code generation request.
        
        Args:
            request_id: The request ID
            status: The new status
            
        Returns:
            True if the update was successful, False otherwise
        """
        query = """
        UPDATE code_requests 
        SET status = %s, updated_at = CURRENT_TIMESTAMP
        WHERE request_id = %s
        """
        
        self.execute(query, (status.value, request_id))
        self.logger.info(f"Updated request {request_id} status to {status.value}")
        
        return True
    
    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """
        Get the status of a code generation request.
        
        Args:
            request_id: The request ID
            
        Returns:
            The request status or None if not found
        """
        query = """
        SELECT status FROM code_requests WHERE request_id = %s
        """
        
        results = self.execute(query, (request_id,))
        
        if not results:
            return None
        
        return RequestStatus(results[0][0])