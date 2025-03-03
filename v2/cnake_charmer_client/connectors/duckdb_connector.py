# connectors/duckdb_connector.py
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import duckdb

from core.models import CythonEntry, PythonEntry, BatchJob, ProcessingError, ProcessingStatus, BatchStatus
from core.exceptions import DuckDBConnectionError


class DuckDBConnector:
    """Connector for interacting with DuckDB database."""
    
    def __init__(self, db_path: str, table_name: str = "cython_entries", 
                 result_table: str = "python_entries", batch_table: str = "batch_jobs", 
                 error_table: str = "processing_errors"):
        """
        Initialize DuckDB connection.
        
        Args:
            db_path: Path to DuckDB database file
            table_name: Name of table containing Cython entries
            result_table: Name of table for storing Python results
            batch_table: Name of table for batch jobs
            error_table: Name of table for processing errors
        """
        self.db_path = db_path
        self.table_name = table_name
        self.result_table = result_table
        self.batch_table = batch_table
        self.error_table = error_table
        self.logger = logging.getLogger(__name__)
        
        try:
            self.conn = duckdb.connect(db_path)
            self._init_schema()
            self.logger.info(f"Connected to DuckDB at {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to DuckDB: {e}")
            raise DuckDBConnectionError(f"Failed to connect to DuckDB: {e}")
    
    def _init_schema(self):
        """Initialize database schema if tables don't exist."""
        try:
            # Create Cython entries table if it doesn't exist
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR PRIMARY KEY,
                    code TEXT NOT NULL,
                    description TEXT,
                    metadata JSON,
                    tags JSON,
                    processed BOOLEAN DEFAULT FALSE,
                    processing_status VARCHAR DEFAULT 'pending',
                    python_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            """)
            
            # Create Python entries table if it doesn't exist
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.result_table} (
                    id VARCHAR PRIMARY KEY,
                    code TEXT NOT NULL,
                    cython_id VARCHAR,
                    metadata JSON,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create batch jobs table if it doesn't exist
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.batch_table} (
                    id VARCHAR PRIMARY KEY,
                    total_entries INTEGER NOT NULL,
                    processed_entries INTEGER DEFAULT 0,
                    successful_entries INTEGER DEFAULT 0,
                    failed_entries INTEGER DEFAULT 0,
                    status VARCHAR DEFAULT 'running',
                    options JSON,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Create processing errors table if it doesn't exist
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.error_table} (
                    id VARCHAR PRIMARY KEY,
                    entry_id VARCHAR NOT NULL,
                    batch_id VARCHAR,
                    error_message TEXT,
                    error_type VARCHAR,
                    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.logger.info("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Error initializing schema: {e}")
            raise DuckDBConnectionError(f"Error initializing schema: {e}")
    
    def get_unprocessed_entries(self, limit: int, offset: int = 0) -> List[CythonEntry]:
        """
        Get unprocessed Cython code entries.
        
        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of CythonEntry objects
        """
        try:
            query = f"""
                SELECT 
                    id, code, description, metadata, tags, processed, 
                    processing_status, python_id, created_at, processed_at
                FROM {self.table_name}
                WHERE processed = False
                LIMIT {limit} OFFSET {offset}
            """
            result = self.conn.execute(query).fetchall()
            
            entries = []
            for row in result:
                entry = CythonEntry(
                    id=row[0],
                    code=row[1],
                    description=row[2],
                    metadata=row[3] if row[3] else {},
                    tags=row[4] if row[4] else [],
                    processed=row[5],
                    processing_status=ProcessingStatus(row[6]) if row[6] else ProcessingStatus.PENDING,
                    python_id=row[7],
                    created_at=row[8],
                    processed_at=row[9]
                )
                entries.append(entry)
            
            return entries
        except Exception as e:
            self.logger.error(f"Error getting unprocessed entries: {e}")
            raise DuckDBConnectionError(f"Error getting unprocessed entries: {e}")
    
    def get_entry_by_id(self, entry_id: str) -> Optional[CythonEntry]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: ID of the entry to get
            
        Returns:
            CythonEntry object or None if not found
        """
        try:
            query = f"""
                SELECT 
                    id, code, description, metadata, tags, processed, 
                    processing_status, python_id, created_at, processed_at
                FROM {self.table_name}
                WHERE id = ?
            """
            result = self.conn.execute(query, [entry_id]).fetchone()
            
            if not result:
                return None
                
            return CythonEntry(
                id=result[0],
                code=result[1],
                description=result[2],
                metadata=result[3] if result[3] else {},
                tags=result[4] if result[4] else [],
                processed=result[5],
                processing_status=ProcessingStatus(result[6]) if result[6] else ProcessingStatus.PENDING,
                python_id=result[7],
                created_at=result[8],
                processed_at=result[9]
            )
        except Exception as e:
            self.logger.error(f"Error getting entry by ID: {e}")
            raise DuckDBConnectionError(f"Error getting entry by ID: {e}")
    
    def mark_as_processed(self, entry_id: str, success: bool, python_id: Optional[str] = None) -> bool:
        """
        Mark an entry as processed with result status.
        
        Args:
            entry_id: ID of the entry to mark
            success: Whether processing was successful
            python_id: ID of the generated Python code (if successful)
            
        Returns:
            True if the entry was marked successfully, False otherwise
        """
        try:
            status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
            
            query = f"""
                UPDATE {self.table_name}
                SET 
                    processed = True,
                    processing_status = ?,
                    python_id = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            self.conn.execute(query, [status.value, python_id, entry_id])
            return True
        except Exception as e:
            self.logger.error(f"Error marking entry as processed: {e}")
            raise DuckDBConnectionError(f"Error marking entry as processed: {e}")
    
    def store_python_code(self, python_code: str, cython_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store generated Python code with metadata.
        
        Args:
            python_code: Generated Python code
            cython_id: ID of the Cython entry that generated this code
            metadata: Additional metadata
            
        Returns:
            ID of the stored Python code
        """
        try:
            # Generate a new ID based on Cython ID
            python_id = f"py_{cython_id}"
            
            query = f"""
                INSERT INTO {self.result_table} (id, code, cython_id, metadata, generated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            self.conn.execute(query, [python_id, python_code, cython_id, metadata or {}])
            
            return python_id
        except Exception as e:
            self.logger.error(f"Error storing Python code: {e}")
            raise DuckDBConnectionError(f"Error storing Python code: {e}")
    
    def create_batch_record(self, total_entries: int, options: Dict[str, Any] = None) -> str:
        """
        Create a record for a new batch processing job.
        
        Args:
            total_entries: Total number of entries in the batch
            options: Batch processing options
            
        Returns:
            Batch ID
        """
        try:
            # Generate a batch ID
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            query = f"""
                INSERT INTO {self.batch_table} (
                    id, total_entries, processed_entries, successful_entries,
                    failed_entries, status, options, started_at
                )
                VALUES (?, ?, 0, 0, 0, ?, ?, CURRENT_TIMESTAMP)
            """
            self.conn.execute(query, [
                batch_id, 
                total_entries, 
                BatchStatus.RUNNING.value, 
                options or {}
            ])
            
            return batch_id
        except Exception as e:
            self.logger.error(f"Error creating batch record: {e}")
            raise DuckDBConnectionError(f"Error creating batch record: {e}")
    
    def update_batch_status(self, batch_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update status of a batch processing job.
        
        Args:
            batch_id: ID of the batch to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Build update query dynamically based on provided fields
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key in ['processed_entries', 'successful_entries', 'failed_entries']:
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                elif key == 'status':
                    set_clauses.append("status = ?")
                    params.append(value.value if isinstance(value, BatchStatus) else value)
                elif key == 'completed_at' and value is True:
                    set_clauses.append("completed_at = CURRENT_TIMESTAMP")
            
            if not set_clauses:
                return False
                
            query = f"""
                UPDATE {self.batch_table}
                SET {', '.join(set_clauses)}
                WHERE id = ?
            """
            params.append(batch_id)
            
            self.conn.execute(query, params)
            return True
        except Exception as e:
            self.logger.error(f"Error updating batch status: {e}")
            raise DuckDBConnectionError(f"Error updating batch status: {e}")
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """
        Get the status of a batch processing job.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            BatchJob object or None if not found
        """
        try:
            query = f"""
                SELECT 
                    id, total_entries, processed_entries, successful_entries,
                    failed_entries, status, options, started_at, completed_at
                FROM {self.batch_table}
                WHERE id = ?
            """
            result = self.conn.execute(query, [batch_id]).fetchone()
            
            if not result:
                return None
                
            return BatchJob(
                id=result[0],
                total_entries=result[1],
                processed_entries=result[2],
                successful_entries=result[3],
                failed_entries=result[4],
                status=BatchStatus(result[5]) if result[5] else BatchStatus.RUNNING,
                options=result[6] if result[6] else {},
                started_at=result[7],
                completed_at=result[8]
            )
        except Exception as e:
            self.logger.error(f"Error getting batch status: {e}")
            raise DuckDBConnectionError(f"Error getting batch status: {e}")
    
    def log_processing_error(self, entry_id: str, error_message: str, 
                            batch_id: Optional[str] = None, error_type: str = "unknown") -> str:
        """
        Log an error encountered during processing.
        
        Args:
            entry_id: ID of the entry that encountered the error
            error_message: Error message
            batch_id: ID of the batch (if applicable)
            error_type: Type of error
            
        Returns:
            ID of the error record
        """
        try:
            # Generate an error ID
            error_id = f"err_{datetime.now().strftime('%Y%m%d%H%M%S')}_{entry_id}"
            
            query = f"""
                INSERT INTO {self.error_table} (
                    id, entry_id, batch_id, error_message, error_type, occurred_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            self.conn.execute(query, [
                error_id, 
                entry_id, 
                batch_id, 
                error_message, 
                error_type
            ])
            
            return error_id
        except Exception as e:
            self.logger.error(f"Error logging processing error: {e}")
            raise DuckDBConnectionError(f"Error logging processing error: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about processing progress.
        
        Returns:
            Dictionary of statistics
        """
        try:
            stats = {}
            
            # Get total entries
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            result = self.conn.execute(query).fetchone()
            stats['total_entries'] = result[0] if result else 0
            
            # Get processed entries
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE processed = True"
            result = self.conn.execute(query).fetchone()
            stats['processed_entries'] = result[0] if result else 0
            
            # Get successful entries
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE processing_status = '{ProcessingStatus.COMPLETED.value}'"
            result = self.conn.execute(query).fetchone()
            stats['successful_entries'] = result[0] if result else 0
            
            # Get failed entries
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE processing_status = '{ProcessingStatus.FAILED.value}'"
            result = self.conn.execute(query).fetchone()
            stats['failed_entries'] = result[0] if result else 0
            
            # Calculate success rate
            if stats['processed_entries'] > 0:
                stats['success_rate'] = stats['successful_entries'] / stats['processed_entries'] * 100
            else:
                stats['success_rate'] = 0
            
            return stats
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise DuckDBConnectionError(f"Error getting statistics: {e}")
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
            self.logger.info("Database connection closed")