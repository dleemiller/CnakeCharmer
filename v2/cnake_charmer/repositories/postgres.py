# repositories/postgres.py
import logging
import os
from typing import Dict, Optional, Any, List

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from core.exceptions import DatabaseError


class PostgresRepository:
    """Base class for PostgreSQL repositories."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize PostgreSQL repository.
        
        Args:
            connection_string: PostgreSQL connection string (optional)
        """
        self.connection_string = connection_string or os.environ.get(
            'DATABASE_URL', 'postgresql://user:password@localhost/cnake_charmer'
        )
        self.logger = logging.getLogger(__name__)
        self._conn = None
    
    @property
    def conn(self):
        """Get a database connection."""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(self.connection_string)
                self.logger.debug("Connected to PostgreSQL")
            except psycopg2.Error as e:
                self.logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise DatabaseError(f"Failed to connect to PostgreSQL: {e}")
        
        return self._conn
    
    def execute(self, query: str, params: tuple = None) -> Any:
        """
        Execute a query and return the result.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Query result
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params or ())
                self.conn.commit()
                
                if cur.description:
                    return cur.fetchall()
                
                return None
        except psycopg2.Error as e:
            self.conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database error: {e}")
    
    def execute_dict(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute a query and return the result as a list of dictionaries.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                self.conn.commit()
                
                if cur.description:
                    return cur.fetchall()
                
                return []
        except psycopg2.Error as e:
            self.conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database error: {e}")
    
    def close(self):
        """Close the database connection."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
            self.logger.debug("Closed PostgreSQL connection")