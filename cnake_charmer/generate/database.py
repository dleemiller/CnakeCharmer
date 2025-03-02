import psycopg2
import json
import os
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("database")

class CodeDatabase:
    """Handles storing code generation results in PostgreSQL."""

    def __init__(self, db_url: str):
        """
        Initialize database connection.
        
        Args:
            db_url: PostgreSQL connection string
        """
        self.db_url = db_url
        self.conn = None
        try:
            self.conn = psycopg2.connect(db_url)
            logger.info("Database connection established")
            # Initialize the schema when the connection is created
            self.initialize_schema()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def initialize_schema(self):
        """Create necessary tables if they don't exist."""
        try:
            with self.conn.cursor() as cur:
                # Check if the table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'generated_code'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    logger.info("Creating generated_code table")
                    cur.execute("""
                        CREATE TABLE generated_code (
                            id SERIAL PRIMARY KEY,
                            prompt_id TEXT NOT NULL,
                            python_code TEXT,
                            cython_code TEXT,
                            build_status TEXT DEFAULT 'pending',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create an index on prompt_id for faster lookups
                    cur.execute("""
                        CREATE INDEX idx_generated_code_prompt_id 
                        ON generated_code (prompt_id)
                    """)
                    
                    self.conn.commit()
                    logger.info("Database schema initialized successfully")
                else:
                    logger.info("generated_code table already exists")
                    
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error initializing schema: {e}")
            raise

    def save_code(self, prompt_id: str, generated_code: Dict[str, str], build_status: str):
        """
        Stores generated code and build results.
        
        Args:
            prompt_id: The prompt identifier
            generated_code: Dictionary with 'python' and 'cython' keys
            build_status: Status of the build process
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generated_code 
                    (prompt_id, python_code, cython_code, build_status) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    (prompt_id, generated_code["python"], generated_code["cython"], build_status)
                )
                self.conn.commit()
                logger.info(f"Saved code for prompt ID: {prompt_id}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving code: {e}")
            raise

    def update_status(self, prompt_id: str, build_status: str):
        """
        Updates the build status.
        
        Args:
            prompt_id: The prompt identifier
            build_status: New status of the build process
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE generated_code 
                    SET build_status=%s 
                    WHERE prompt_id=%s
                    """,
                    (build_status, prompt_id)
                )
                self.conn.commit()
                logger.info(f"Updated status for prompt ID {prompt_id}: {build_status}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating status: {e}")
            raise

    def get_code(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves code entry by prompt ID.
        
        Args:
            prompt_id: The prompt identifier
            
        Returns:
            Dictionary with code details or None if not found
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, prompt_id, python_code, cython_code, build_status, created_at
                    FROM generated_code 
                    WHERE prompt_id = %s
                    ORDER BY created_at DESC 
                    LIMIT 1
                    """,
                    (prompt_id,)
                )
                
                result = cur.fetchone()
                if not result:
                    logger.warning(f"No code found for prompt ID: {prompt_id}")
                    return None
                    
                return {
                    "id": result[0],
                    "prompt_id": result[1],
                    "python_code": result[2],
                    "cython_code": result[3],
                    "build_status": result[4],
                    "created_at": result[5]
                }
        except Exception as e:
            logger.error(f"Error retrieving code: {e}")
            raise