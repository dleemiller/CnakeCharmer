# scripts/init_db.py
#!/usr/bin/env python3
"""
Script to initialize the database schema.
"""
import argparse
import logging
import os
import sys

from utils.logging import configure_logging
from utils.db_init import initialize_database

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize CnakeCharmer database")
    parser.add_argument("--db-url", help="Database connection URL")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logger = configure_logging("db_init", args.log_level)
    
    # Get database URL from environment if not provided
    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("Database URL not provided. Use --db-url or set DATABASE_URL environment variable.")
        return 1
    
    try:
        # Initialize database
        logger.info(f"Initializing database at {db_url}")
        initialize_database(db_url)
        logger.info("Database initialization completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())