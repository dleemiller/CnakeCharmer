# cli/commands.py
import argparse
import asyncio
import json
import logging
import sys
from typing import Dict, Any

from connectors.duckdb_connector import DuckDBConnector
from connectors.api_client import CnakeCharmerClient
from processing.batch_processor import BatchProcessor
from core.config import load_config


async def process_command(args: argparse.Namespace):
    """
    Run the batch processing command.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Configure logging
    logging_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.logging.file
    )
    
    # Create connectors
    try:
        db = DuckDBConnector(
            db_path=args.db_path or config.duckdb.db_path,
            table_name=config.duckdb.table_name,
            result_table=config.duckdb.result_table,
            batch_table=config.duckdb.batch_table,
            error_table=config.duckdb.error_table
        )
        
        api = CnakeCharmerClient(
            base_url=args.api_url or config.api.base_url,
            api_key=args.api_key or config.api.api_key,
            timeout=config.api.timeout,
            max_retries=config.api.max_retries,
            retry_delay=config.api.retry_delay
        )
        
        processor = BatchProcessor(
            db_connector=db,
            api_client=api,
            max_concurrent=args.concurrent or config.processing.max_concurrent,
            requests_per_minute=config.processing.requests_per_minute
        )
        
        # Process batch
        batch_id = await processor.process_batch(
            batch_size=args.batch_size or config.processing.batch_size,
            options={}
        )
        
        if not batch_id:
            print("No entries to process")
            return
        
        print(f"Batch {batch_id} started")
        
        # Monitor progress
        async for update in processor.monitor_progress(batch_id):
            print(f"\rProgress: {update['progress']:.2f}% ({update['processed']}/{update['total']}) "
                  f"- Success: {update['successful']}, Failed: {update['failed']}", end="")
        
        print("\nBatch processing completed")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        if 'api' in locals():
            await api.close()
        if 'db' in locals():
            db.close()


async def status_command(args: argparse.Namespace):
    """
    Run the status checking command.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Create DuckDB connector
    try:
        db = DuckDBConnector(
            db_path=args.db_path or config.duckdb.db_path,
            table_name=config.duckdb.table_name,
            result_table=config.duckdb.result_table,
            batch_table=config.duckdb.batch_table,
            error_table=config.duckdb.error_table
        )
        
        # Get batch status
        batch = db.get_batch_status(args.batch_id)
        
        if not batch:
            print(f"Batch {args.batch_id} not found")
            return
        
        # Display status
        print(f"Batch: {batch.id}")
        print(f"Status: {batch.status.value}")
        print(f"Progress: {batch.processed_entries}/{batch.total_entries} "
              f"({batch.processed_entries/batch.total_entries*100:.2f}%)")
        print(f"Success: {batch.successful_entries} "
              f"({batch.successful_entries/batch.total_entries*100:.2f}%)")
        print(f"Failed: {batch.failed_entries} "
              f"({batch.failed_entries/batch.total_entries*100:.2f}%)")
        print(f"Started: {batch.started_at}")
        print(f"Completed: {batch.completed_at or 'In progress'}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()


async def stats_command(args: argparse.Namespace):
    """
    Run the statistics command.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Create DuckDB connector
    try:
        db = DuckDBConnector(
            db_path=args.db_path or config.duckdb.db_path,
            table_name=config.duckdb.table_name,
            result_table=config.duckdb.result_table,
            batch_table=config.duckdb.batch_table,
            error_table=config.duckdb.error_table
        )
        
        # Get statistics
        stats = db.get_statistics()
        
        # Display statistics
        print("Processing Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Processed entries: {stats['processed_entries']}")
        print(f"Successful entries: {stats['successful_entries']}")
        print(f"Failed entries: {stats['failed_entries']}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="CnakeCharmer DuckDB Client")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--api-url", help="CnakeCharmer API URL")
    parser.add_argument("--api-key", help="CnakeCharmer API key")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a batch of entries")
    process_parser.add_argument("--batch-size", type=int, help="Number of entries to process")
    process_parser.add_argument("--concurrent", type=int, help="Maximum concurrent requests")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of a batch")
    status_parser.add_argument("batch_id", help="ID of the batch")
    
    # Stats command
    subparsers.add_parser("stats", help="Show processing statistics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "process":
        asyncio.run(process_command(args))
    elif args.command == "status":
        asyncio.run(status_command(args))
    elif args.command == "stats":
        asyncio.run(stats_command(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()