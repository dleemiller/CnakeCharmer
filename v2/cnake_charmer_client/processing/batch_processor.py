# processing/batch_processor.py
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator

from connectors.duckdb_connector import DuckDBConnector
from connectors.api_client import CnakeCharmerClient
from core.models import (
    CythonEntry, ProcessingResult, BatchJob, BatchStatus, 
    ProcessingStatus, ApiResponse
)
from core.exceptions import BatchProcessingError, EntryProcessingError
from utils.rate_limiter import RateLimiter


class BatchProcessor:
    """Processor for batch processing of Cython code."""
    
    def __init__(self, db_connector: DuckDBConnector, api_client: CnakeCharmerClient, 
                 max_concurrent: int = 5, requests_per_minute: int = 60):
        """
        Initialize batch processor.
        
        Args:
            db_connector: DuckDB connector
            api_client: CnakeCharmer API client
            max_concurrent: Maximum number of concurrent requests
            requests_per_minute: Maximum requests per minute
        """
        self.db = db_connector
        self.api = api_client
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(requests_per_minute)
        
        # Keep track of running tasks
        self.running_tasks = set()
        self.batch_status = {}
    
    async def process_batch(self, batch_size: int, options: Dict[str, Any] = None) -> str:
        """
        Process a batch of entries.
        
        Args:
            batch_size: Number of entries to process
            options: Processing options
            
        Returns:
            Batch ID
        """
        try:
            # Get unprocessed entries
            entries = self.db.get_unprocessed_entries(batch_size)
            
            if not entries:
                self.logger.info("No unprocessed entries found")
                return None
            
            # Create batch record
            batch_id = self.db.create_batch_record(len(entries), options)
            self.logger.info(f"Created batch {batch_id} with {len(entries)} entries")
            
            # Store batch status for monitoring
            self.batch_status[batch_id] = {
                'total': len(entries),
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'start_time': time.time()
            }
            
            # Process entries concurrently with rate limiting
            queue = asyncio.Queue()
            for entry in entries:
                await queue.put(entry)
            
            # Start consumer tasks
            consumers = [
                asyncio.create_task(self._process_queue_item(queue, batch_id))
                for _ in range(min(self.max_concurrent, len(entries)))
            ]
            
            # Wait for all entries to be processed
            await queue.join()
            
            # Cancel consumers
            for c in consumers:
                c.cancel()
            
            # Update batch status to completed
            self.db.update_batch_status(batch_id, {
                'status': BatchStatus.COMPLETED,
                'completed_at': True
            })
            
            self.logger.info(f"Batch {batch_id} processing completed")
            
            # Return batch ID for monitoring
            return batch_id
        
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise BatchProcessingError(f"Error processing batch: {e}")
    
    async def _process_queue_item(self, queue: asyncio.Queue, batch_id: str):
        """
        Process items from the queue.
        
        Args:
            queue: Queue of entries to process
            batch_id: ID of the current batch
        """
        while True:
            entry = await queue.get()
            try:
                await self.process_entry(entry, batch_id)
            except Exception as e:
                self.logger.error(f"Error processing entry {entry.id}: {e}")
                # Log the error but don't stop processing
                self.db.log_processing_error(
                    entry_id=entry.id,
                    error_message=str(e),
                    batch_id=batch_id,
                    error_type="processing_error"
                )
            finally:
                queue.task_done()
    
    async def process_entry(self, entry: CythonEntry, batch_id: str) -> ProcessingResult:
        """
        Process a single entry.
        
        Args:
            entry: Entry to process
            batch_id: ID of the current batch
            
        Returns:
            ProcessingResult object
        """
        self.logger.debug(f"Processing entry {entry.id}")
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Submit to API
            response = await self.api.generate_python_from_cython(
                entry.code,
                metadata={'source_id': entry.id, 'batch_id': batch_id}
            )
            
            if not response.success:
                raise EntryProcessingError(response.error or "API request failed")
            
            request_id = response.request_id
            if not request_id:
                raise EntryProcessingError("No request ID returned")
            
            # Poll for results
            python_code = await self._poll_for_results(request_id)
            
            # Store result in database
            python_id = self.db.store_python_code(
                python_code=python_code,
                cython_id=entry.id,
                metadata={'source_id': entry.id, 'batch_id': batch_id}
            )
            
            # Mark entry as processed
            self.db.mark_as_processed(entry.id, True, python_id)
            
            # Update batch status
            self._update_batch_status(batch_id, success=True)
            
            return ProcessingResult(
                entry_id=entry.id,
                success=True,
                python_code=python_code,
                python_id=python_id
            )
        
        except Exception as e:
            self.logger.error(f"Error processing entry {entry.id}: {e}")
            
            # Mark entry as failed
            self.db.mark_as_processed(entry.id, False)
            
            # Log the error
            self.db.log_processing_error(
                entry_id=entry.id,
                error_message=str(e),
                batch_id=batch_id,
                error_type=type(e).__name__
            )
            
            # Update batch status
            self._update_batch_status(batch_id, success=False)
            
            return ProcessingResult(
                entry_id=entry.id,
                success=False,
                error=str(e)
            )
    
    async def _poll_for_results(self, request_id: str, max_polls: int = 30, poll_interval: int = 2) -> str:
        """
        Poll for code generation results.
        
        Args:
            request_id: ID of the request
            max_polls: Maximum number of polls
            poll_interval: Interval between polls in seconds
            
        Returns:
            Generated Python code
        """
        polls = 0
        
        while polls < max_polls:
            await asyncio.sleep(poll_interval)
            
            # Check status
            status_response = await self.api.get_generation_status(request_id)
            
            if not status_response.success:
                raise EntryProcessingError(f"Failed to get status: {status_response.error}")
            
            status = status_response.data.get('status')
            
            if status == 'completed':
                # Get results
                result_response = await self.api.get_generation_result(request_id)
                
                if not result_response.success:
                    raise EntryProcessingError(f"Failed to get results: {result_response.error}")
                
                # Extract Python code from response
                code = result_response.data.get('python_code')
                
                if not code:
                    raise EntryProcessingError("No Python code in results")
                
                return code
            
            elif status == 'failed':
                error = status_response.data.get('error', 'Unknown error')
                raise EntryProcessingError(f"Generation failed: {error}")
            
            polls += 1
        
        raise EntryProcessingError(f"Polling timeout after {max_polls} attempts")
    
    def _update_batch_status(self, batch_id: str, success: bool):
        """
        Update batch status in memory and database.
        
        Args:
            batch_id: ID of the batch
            success: Whether processing was successful
        """
        if batch_id in self.batch_status:
            self.batch_status[batch_id]['processed'] += 1
            
            if success:
                self.batch_status[batch_id]['successful'] += 1
            else:
                self.batch_status[batch_id]['failed'] += 1
            
            # Update database
            self.db.update_batch_status(batch_id, {
                'processed_entries': self.batch_status[batch_id]['processed'],
                'successful_entries': self.batch_status[batch_id]['successful'],
                'failed_entries': self.batch_status[batch_id]['failed']
            })
    
# processing/batch_processor.py (continued)
    async def monitor_progress(self, batch_id: str, interval: int = 2) -> AsyncIterator[Dict]:
        """
        Monitor and yield progress updates for a batch.
        
        Args:
            batch_id: ID of the batch
            interval: Interval between updates in seconds
            
        Yields:
            Progress update dictionaries
        """
        if batch_id not in self.batch_status:
            # Get batch status from database
            batch = self.db.get_batch_status(batch_id)
            
            if not batch:
                raise BatchProcessingError(f"Batch {batch_id} not found")
            
            self.batch_status[batch_id] = {
                'total': batch.total_entries,
                'processed': batch.processed_entries,
                'successful': batch.successful_entries,
                'failed': batch.failed_entries,
                'start_time': batch.started_at.timestamp() if batch.started_at else time.time()
            }
        
        while True:
            # Get latest status from database
            batch = self.db.get_batch_status(batch_id)
            
            if not batch:
                raise BatchProcessingError(f"Batch {batch_id} not found")
            
            status = self.batch_status[batch_id]
            
            # Calculate progress percentage
            progress = (status['processed'] / status['total']) * 100 if status['total'] > 0 else 0
            
            # Calculate elapsed time
            elapsed = time.time() - status['start_time']
            
            # Estimate remaining time
            remaining = None
            if progress > 0:
                remaining = (elapsed / progress) * (100 - progress)
            
            update = {
                'batch_id': batch_id,
                'total': status['total'],
                'processed': status['processed'],
                'successful': status['successful'],
                'failed': status['failed'],
                'progress': progress,
                'elapsed_time': elapsed,
                'estimated_remaining': remaining,
                'status': batch.status
            }
            
            yield update
            
            # If batch is complete, stop monitoring
            if batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
                break
            
            await asyncio.sleep(interval)
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel an in-progress batch.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            True if the batch was cancelled, False otherwise
        """
        # Update batch status in database
        success = self.db.update_batch_status(batch_id, {
            'status': BatchStatus.CANCELLED
        })
        
        if success and batch_id in self.batch_status:
            # Clean up in-memory status
            del self.batch_status[batch_id]
        
        return success