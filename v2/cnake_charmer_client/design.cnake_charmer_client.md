  # DuckDB Processing Client Design

## 1. Overview

The DuckDB Processing Client is a standalone application designed to process existing Cython code from a DuckDB database through the CnakeCharmer API. It handles batch processing, tracks progress, and stores results back to the DuckDB database.

## 2. Directory Structure

```
cnake_charmer_client/
├── __init__.py
├── main.py                     # Entry point
├── core/
│   ├── __init__.py
│   ├── config.py               # Configuration handling
│   ├── models.py               # Data models
│   └── exceptions.py           # Custom exceptions
│
├── connectors/
│   ├── __init__.py
│   ├── duckdb_connector.py     # DuckDB connection handling
│   └── api_client.py           # CnakeCharmer API client
│
├── processing/
│   ├── __init__.py
│   ├── batch_processor.py      # Batch processing logic
│   ├── result_handler.py       # Process and store results
│   └── retry_manager.py        # Handle retries for failed requests
│
├── monitoring/
│   ├── __init__.py
│   ├── progress_tracker.py     # Track processing progress
│   ├── status_reporter.py      # Generate status reports
│   └── logger.py               # Custom logging
│
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py         # API rate limiting
│   └── error_analyzer.py       # Analyze errors for patterns
│
└── cli/
    ├── __init__.py
    └── commands.py             # CLI commands
```

## 3. Key Components

### 3.1. DuckDB Connector

Responsible for interacting with the DuckDB database:

```python
class DuckDBConnector:
    def __init__(self, db_path: str):
        """Initialize DuckDB connection."""
        
    def get_unprocessed_entries(self, limit: int, offset: int) -> List[Dict]:
        """Get unprocessed Cython code entries."""
        
    def get_entry_by_id(self, entry_id: str) -> Optional[Dict]:
        """Get a specific entry by ID."""
        
    def mark_as_processed(self, entry_id: str, success: bool, result_id: Optional[str] = None) -> bool:
        """Mark an entry as processed with result status."""
        
    def store_python_code(self, entry_id: str, python_code: str, metadata: Dict) -> str:
        """Store generated Python code with metadata."""
        
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about processing progress."""
        
    def create_batch_record(self, batch_size: int, options: Dict) -> str:
        """Create a record for a new batch processing job."""
        
    def update_batch_status(self, batch_id: str, updates: Dict) -> bool:
        """Update status of a batch processing job."""
```

### 3.2. API Client

Handles communication with the CnakeCharmer API:

```python
class CnakeCharmerClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize API client."""
        
    async def generate_python_from_cython(self, cython_code: str, metadata: Dict) -> Dict:
        """Request Python code generation from Cython."""
        
    async def get_generation_status(self, request_id: str) -> Dict:
        """Check status of a code generation request."""
        
    async def get_generation_result(self, request_id: str) -> Dict:
        """Get the result of a completed generation request."""
```

### 3.3. Batch Processor

Coordinates the batch processing workflow:

```python
class BatchProcessor:
    def __init__(self, db_connector: DuckDBConnector, api_client: CnakeCharmerClient, config: Dict):
        """Initialize batch processor."""
        
    async def process_batch(self, batch_size: int, options: Dict) -> str:
        """Process a batch of entries."""
        
    async def process_entry(self, entry: Dict) -> Dict:
        """Process a single entry."""
        
    async def monitor_progress(self, batch_id: str) -> AsyncIterator[Dict]:
        """Monitor and yield progress updates for a batch."""
        
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel an in-progress batch."""
```

### 3.4. Status Reporter

Generates reports on processing status:

```python
class StatusReporter:
    def __init__(self, db_connector: DuckDBConnector):
        """Initialize status reporter."""
        
    def generate_summary_report(self, batch_id: Optional[str] = None) -> Dict:
        """Generate a summary report of processing status."""
        
    def generate_detailed_report(self, batch_id: str) -> Dict:
        """Generate a detailed report for a specific batch."""
        
    def generate_error_report(self, batch_id: Optional[str] = None) -> Dict:
        """Generate a report of processing errors."""
```

## 4. Workflow

### 4.1. Batch Processing Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. Configure    │────▶│ 2. Create Batch │────▶│ 3. Fetch        │
│    Settings     │     │    Record       │     │    Entries      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 6. Store        │◀────│ 5. Process      │◀────│ 4. Submit to    │
│    Results      │     │    Response     │     │    API          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │               ┌─────────────────┐             │
         └──────────────▶│ 7. Update       │◀────────────┘
                         │    Status       │
                         └─────────────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │ 8. Generate     │
                         │    Reports      │
                         └─────────────────┘
```

### 4.2. Entry Processing States

- **Pending**: Initial state, not yet submitted
- **Submitted**: Sent to the API, awaiting response
- **Processing**: Being processed by the API
- **Completed**: Successfully processed
- **Failed**: Failed to process
- **Retrying**: Failed but being retried

## 5. Command Line Interface

The client will provide a command-line interface for batch processing:

```
# Start a new batch processing job
python -m cnake_charmer_client process --batch-size 100 --db-path /path/to/db.duckdb

# Check status of a batch
python -m cnake_charmer_client status --batch-id BATCH123

# Generate a report
python -m cnake_charmer_client report --batch-id BATCH123 --output report.json

# Resume a failed or cancelled batch
python -m cnake_charmer_client resume --batch-id BATCH123

# Show statistics
python -m cnake_charmer_client stats
```

## 6. Configuration Options

Configuration can be specified via a config file, environment variables, or command-line arguments:

```python
{
    "api": {
        "base_url": "http://localhost:8000/api",
        "api_key": "your-api-key",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 5
    },
    "duckdb": {
        "db_path": "./cython_code.duckdb",
        "table_name": "cython_entries",
        "result_table": "python_entries"
    },
    "processing": {
        "batch_size": 50,
        "max_concurrent": 5,
        "rate_limit": {
            "requests_per_minute": 60
        }
    },
    "logging": {
        "level": "INFO",
        "file": "./processing.log"
    }
}
```

## 7. Error Handling and Retry Logic

The client implements robust error handling and retry logic:

1. **Transient Errors**: Automatically retry with exponential backoff
2. **Permanent Errors**: Log and continue with next entry
3. **Rate Limiting**: Respect API rate limits, adjust as needed
4. **Batch Resumption**: Ability to resume from where processing stopped

## 8. Monitoring and Logging

The client provides comprehensive monitoring and logging:

1. **Progress Tracking**: Real-time updates on batch progress
2. **Success Rate**: Track success/failure ratio
3. **Performance Metrics**: Track API response times
4. **Error Aggregation**: Identify common error patterns
5. **Detailed Logging**: Configurable logging levels

## 9. DuckDB Schema

The DuckDB database should have the following schema:

```sql
-- Table for Cython code entries
CREATE TABLE cython_entries (
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
);

-- Table for generated Python code
CREATE TABLE python_entries (
    id VARCHAR PRIMARY KEY,
    code TEXT NOT NULL,
    cython_id VARCHAR,
    metadata JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for batch processing records
CREATE TABLE batch_jobs (
    id VARCHAR PRIMARY KEY,
    total_entries INTEGER NOT NULL,
    processed_entries INTEGER DEFAULT 0,
    successful_entries INTEGER DEFAULT 0,
    failed_entries INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'running',
    options JSON,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Table for errors encountered during processing
CREATE TABLE processing_errors (
    id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    batch_id VARCHAR,
    error_message TEXT,
    error_type VARCHAR,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 10. Implementation Path

1. **Core Models and Config**: Define data models and configuration handling
2. **DuckDB Integration**: Implement DuckDB connector and schema creation
3. **API Client**: Develop the API client for CnakeCharmer
4. **Batch Processing Logic**: Implement the batch processing workflow
5. **CLI Commands**: Create the command-line interface
6. **Error Handling**: Add comprehensive error handling and retry logic
7. **Monitoring**: Implement progress tracking and reporting
8. **Performance Optimization**: Optimize for throughput and reliability  