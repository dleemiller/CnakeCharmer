# Enhanced CnakeCharmer System Design Document

## 1. System Overview

CnakeCharmer is a service that generates, analyzes, and optimizes code in multiple languages, with a primary focus on Python and Cython equivalency. The system accepts natural language prompts, generates code, builds it, analyzes it, and provides feedback for improvements.

### 1.1. Core Features

- Generate equivalent implementations in multiple languages (Python, Cython, etc.)
- Build and run the generated code
- Analyze code quality, performance, and security
- Check equivalency between implementations
- Provide feedback for code improvement
- Support iterative refinement based on analysis results
- **Process existing Cython code from DuckDB to generate Python equivalents**
- **Track progress of batch processing operations**

### 1.2. System Architecture

```
┌─────────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────────┐
│   FastAPI   │────▶│   Queue   │────▶│   Workers   │────▶│   Analyzers  │
│  Endpoints  │     │  (Celery) │     │  (Builders) │     │              │
└─────────────┘     └───────────┘     └─────────────┘     └──────────────┘
       ▲                                     │                    │
       │                                     │                    │
       │                                     ▼                    ▼
┌─────────────┐     ┌───────────────────────────────────────────────────┐
│  Database   │◀───▶│            Event-Driven Feedback Loop             │
└─────────────┘     └───────────────────────────────────────────────────┘
       ▲
       │
┌─────────────┐     ┌───────────┐
│  Batch      │────▶│  DuckDB   │
│  Processor  │     │  Storage  │
└─────────────┘     └───────────┘
```

## 2. Core Domain Models

[... existing domain models remain the same ...]

### 2.5. Data Source Models

```python
# Information about a Cython code entry from DuckDB
class CythonDatasetEntry:
    id: str                            # Unique identifier for the entry
    code: str                          # Cython code
    metadata: Dict[str, Any]           # Additional metadata
    description: Optional[str] = None  # Description of what the code does
    processed: bool = False            # Whether this entry has been processed
    tags: List[str] = None             # Tags for categorization
    python_id: Optional[str] = None    # Reference to generated Python equivalent

# Batch processing status
class BatchProcessingStatus:
    batch_id: str                      # Unique identifier for the batch
    total_entries: int                 # Total number of entries in the batch
    processed_count: int               # Number of processed entries
    successful_count: int              # Number of successfully processed entries
    failed_count: int                  # Number of failed entries
    started_at: datetime               # When processing started
    completed_at: Optional[datetime] = None  # When processing completed
    status: str                        # "running", "completed", "failed"
```

## 3. Core Protocol Interfaces

[... existing protocol interfaces remain the same ...]

### 3.5. Data Source Interface

```python
class DataSource(Protocol):
    def get_entries(self, limit: int, offset: int) -> List[CythonDatasetEntry]:
        """Get a batch of entries from the data source."""
        ...
    
    def get_entry(self, id: str) -> Optional[CythonDatasetEntry]:
        """Get a specific entry by ID."""
        ...
    
    def mark_as_processed(self, id: str, success: bool, result_id: Optional[str] = None) -> bool:
        """Mark an entry as processed."""
        ...
    
    def get_processing_status(self) -> Dict[str, int]:
        """Get overall processing status."""
        ...
```

### 3.6. Batch Processor Interface

```python
class BatchProcessor(Protocol):
    def start_batch(self, batch_size: int, options: Dict[str, Any]) -> str:
        """Start a new batch processing job."""
        ...
    
    def get_batch_status(self, batch_id: str) -> BatchProcessingStatus:
        """Get the status of a batch processing job."""
        ...
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch processing job."""
        ...
    
    def resume_batch(self, batch_id: str) -> bool:
        """Resume a previously stopped batch."""
        ...
```

## 4. Repository Interfaces

[... existing repository interfaces remain the same ...]

### 4.3. Dataset Repository Interfaces

```python
class DatasetEntryRepository(Protocol):
    def get_entries(self, limit: int, offset: int, filters: Dict[str, Any]) -> List[CythonDatasetEntry]:
        """Get dataset entries based on filters."""
        ...
    
    def get_entry_by_id(self, id: str) -> Optional[CythonDatasetEntry]:
        """Get a specific entry by ID."""
        ...
    
    def update_entry_status(self, id: str, status: str, result_id: Optional[str] = None) -> bool:
        """Update the processing status of an entry."""
        ...
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the dataset entries."""
        ...
```

```python
class BatchStatusRepository(Protocol):
    def create_batch(self, batch_size: int, options: Dict[str, Any]) -> str:
        """Create a new batch processing record."""
        ...
    
    def update_batch_status(self, batch_id: str, status_updates: Dict[str, Any]) -> bool:
        """Update the status of a batch."""
        ...
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchProcessingStatus]:
        """Get the status of a specific batch."""
        ...
    
    def get_all_batches(self, limit: int, offset: int) -> List[BatchProcessingStatus]:
        """Get a list of all batch processing jobs."""
        ...
```

## 5. Service Layer

[... existing service layer remains the same ...]

### 5.5. Dataset Service

Manages interactions with the DuckDB dataset, tracking processing status, and coordinating batch operations.

### 5.6. Batch Processing Service

Handles batch processing of Cython code, managing job queues, and tracking progress.

## 6. API Layer

[... existing API endpoints remain the same ...]

### 6.4. Dataset Processing Endpoints

```
POST /api/dataset/process-batch
GET /api/dataset/batch/{batch_id}
GET /api/dataset/statistics
GET /api/dataset/entries
```

## 7. Worker Layer

[... existing worker tasks remain the same ...]

### 7.4. Dataset Processing Tasks

Tasks for processing Cython code from the dataset and generating Python equivalents.

## 8. Implementation Strategy

[... existing implementation steps 1-8 remain the same ...]

9. Implement DuckDB integration
10. Create batch processing service
11. Add dataset processing endpoints
12. Implement processing status tracking
13. Develop visualization tools for dataset statistics

## 9. Database Schema

[... existing schema tables remain the same ...]

```sql
-- Dataset entries
CREATE TABLE dataset_entries (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    metadata JSONB,
    description TEXT,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    tags JSONB,
    python_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch processing jobs
CREATE TABLE batch_jobs (
    batch_id TEXT PRIMARY KEY,
    total_entries INT NOT NULL,
    processed_count INT NOT NULL DEFAULT 0,
    successful_count INT NOT NULL DEFAULT 0,
    failed_count INT NOT NULL DEFAULT 0,
    options JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch entry mappings
CREATE TABLE batch_entries (
    id SERIAL PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batch_jobs(batch_id),
    entry_id TEXT NOT NULL REFERENCES dataset_entries(id),
    status TEXT NOT NULL DEFAULT 'pending',
    processed_at TIMESTAMP,
    error_message TEXT,
    result_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(batch_id, entry_id)
);
```

## 10. Batch Processing Client Application

To facilitate processing the DuckDB dataset through the CnakeCharmer API, we'll develop a client application:

### 10.1. Key Features

1. **Connection Management**: Connect to both DuckDB and CnakeCharmer API
2. **Batch Processing**: Process entries in configurable batch sizes
3. **Rate Limiting**: Control API request rates to prevent overloading
4. **Status Tracking**: Monitor and report on processing status
5. **Error Handling**: Retry failed entries and log errors
6. **Result Storage**: Store generated Python code back to DuckDB

### 10.2. Client Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Batch Processing Client                │
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    DuckDB   │◀──▶│   Process   │◀──▶│    API      │  │
│  │  Connector  │    │  Controller │    │   Client    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                            │                            │
│                     ┌──────────────┐                    │
│                     │     Status   │                    │
│                     │    Reporter  │                    │
│                     └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
           │                                     │
           ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────┐
│     DuckDB File     │              │   CnakeCharmer API  │
└─────────────────────┘              └─────────────────────┘
```

### 10.3. Processing Flow

1. **Initialization**:
   - Connect to DuckDB
   - Authenticate with CnakeCharmer API
   - Create a new batch job

2. **Processing Loop**:
   - Fetch unprocessed entries from DuckDB
   - Submit entries to the API for processing
   - Monitor status of submitted entries
   - Update DuckDB with results
   - Report progress

3. **Completion**:
   - Generate final report
   - Update all statistics
   - Close connections

### 10.4. Configuration Options

- Batch size
- API endpoint URL
- Request rate limits
- Retry settings
- Logging options
- Filtering criteria for DuckDB entries

## 11. Monitoring and Visualization

To track the progress of dataset processing, we'll implement monitoring and visualization:

### 11.1. Dashboard Features

1. **Overall Progress**: Visual representation of processing progress
2. **Success Rates**: Charts showing success vs. failure rates
3. **Error Analysis**: Categorization and visualization of common errors
4. **Performance Metrics**: Processing times and throughput
5. **Dataset Insights**: Statistics and patterns from the dataset

### 11.2. Monitoring Endpoints

```
GET /api/monitoring/dashboard
GET /api/monitoring/batch-progress/{batch_id}
GET /api/monitoring/error-analysis
GET /api/monitoring/performance
```

## 12. Future Enhancements

[... existing enhancements remain the same ...]

6. **Dataset Expansion**: Support for more languages and frameworks
7. **Code Similarity Analysis**: Detect and group similar code patterns
8. **Automated Dataset Labeling**: Use ML to categorize and tag dataset entries
9. **Export Capabilities**: Export processed results in various formats
10. **Integration with Code Libraries**: Link with popular code repositories and libraries