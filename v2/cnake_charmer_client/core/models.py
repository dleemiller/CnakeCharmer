# core/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class BatchStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class CythonEntry:
    """Represents a Cython code entry from the DuckDB database."""
    id: str
    code: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    processed: bool = False
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    python_id: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


@dataclass
class PythonEntry:
    """Represents a generated Python code entry."""
    id: str
    code: str
    cython_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[datetime] = None


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    id: str
    total_entries: int
    processed_entries: int = 0
    successful_entries: int = 0
    failed_entries: int = 0
    status: BatchStatus = BatchStatus.RUNNING
    options: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ProcessingError:
    """Represents an error encountered during processing."""
    id: str
    entry_id: str
    batch_id: Optional[str] = None
    error_message: str = ""
    error_type: str = "unknown"
    occurred_at: Optional[datetime] = None


@dataclass
class ApiResponse:
    """Represents a response from the CnakeCharmer API."""
    success: bool
    request_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Represents the result of processing a single entry."""
    entry_id: str
    success: bool
    python_code: Optional[str] = None
    python_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchReport:
    """Represents a status report for a batch job."""
    batch_id: str
    status: BatchStatus
    total_entries: int
    processed_entries: int
    successful_entries: int
    failed_entries: int
    progress_percentage: float
    elapsed_time: float
    estimated_time_remaining: Optional[float] = None
    recent_errors: List[ProcessingError] = field(default_factory=list)