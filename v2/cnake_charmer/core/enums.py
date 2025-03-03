# core/enums.py
from enum import Enum, auto


class LanguageType(str, Enum):
    """Types of programming languages supported by the system."""
    PYTHON = "python"
    CYTHON = "cython"
    RUST = "rust"
    CPP = "cpp"
    C = "c"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed on code."""
    STATIC = "static"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    COMPLEXITY = "complexity"
    SECURITY = "security"


class RequestStatus(str, Enum):
    """Status of a code generation request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"