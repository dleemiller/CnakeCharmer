# core/exceptions.py
class CnakeCharmerClientError(Exception):
    """Base exception for all client errors."""
    pass


class ConfigurationError(CnakeCharmerClientError):
    """Raised when there's an error in the configuration."""
    pass


class DuckDBConnectionError(CnakeCharmerClientError):
    """Raised when there's an error connecting to DuckDB."""
    pass


class ApiConnectionError(CnakeCharmerClientError):
    """Raised when there's an error connecting to the API."""
    pass


class ApiResponseError(CnakeCharmerClientError):
    """Raised when there's an error in the API response."""
    pass


class EntryProcessingError(CnakeCharmerClientError):
    """Raised when there's an error processing an entry."""
    pass


class BatchProcessingError(CnakeCharmerClientError):
    """Raised when there's an error processing a batch."""
    pass


class RetryError(CnakeCharmerClientError):
    """Raised when the maximum number of retries is exceeded."""
    pass


class RateLimitError(CnakeCharmerClientError):
    """Raised when the API rate limit is exceeded."""
    pass