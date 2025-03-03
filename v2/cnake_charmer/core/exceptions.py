# core/exceptions.py
class CnakeCharmerError(Exception):
    """Base exception for all CnakeCharmer errors."""
    pass


class ConfigurationError(CnakeCharmerError):
    """Raised when there's an error in the configuration."""
    pass


class DatabaseError(CnakeCharmerError):
    """Raised when there's a database error."""
    pass


class CodeGenerationError(CnakeCharmerError):
    """Raised when there's an error generating code."""
    pass


class BuildError(CnakeCharmerError):
    """Raised when there's an error building code."""
    pass


class AnalysisError(CnakeCharmerError):
    """Raised when there's an error analyzing code."""
    pass


class EquivalencyError(CnakeCharmerError):
    """Raised when there's an error checking equivalency."""
    pass


class NotFoundError(CnakeCharmerError):
    """Raised when a requested resource is not found."""
    pass


class ValidationError(CnakeCharmerError):
    """Raised when input validation fails."""
    pass