"""
Custom exceptions for the ephemeral_runner package.
"""


class EphemeralRunnerError(Exception):
    """Base exception for all ephemeral_runner errors."""

    pass


class VenvCreationError(EphemeralRunnerError):
    """Exception raised when virtual environment creation fails."""

    pass


class FileWriteError(EphemeralRunnerError):
    """Exception raised when writing a file fails."""

    pass


class CompilationError(EphemeralRunnerError):
    """Exception raised when code compilation fails."""

    pass


class ExecutionError(EphemeralRunnerError):
    """Exception raised when code execution fails."""

    pass


class DependencyError(EphemeralRunnerError):
    """Exception raised when dependency installation fails."""

    pass


class ParseError(EphemeralRunnerError):
    """Exception raised when code parsing fails."""

    pass
