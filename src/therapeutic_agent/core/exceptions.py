"""Custom exceptions for the therapeutic agent application."""

from typing import Any


class TherapeuticAgentException(Exception):
    """Base exception for therapeutic agent errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SafetyViolationError(TherapeuticAgentException):
    """Raised when content violates safety policies."""

    pass


class SessionNotFoundError(TherapeuticAgentException):
    """Raised when requested session cannot be found."""

    pass


class SessionLimitExceededError(TherapeuticAgentException):
    """Raised when user exceeds maximum active sessions."""

    pass


class AnthropicAPIError(TherapeuticAgentException):
    """Raised when Anthropic API calls fail."""

    pass


class ConfigurationError(TherapeuticAgentException):
    """Raised when application configuration is invalid."""

    pass


class DatabaseConnectionError(TherapeuticAgentException):
    """Raised when database connection fails."""

    pass


class RateLimitExceededError(TherapeuticAgentException):
    """Raised when rate limits are exceeded."""

    pass
