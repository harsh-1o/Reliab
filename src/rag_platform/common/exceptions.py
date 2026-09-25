"""Core platform exception hierarchy."""


class PlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProvenanceError(PlatformError):
    """Raised when run provenance or reproducibility verification fails."""
    pass


class ImmutabilityError(PlatformError):
    """Raised when attempting to mutate a published or completed entity."""
    pass


class AdapterTimeoutError(PlatformError):
    """Raised when SUT adapter times out (classified as OPS-01)."""
    pass


class AdapterExecutionError(PlatformError):
    """Raised when SUT adapter encounters connection/runtime failures."""
    pass


class PolicyViolationError(PlatformError):
    """Raised when an operation violates configured release or budget policy."""
    pass
