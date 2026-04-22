"""
Logging service (non-prod version).

Application Insights / OpenTelemetry telemetry is DISABLED in this build.
Only plain console logging is configured. The public Logger API is kept
unchanged so the rest of the codebase continues to work without edits.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class _NoopSpan:
    """Minimal stand-in for an OpenTelemetry span (telemetry disabled)."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class Logger:
    """
    Centralized logging service.

    NOTE: Application Insights / OpenTelemetry integration has been disabled
    for the non-prod build. All tracing / span methods are no-ops, and only
    console logging is active.
    """

    _configured: bool = False

    def __init__(self, name: str = "knowledge_assistant") -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self._configure_console_once()

    @classmethod
    def _configure_console_once(cls) -> None:
        if cls._configured:
            return

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True,
        )
        logging.getLogger(__name__).info(
            "Application Insights telemetry DISABLED (non-prod build) - console logging only"
        )
        cls._configured = True

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.logger.info(message, extra=extra or {}, **kwargs)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.logger.warning(message, extra=extra or {}, **kwargs)

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs: Any,
    ) -> None:
        self.logger.error(message, extra=extra or {}, exc_info=exc_info, **kwargs)

    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.logger.exception(message, extra=extra or {}, **kwargs)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.logger.debug(message, extra=extra or {}, **kwargs)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.logger.critical(message, extra=extra or {}, **kwargs)

    # ------------------------------------------------------------------
    # Telemetry helpers -- no-ops in non-prod build
    # ------------------------------------------------------------------

    def add_span_attributes(self, **_attributes: Any) -> None:
        """No-op: telemetry disabled."""
        return

    @contextmanager
    def trace_operation(self, operation_name: str, **_attributes: Any):
        """
        No-op tracer. Measures duration locally and logs on error,
        but does NOT emit OpenTelemetry spans.
        """
        start_time = time.time()
        try:
            yield _NoopSpan()
        except Exception:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.debug(
                f"Operation '{operation_name}' failed after {duration_ms:.1f}ms"
            )
            raise

    def log_operation_start(self, operation: str, **context: Any) -> None:
        extra: Dict[str, Any] = {"operation": operation, "status": "started", **context}
        self.info(f"Starting operation: {operation}", extra=extra)

    def log_operation_complete(self, operation: str, **context: Any) -> None:
        extra: Dict[str, Any] = {"operation": operation, "status": "completed", **context}
        self.info(f"Completed operation: {operation}", extra=extra)

    def log_operation_failed(self, operation: str, error: Exception, **context: Any) -> None:
        extra: Dict[str, Any] = {
            "operation": operation,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context,
        }
        self.error(f"Failed operation: {operation}", extra=extra, exc_info=True)


def create_logger(name: str) -> Logger:
    """Create a Logger instance with a custom name."""
    return Logger(name)
