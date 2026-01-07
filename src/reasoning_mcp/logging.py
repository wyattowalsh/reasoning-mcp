"""Logging configuration for reasoning-mcp server.

This module provides structured logging setup with support for
different log levels, formats, and output handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings


# Package logger
logger = logging.getLogger("reasoning_mcp")


def setup_logging(
    settings: Settings | None = None,
    *,
    log_level: str | None = None,
    log_format: str | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    """Set up logging for the reasoning-mcp server.

    Args:
        settings: Optional Settings instance. If not provided,
            uses get_settings() or defaults.
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Override log format string.
        log_file: Optional path to log file for file logging.

    Returns:
        The configured root logger for reasoning_mcp.

    Example:
        >>> from reasoning_mcp.logging import setup_logging
        >>> logger = setup_logging(log_level="DEBUG")
        >>> logger.info("Server starting")
    """
    # Get settings if not provided
    if settings is None:
        try:
            from reasoning_mcp.config import get_settings
            settings = get_settings()
            effective_level = log_level or settings.log_level
            effective_format = log_format or settings.log_format
        except ImportError:
            effective_level = log_level or "INFO"
            effective_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        effective_level = log_level or settings.log_level
        effective_format = log_format or settings.log_format

    # Convert string level to logging constant
    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)

    # Configure the package logger
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(effective_format)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: The name of the module or component.

    Returns:
        A logger instance that is a child of the package logger.

    Example:
        >>> from reasoning_mcp.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing request")
    """
    if name.startswith("reasoning_mcp."):
        return logging.getLogger(name)
    return logging.getLogger(f"reasoning_mcp.{name}")


class LogContext:
    """Context manager for temporary log level changes.

    Example:
        >>> with LogContext(level="DEBUG"):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This might not be logged")
    """

    def __init__(
        self,
        level: str = "DEBUG",
        logger_name: str = "reasoning_mcp",
    ) -> None:
        """Initialize the log context.

        Args:
            level: The temporary log level.
            logger_name: Name of the logger to modify.
        """
        self.level = level
        self.logger_name = logger_name
        self._original_level: int | None = None
        self._logger: logging.Logger | None = None

    def __enter__(self) -> logging.Logger:
        """Enter the context and set temporary log level."""
        self._logger = logging.getLogger(self.logger_name)
        self._original_level = self._logger.level
        numeric_level = getattr(logging, self.level.upper(), logging.DEBUG)
        self._logger.setLevel(numeric_level)
        return self._logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore original log level."""
        if self._logger is not None and self._original_level is not None:
            self._logger.setLevel(self._original_level)


__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "LogContext",
]
