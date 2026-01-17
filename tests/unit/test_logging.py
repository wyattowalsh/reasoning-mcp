"""Tests for reasoning_mcp.logging module."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from reasoning_mcp.logging import LogContext, get_logger, logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Test setup_logging returns a logger."""
        result = setup_logging()
        assert isinstance(result, logging.Logger)
        assert result.name == "reasoning_mcp"

    def test_sets_log_level(self):
        """Test setup_logging sets the log level."""
        log = setup_logging(log_level="DEBUG")
        assert log.level == logging.DEBUG

        log = setup_logging(log_level="WARNING")
        assert log.level == logging.WARNING

    def test_case_insensitive_level(self):
        """Test log level is case insensitive."""
        log = setup_logging(log_level="debug")
        assert log.level == logging.DEBUG

    def test_adds_console_handler(self):
        """Test setup_logging adds a console handler."""
        log = setup_logging()
        console_handlers = [h for h in log.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1

    def test_file_handler(self):
        """Test setup_logging can add a file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log = setup_logging(log_file=log_file)

            file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1

            # Test writing to file
            log.info("Test message")
            assert log_file.exists()

    def test_creates_log_directory(self):
        """Test setup_logging creates log file directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "test.log"
            setup_logging(log_file=log_file)
            assert log_file.parent.exists()

    def test_clears_existing_handlers(self):
        """Test setup_logging clears existing handlers."""
        setup_logging()
        initial_count = len(logger.handlers)
        setup_logging()  # Call again
        assert len(logger.handlers) == initial_count  # Should not accumulate

    def test_prevents_propagation(self):
        """Test logger does not propagate to root logger."""
        log = setup_logging()
        assert log.propagate is False


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_child_logger(self):
        """Test get_logger returns a child logger."""
        child = get_logger("test_module")
        assert child.name == "reasoning_mcp.test_module"

    def test_already_prefixed_name(self):
        """Test get_logger handles already prefixed names."""
        child = get_logger("reasoning_mcp.submodule")
        assert child.name == "reasoning_mcp.submodule"


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_changes_level_temporarily(self):
        """Test LogContext changes level temporarily."""
        setup_logging(log_level="INFO")
        original_level = logger.level

        with LogContext(level="DEBUG") as ctx_logger:
            assert ctx_logger.level == logging.DEBUG

        # Should be restored
        assert logger.level == original_level

    def test_restores_level_on_exception(self):
        """Test LogContext restores level even on exception."""
        setup_logging(log_level="INFO")
        original_level = logger.level

        try:
            with LogContext(level="DEBUG"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still be restored
        assert logger.level == original_level

    def test_custom_logger_name(self):
        """Test LogContext with custom logger name."""
        custom_logger = logging.getLogger("custom_test")
        custom_logger.setLevel(logging.INFO)

        with LogContext(level="DEBUG", logger_name="custom_test") as ctx_logger:
            assert ctx_logger.level == logging.DEBUG

        # Restore check
        assert custom_logger.level == logging.INFO


class TestModuleLogger:
    """Tests for module-level logger."""

    def test_logger_exists(self):
        """Test module-level logger is accessible."""
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "reasoning_mcp"
