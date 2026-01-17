"""Tests for CLI main module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from reasoning_mcp.cli.main import (
    CLIContext,
    app,
    get_cli_context,
    get_settings_from_config,
    main,
    setup_logging_from_verbosity,
    version_callback,
)
from reasoning_mcp.config import Settings
from reasoning_mcp.registry import MethodRegistry

if TYPE_CHECKING:
    pass


runner = CliRunner()


class TestCLIContext:
    """Tests for CLIContext class."""

    def test_cli_context_initialization(self) -> None:
        """Test CLIContext initializes with all required attributes."""
        settings = MagicMock(spec=Settings)
        logger = MagicMock(spec=logging.Logger)
        registry = MagicMock(spec=MethodRegistry)

        ctx = CLIContext(
            settings=settings,
            logger=logger,
            registry=registry,
            verbose=2,
        )

        assert ctx.settings is settings
        assert ctx.logger is logger
        assert ctx.registry is registry
        assert ctx.verbose == 2

    def test_cli_context_default_verbose(self) -> None:
        """Test CLIContext uses default verbose=0."""
        settings = MagicMock(spec=Settings)
        logger = MagicMock(spec=logging.Logger)
        registry = MagicMock(spec=MethodRegistry)

        ctx = CLIContext(
            settings=settings,
            logger=logger,
            registry=registry,
        )

        assert ctx.verbose == 0


class TestGetCLIContext:
    """Tests for get_cli_context function."""

    def test_get_cli_context_raises_when_not_initialized(self) -> None:
        """Test get_cli_context raises Exit when context not set."""
        # Reset the global context
        import reasoning_mcp.cli.main as main_module

        original = main_module._cli_context
        main_module._cli_context = None

        try:
            with pytest.raises(typer.Exit) as exc_info:
                get_cli_context()
            assert exc_info.value.exit_code == 1
        finally:
            main_module._cli_context = original


class TestGetSettingsFromConfig:
    """Tests for get_settings_from_config function."""

    def test_returns_default_settings_when_no_config_path(self) -> None:
        """Test returns default settings when config_path is None."""
        with patch("reasoning_mcp.cli.main.get_settings") as mock_get:
            mock_settings = MagicMock(spec=Settings)
            mock_get.return_value = mock_settings

            result = get_settings_from_config(None)

            assert result is mock_settings
            mock_get.assert_called_once()

    def test_raises_exit_for_nonexistent_config_file(self, tmp_path: Path) -> None:
        """Test raises Exit when config file does not exist."""
        nonexistent = str(tmp_path / "nonexistent.env")

        with pytest.raises(typer.Exit) as exc_info:
            get_settings_from_config(nonexistent)
        assert exc_info.value.exit_code == 1

    def test_raises_exit_for_directory_path(self, tmp_path: Path) -> None:
        """Test raises Exit when config path is a directory."""
        with pytest.raises(typer.Exit) as exc_info:
            get_settings_from_config(str(tmp_path))
        assert exc_info.value.exit_code == 1

    def test_raises_exit_for_invalid_config_file(self, tmp_path: Path) -> None:
        """Test raises Exit when config file cannot be loaded."""
        config_file = tmp_path / "bad.env"
        config_file.write_text("not = valid = env = file\n")

        # Patch at the source since Settings is imported inside the function
        with patch("reasoning_mcp.config.Settings", side_effect=Exception("parse error")):
            with pytest.raises(typer.Exit) as exc_info:
                get_settings_from_config(str(config_file))
            assert exc_info.value.exit_code == 1


class TestSetupLoggingFromVerbosity:
    """Tests for setup_logging_from_verbosity function."""

    def test_verbosity_zero_uses_settings_level(self) -> None:
        """Test verbosity=0 uses the settings log level."""
        settings = MagicMock(spec=Settings)
        settings.log_level = "WARNING"

        with patch("reasoning_mcp.cli.main.setup_logging") as mock_setup:
            mock_logger = MagicMock(spec=logging.Logger)
            mock_setup.return_value = mock_logger

            result = setup_logging_from_verbosity(0, settings)

            assert result is mock_logger
            mock_setup.assert_called_once_with(settings=settings, log_level="WARNING")

    def test_verbosity_one_uses_debug(self) -> None:
        """Test verbosity=1 uses DEBUG level."""
        settings = MagicMock(spec=Settings)
        settings.log_level = "INFO"

        with patch("reasoning_mcp.cli.main.setup_logging") as mock_setup:
            mock_logger = MagicMock(spec=logging.Logger)
            mock_setup.return_value = mock_logger

            result = setup_logging_from_verbosity(1, settings)

            assert result is mock_logger
            mock_setup.assert_called_once_with(settings=settings, log_level="DEBUG")

    def test_verbosity_two_uses_debug(self) -> None:
        """Test verbosity=2 also uses DEBUG level."""
        settings = MagicMock(spec=Settings)
        settings.log_level = "INFO"

        with patch("reasoning_mcp.cli.main.setup_logging") as mock_setup:
            mock_logger = MagicMock(spec=logging.Logger)
            mock_setup.return_value = mock_logger

            result = setup_logging_from_verbosity(2, settings)

            assert result is mock_logger
            mock_setup.assert_called_once_with(settings=settings, log_level="DEBUG")


class TestVersionCallback:
    """Tests for version_callback function."""

    def test_version_callback_false_does_nothing(self) -> None:
        """Test version_callback does nothing when value is False."""
        # Should not raise
        version_callback(False)

    def test_version_callback_true_raises_exit(self) -> None:
        """Test version_callback raises Exit when value is True."""
        with pytest.raises(typer.Exit):
            version_callback(True)


class TestCLIApp:
    """Integration tests for CLI app."""

    def test_version_flag_shows_version(self) -> None:
        """Test --version flag shows version and exits."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "reasoning-mcp version" in result.stdout

    def test_help_flag_shows_help(self) -> None:
        """Test --help shows help message."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "reasoning-mcp" in result.stdout


class TestMain:
    """Tests for main function."""

    def test_main_handles_keyboard_interrupt(self) -> None:
        """Test main handles KeyboardInterrupt gracefully."""
        # Patch the app module-level call in main
        with patch("reasoning_mcp.cli.main.app", side_effect=KeyboardInterrupt()):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 130

    def test_main_handles_exceptions(self) -> None:
        """Test main handles unexpected exceptions."""
        # Patch the app module-level call in main
        with patch("reasoning_mcp.cli.main.app", side_effect=Exception("test error")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
