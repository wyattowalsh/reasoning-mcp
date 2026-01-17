"""Tests for CLI health command."""
from __future__ import annotations

import asyncio
import json
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from reasoning_mcp.cli.commands.health import (
    HealthCheck,
    check_config,
    check_core_imports,
    check_dependencies,
    check_plugins,
    check_registry,
    fix_issues,
    format_json_output,
    format_output,
    health_check,
)

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_health_check_creation(self) -> None:
        """Test HealthCheck can be created with required fields."""
        check = HealthCheck(
            name="Test Check",
            status="ok",
            message="All good",
        )
        assert check.name == "Test Check"
        assert check.status == "ok"
        assert check.message == "All good"
        assert check.details == {}
        assert check.fixable is False

    def test_health_check_with_details(self) -> None:
        """Test HealthCheck with details and fixable flag."""
        check = HealthCheck(
            name="Dependency Check",
            status="error",
            message="Missing packages",
            details={"missing": ["pkg1", "pkg2"]},
            fixable=True,
        )
        assert check.details == {"missing": ["pkg1", "pkg2"]}
        assert check.fixable is True


class TestCheckCoreImports:
    """Tests for check_core_imports function."""

    @pytest.mark.asyncio
    async def test_check_core_imports_succeeds(self) -> None:
        """Test core imports check succeeds when imports work."""
        result = await check_core_imports()
        assert result.name == "Core Imports"
        assert result.status == "ok"
        assert result.message == "Imported"


class TestCheckRegistry:
    """Tests for check_registry function."""

    @pytest.mark.asyncio
    async def test_check_registry_succeeds(self) -> None:
        """Test registry check succeeds when registry works."""
        # Patch at the source since import happens inside the function
        with patch("reasoning_mcp.registry.MethodRegistry") as mock_registry:
            mock_instance = MagicMock()
            mock_instance.method_count = 50
            mock_registry.return_value = mock_instance

            result = await check_registry()

            assert result.name == "Registry"
            assert result.status == "ok"
            assert "50 methods" in result.message

    @pytest.mark.asyncio
    async def test_check_registry_fails_on_error(self) -> None:
        """Test registry check fails when registry raises error."""
        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.registry.MethodRegistry",
            side_effect=Exception("Registry error"),
        ):
            result = await check_registry()

            assert result.name == "Registry"
            assert result.status == "error"
            assert "Registry error" in result.message


class TestCheckPlugins:
    """Tests for check_plugins function."""

    @pytest.mark.asyncio
    async def test_check_plugins_disabled(self) -> None:
        """Test plugin check when plugins are disabled."""
        mock_settings = MagicMock()
        mock_settings.enable_plugins = False

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            result = await check_plugins()

            assert result.name == "Plugins"
            assert result.status == "ok"
            assert result.message == "Disabled"

    @pytest.mark.asyncio
    async def test_check_plugins_dir_exists(self, tmp_path) -> None:
        """Test plugin check when plugins dir exists."""
        mock_settings = MagicMock()
        mock_settings.enable_plugins = True
        mock_settings.plugins_dir = tmp_path

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            result = await check_plugins()

            assert result.name == "Plugins"
            assert result.status == "ok"
            assert result.message == "OK"

    @pytest.mark.asyncio
    async def test_check_plugins_dir_missing(self, tmp_path) -> None:
        """Test plugin check when plugins dir is missing."""
        mock_settings = MagicMock()
        mock_settings.enable_plugins = True
        mock_settings.plugins_dir = tmp_path / "nonexistent"

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            result = await check_plugins()

            assert result.name == "Plugins"
            assert result.status == "warning"
            assert result.message == "Missing"
            assert result.fixable is True


class TestCheckConfig:
    """Tests for check_config function."""

    @pytest.mark.asyncio
    async def test_check_config_valid(self) -> None:
        """Test config check with valid settings."""
        mock_settings = MagicMock()
        mock_settings.session_timeout = 3600
        mock_settings.max_sessions = 100

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            result = await check_config()

            assert result.name == "Config"
            assert result.status == "ok"

    @pytest.mark.asyncio
    async def test_check_config_warnings(self) -> None:
        """Test config check with problematic settings."""
        mock_settings = MagicMock()
        mock_settings.session_timeout = 30  # Too low
        mock_settings.max_sessions = 0  # Too low

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            result = await check_config()

            assert result.name == "Config"
            assert result.status == "warning"


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    @pytest.mark.asyncio
    async def test_check_dependencies_all_present(self) -> None:
        """Test dependencies check when all deps are present."""
        result = await check_dependencies()

        assert result.name == "Dependencies"
        # Core deps should be installed in test environment
        assert isinstance(result.status, str)

    @pytest.mark.asyncio
    async def test_check_dependencies_some_missing(self) -> None:
        """Test dependencies check when some deps are missing."""
        with patch("importlib.util.find_spec", return_value=None):
            result = await check_dependencies()

            assert result.name == "Dependencies"
            assert result.status == "error"
            assert "Missing" in result.message
            assert result.fixable is True


class TestFixIssues:
    """Tests for fix_issues function."""

    @pytest.mark.asyncio
    async def test_fix_plugins_dir(self, tmp_path) -> None:
        """Test fixing missing plugins directory."""
        mock_settings = MagicMock()
        mock_settings.plugins_dir = tmp_path / "plugins"

        check = HealthCheck(
            name="Plugins",
            status="warning",
            message="Missing",
            fixable=True,
        )

        # Patch at the source since import happens inside the function
        with patch(
            "reasoning_mcp.config.get_settings",
            return_value=mock_settings,
        ):
            results = await fix_issues([check])

            assert "Plugins" in results
            assert results["Plugins"] is True
            assert mock_settings.plugins_dir.exists()

    @pytest.mark.asyncio
    async def test_fix_unfixable_issue(self) -> None:
        """Test that unfixable issues are not attempted."""
        check = HealthCheck(
            name="Some Issue",
            status="error",
            message="Cannot fix",
            fixable=False,
        )

        results = await fix_issues([check])

        # Unfixable issues should not appear in results
        assert "Some Issue" not in results

    @pytest.mark.asyncio
    async def test_fix_already_ok(self) -> None:
        """Test that OK checks are not fixed."""
        check = HealthCheck(
            name="Already OK",
            status="ok",
            message="All good",
            fixable=True,  # Even if fixable, should be skipped
        )

        results = await fix_issues([check])

        assert "Already OK" not in results


class TestFormatOutput:
    """Tests for format_output function."""

    def test_format_output_verbose(self) -> None:
        """Test format_output in verbose mode shows all checks."""
        checks = [
            HealthCheck(name="Check1", status="ok", message="OK"),
            HealthCheck(name="Check2", status="warning", message="Warn"),
        ]

        # Patch at the source since import happens inside the function
        with patch("rich.console.Console"):
            format_output(checks, verbose=True)

    def test_format_output_non_verbose(self) -> None:
        """Test format_output in non-verbose mode shows only issues."""
        checks = [
            HealthCheck(name="Check1", status="ok", message="OK"),
            HealthCheck(name="Check2", status="error", message="Error"),
        ]

        # Patch at the source since import happens inside the function
        with patch("rich.console.Console"):
            format_output(checks, verbose=False)


class TestFormatJsonOutput:
    """Tests for format_json_output function."""

    def test_format_json_output_healthy(self, capsys) -> None:
        """Test JSON output for healthy system."""
        checks = [
            HealthCheck(name="Check1", status="ok", message="OK"),
            HealthCheck(name="Check2", status="ok", message="OK"),
        ]

        format_json_output(checks)
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "healthy"
        assert len(output["checks"]) == 2
        assert output["summary"]["ok"] == 2
        assert output["summary"]["error"] == 0

    def test_format_json_output_degraded(self, capsys) -> None:
        """Test JSON output for degraded system."""
        checks = [
            HealthCheck(name="Check1", status="ok", message="OK"),
            HealthCheck(name="Check2", status="warning", message="Warn"),
        ]

        format_json_output(checks)
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "degraded"

    def test_format_json_output_unhealthy(self, capsys) -> None:
        """Test JSON output for unhealthy system."""
        checks = [
            HealthCheck(name="Check1", status="error", message="Error"),
        ]

        format_json_output(checks)
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "unhealthy"


class TestHealthCheckCommand:
    """Tests for the health_check main function."""

    def test_health_check_exits_on_error(self) -> None:
        """Test health_check exits with code 1 when errors exist."""
        ctx = MagicMock()

        with patch(
            "reasoning_mcp.cli.commands.health.check_core_imports",
            return_value=HealthCheck(name="Core", status="error", message="Failed"),
        ):
            with patch(
                "reasoning_mcp.cli.commands.health.check_registry",
                return_value=HealthCheck(name="Registry", status="ok", message="OK"),
            ):
                with patch(
                    "reasoning_mcp.cli.commands.health.check_config",
                    return_value=HealthCheck(name="Config", status="ok", message="OK"),
                ):
                    with patch(
                        "reasoning_mcp.cli.commands.health.check_dependencies",
                        return_value=HealthCheck(
                            name="Dependencies", status="ok", message="OK"
                        ),
                    ):
                        with patch(
                            "reasoning_mcp.cli.commands.health.check_plugins",
                            return_value=HealthCheck(
                                name="Plugins", status="ok", message="OK"
                            ),
                        ):
                            with patch(
                                "reasoning_mcp.cli.commands.health.check_methods",
                                return_value=[],
                            ):
                                with pytest.raises(typer.Exit) as exc_info:
                                    health_check(ctx)
                                assert exc_info.value.exit_code == 1
