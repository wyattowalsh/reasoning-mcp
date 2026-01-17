"""Tests for CLI install command."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from reasoning_mcp.cli.commands.install import (
    InstallError,
    _clone_git_repo,
    _parse_github_source,
    _validate_dependency_specifier,
    _validate_git_ref,
    _validate_git_url,
    copy_plugin_files,
    get_plugins_directory,
    install_dependencies,
    install_plugin_impl,
    list_plugins_impl,
    resolve_source,
    uninstall_plugin_impl,
    validate_plugin_structure,
)
from reasoning_mcp.plugins.interface import PluginMetadata

if TYPE_CHECKING:
    from reasoning_mcp.cli.main import CLIContext


class TestInstallError:
    """Tests for InstallError exception."""

    def test_install_error_is_exception(self) -> None:
        """Test InstallError is an Exception."""
        err = InstallError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


class TestGetPluginsDirectory:
    """Tests for get_plugins_directory function."""

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Test get_plugins_directory creates directory if missing."""
        mock_settings = MagicMock()
        mock_settings.plugins_dir = tmp_path / "plugins"

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            result = get_plugins_directory()

            assert result == mock_settings.plugins_dir
            assert result.exists()


class TestValidateDependencySpecifier:
    """Tests for _validate_dependency_specifier function."""

    def test_valid_simple_package(self) -> None:
        """Test valid simple package name."""
        result = _validate_dependency_specifier("numpy")
        assert result == "numpy"

    def test_valid_package_with_version(self) -> None:
        """Test valid package with version specifier."""
        result = _validate_dependency_specifier("numpy>=1.24.0")
        assert result == "numpy>=1.24.0"

    def test_valid_package_with_extras(self) -> None:
        """Test valid package with extras."""
        result = _validate_dependency_specifier("package[extra1,extra2]")
        assert result == "package[extra1,extra2]"

    def test_rejects_command_injection_semicolon(self) -> None:
        """Test rejects semicolon (command injection)."""
        with pytest.raises(InstallError) as exc_info:
            _validate_dependency_specifier("package; rm -rf /")
        assert "unsafe characters" in str(exc_info.value)

    def test_rejects_command_injection_pipe(self) -> None:
        """Test rejects pipe (command injection)."""
        with pytest.raises(InstallError) as exc_info:
            _validate_dependency_specifier("package | cat /etc/passwd")
        assert "unsafe characters" in str(exc_info.value)

    def test_rejects_command_injection_ampersand(self) -> None:
        """Test rejects ampersand (command injection)."""
        with pytest.raises(InstallError) as exc_info:
            _validate_dependency_specifier("package && malicious")
        assert "unsafe characters" in str(exc_info.value)

    def test_rejects_command_injection_backtick(self) -> None:
        """Test rejects backtick (command injection)."""
        with pytest.raises(InstallError) as exc_info:
            _validate_dependency_specifier("package`whoami`")
        assert "unsafe characters" in str(exc_info.value)


class TestValidateGitUrl:
    """Tests for _validate_git_url function."""

    def test_valid_https_url(self) -> None:
        """Test valid HTTPS URL."""
        result = _validate_git_url("https://github.com/user/repo.git")
        assert result == "https://github.com/user/repo.git"

    def test_valid_https_url_without_git(self) -> None:
        """Test valid HTTPS URL without .git."""
        result = _validate_git_url("https://github.com/user/repo")
        assert result == "https://github.com/user/repo"

    def test_valid_ssh_url(self) -> None:
        """Test valid SSH URL."""
        result = _validate_git_url("git@github.com:user/repo.git")
        assert result == "git@github.com:user/repo.git"

    def test_rejects_command_injection(self) -> None:
        """Test rejects URL with command injection."""
        with pytest.raises(InstallError) as exc_info:
            _validate_git_url("https://github.com/user/repo.git; rm -rf /")
        assert "unsafe characters" in str(exc_info.value)

    def test_rejects_invalid_url(self) -> None:
        """Test rejects invalid URL format."""
        with pytest.raises(InstallError) as exc_info:
            _validate_git_url("not-a-valid-url")
        assert "Invalid Git URL format" in str(exc_info.value)


class TestValidateGitRef:
    """Tests for _validate_git_ref function."""

    def test_valid_branch_name(self) -> None:
        """Test valid branch name."""
        result = _validate_git_ref("main")
        assert result == "main"

    def test_valid_tag(self) -> None:
        """Test valid tag."""
        result = _validate_git_ref("v1.0.0")
        assert result == "v1.0.0"

    def test_valid_feature_branch(self) -> None:
        """Test valid feature branch with slash."""
        result = _validate_git_ref("feature/new-feature")
        assert result == "feature/new-feature"

    def test_rejects_command_injection(self) -> None:
        """Test rejects ref with command injection."""
        with pytest.raises(InstallError) as exc_info:
            _validate_git_ref("main; rm -rf /")
        assert "unsafe characters" in str(exc_info.value)

    def test_rejects_path_traversal(self) -> None:
        """Test rejects path traversal."""
        with pytest.raises(InstallError) as exc_info:
            _validate_git_ref("../../../etc/passwd")
        assert "contains '..'" in str(exc_info.value)


class TestParseGithubSource:
    """Tests for _parse_github_source function."""

    def test_github_prefix(self) -> None:
        """Test github: prefix."""
        url, ref = _parse_github_source("github:user/repo")
        assert url == "https://github.com/user/repo.git"
        assert ref is None

    def test_gh_prefix(self) -> None:
        """Test gh: prefix."""
        url, ref = _parse_github_source("gh:user/repo")
        assert url == "https://github.com/user/repo.git"
        assert ref is None

    def test_user_repo_format(self) -> None:
        """Test user/repo format."""
        url, ref = _parse_github_source("user/repo")
        assert url == "https://github.com/user/repo.git"
        assert ref is None

    def test_github_with_branch(self) -> None:
        """Test github: with @branch."""
        url, ref = _parse_github_source("github:user/repo@main")
        assert url == "https://github.com/user/repo.git"
        assert ref == "main"

    def test_github_with_tag(self) -> None:
        """Test github: with #tag."""
        url, ref = _parse_github_source("gh:user/repo#v1.0.0")
        assert url == "https://github.com/user/repo.git"
        assert ref == "v1.0.0"

    def test_full_url_passthrough(self) -> None:
        """Test full URL passes through."""
        url, ref = _parse_github_source("https://example.com/repo.git")
        assert url == "https://example.com/repo.git"
        assert ref is None


class TestValidatePluginStructure:
    """Tests for validate_plugin_structure function."""

    def test_valid_plugin_json(self, tmp_path: Path) -> None:
        """Test valid plugin with plugin.json."""
        plugin_path = tmp_path / "test-plugin"
        plugin_path.mkdir()

        plugin_json = plugin_path / "plugin.json"
        plugin_json.write_text(
            json.dumps(
                {
                    "name": "test-plugin",
                    "version": "1.0.0",
                    "author": "Test Author",
                    "description": "Test plugin",
                }
            )
        )

        result = validate_plugin_structure(plugin_path)

        assert isinstance(result, PluginMetadata)
        assert result.name == "test-plugin"
        assert result.version == "1.0.0"
        assert result.author == "Test Author"
        assert result.description == "Test plugin"

    def test_valid_plugin_toml(self, tmp_path: Path) -> None:
        """Test valid plugin with plugin.toml."""
        plugin_path = tmp_path / "test-plugin"
        plugin_path.mkdir()

        plugin_toml = plugin_path / "plugin.toml"
        # Note: validate_plugin_structure expects flat TOML structure
        # (not [plugin] section like the loader uses)
        plugin_toml.write_text(
            """
name = "test-plugin"
version = "1.0.0"
author = "Test Author"
description = "Test plugin"
"""
        )

        result = validate_plugin_structure(plugin_path)

        assert isinstance(result, PluginMetadata)
        assert result.name == "test-plugin"

    def test_missing_metadata_file(self, tmp_path: Path) -> None:
        """Test error when metadata file is missing."""
        plugin_path = tmp_path / "test-plugin"
        plugin_path.mkdir()

        with pytest.raises(InstallError) as exc_info:
            validate_plugin_structure(plugin_path)
        assert "missing metadata file" in str(exc_info.value)

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test error when JSON is invalid."""
        plugin_path = tmp_path / "test-plugin"
        plugin_path.mkdir()

        plugin_json = plugin_path / "plugin.json"
        plugin_json.write_text("not valid json")

        with pytest.raises(InstallError) as exc_info:
            validate_plugin_structure(plugin_path)
        assert "Invalid plugin.json" in str(exc_info.value)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Test error when required fields are missing."""
        plugin_path = tmp_path / "test-plugin"
        plugin_path.mkdir()

        plugin_json = plugin_path / "plugin.json"
        plugin_json.write_text(json.dumps({"name": "test"}))  # Missing fields

        with pytest.raises(InstallError) as exc_info:
            validate_plugin_structure(plugin_path)
        assert "missing required fields" in str(exc_info.value)


class TestInstallDependencies:
    """Tests for install_dependencies function."""

    def test_empty_dependencies_returns(self) -> None:
        """Test empty dependencies list returns early."""
        # Should not call subprocess
        install_dependencies([])

    def test_installs_valid_dependencies(self) -> None:
        """Test installing valid dependencies."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            install_dependencies(["numpy>=1.24.0"])

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "--" in call_args  # Security: verify -- separator
            assert "numpy>=1.24.0" in call_args

    def test_rejects_invalid_dependencies(self) -> None:
        """Test rejects invalid dependencies."""
        with pytest.raises(InstallError):
            install_dependencies(["package; rm -rf /"])


class TestCopyPluginFiles:
    """Tests for copy_plugin_files function."""

    def test_copies_files(self, tmp_path: Path) -> None:
        """Test copies files from source to destination."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        dest = tmp_path / "dest"

        copy_plugin_files(source, dest)

        assert dest.exists()
        assert (dest / "file.txt").exists()
        assert (dest / "file.txt").read_text() == "content"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        """Test overwrites existing destination."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "new.txt").write_text("new")

        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "old.txt").write_text("old")

        copy_plugin_files(source, dest)

        assert (dest / "new.txt").exists()
        assert not (dest / "old.txt").exists()


class TestResolveSource:
    """Tests for resolve_source function."""

    def test_resolves_local_path(self, tmp_path: Path) -> None:
        """Test resolves local directory path."""
        plugin_path = tmp_path / "local-plugin"
        plugin_path.mkdir()

        result = resolve_source(str(plugin_path))

        assert result == plugin_path.resolve()

    def test_handles_github_shorthand(self) -> None:
        """Test handles github: shorthand."""
        with patch(
            "reasoning_mcp.cli.commands.install._clone_git_repo"
        ) as mock_clone:
            mock_clone.return_value = Path("/tmp/cloned")

            result = resolve_source("github:user/repo")

            mock_clone.assert_called_once()
            assert result == Path("/tmp/cloned")

    def test_handles_user_repo_format(self) -> None:
        """Test handles user/repo format."""
        with patch(
            "reasoning_mcp.cli.commands.install._clone_git_repo"
        ) as mock_clone:
            mock_clone.return_value = Path("/tmp/cloned")

            result = resolve_source("user/repo")

            mock_clone.assert_called_once()


class TestInstallPluginImpl:
    """Tests for install_plugin_impl function."""

    def test_successful_install(self, tmp_path: Path) -> None:
        """Test successful plugin installation."""
        ctx = MagicMock()

        # Create source plugin
        source = tmp_path / "source-plugin"
        source.mkdir()
        (source / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "test-plugin",
                    "version": "1.0.0",
                    "author": "Test",
                    "description": "Test plugin",
                }
            )
        )

        mock_settings = MagicMock()
        mock_settings.plugins_dir = tmp_path / "plugins"

        with patch(
            "reasoning_mcp.cli.commands.install.resolve_source",
            return_value=source,
        ):
            with patch(
                "reasoning_mcp.cli.commands.install.get_settings",
                return_value=mock_settings,
            ):
                install_plugin_impl(ctx, str(source), upgrade=False)

        assert (mock_settings.plugins_dir / "test-plugin").exists()

    def test_rejects_existing_without_upgrade(self, tmp_path: Path) -> None:
        """Test rejects installing existing plugin without --upgrade."""
        ctx = MagicMock()

        source = tmp_path / "source-plugin"
        source.mkdir()
        (source / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "existing-plugin",
                    "version": "1.0.0",
                    "author": "Test",
                    "description": "Test",
                }
            )
        )

        mock_settings = MagicMock()
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "existing-plugin").mkdir()  # Already exists
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.resolve_source",
            return_value=source,
        ):
            with patch(
                "reasoning_mcp.cli.commands.install.get_settings",
                return_value=mock_settings,
            ):
                with pytest.raises(typer.Exit) as exc_info:
                    install_plugin_impl(ctx, str(source), upgrade=False)
                assert exc_info.value.exit_code == 1


class TestListPluginsImpl:
    """Tests for list_plugins_impl function."""

    def test_lists_installed_plugins(self, tmp_path: Path) -> None:
        """Test lists installed plugins."""
        ctx = MagicMock()

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        # Create a valid plugin
        plugin = plugins_dir / "test-plugin"
        plugin.mkdir()
        (plugin / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "test-plugin",
                    "version": "1.0.0",
                    "author": "Test",
                    "description": "Test",
                }
            )
        )

        mock_settings = MagicMock()
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            # Should not raise
            list_plugins_impl(ctx)

    def test_handles_empty_plugins_dir(self, tmp_path: Path) -> None:
        """Test handles empty plugins directory."""
        ctx = MagicMock()

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        mock_settings = MagicMock()
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            # Should not raise
            list_plugins_impl(ctx)


class TestUninstallPluginImpl:
    """Tests for uninstall_plugin_impl function."""

    def test_uninstalls_existing_plugin(self, tmp_path: Path) -> None:
        """Test uninstalls existing plugin."""
        ctx = MagicMock()

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        plugin = plugins_dir / "test-plugin"
        plugin.mkdir()

        mock_settings = MagicMock()
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            with patch("typer.confirm", return_value=True):
                uninstall_plugin_impl(ctx, "test-plugin")

        assert not plugin.exists()

    def test_fails_for_nonexistent_plugin(self, tmp_path: Path) -> None:
        """Test fails for non-existent plugin."""
        ctx = MagicMock()

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        mock_settings = MagicMock()
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                uninstall_plugin_impl(ctx, "nonexistent")
            assert exc_info.value.exit_code == 1

    def test_cancelled_uninstall(self, tmp_path: Path) -> None:
        """Test cancelled uninstall."""
        ctx = MagicMock()

        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        plugin = plugins_dir / "test-plugin"
        plugin.mkdir()

        mock_settings = MagicMock()
        mock_settings.plugins_dir = plugins_dir

        with patch(
            "reasoning_mcp.cli.commands.install.get_settings",
            return_value=mock_settings,
        ):
            with patch("typer.confirm", return_value=False):
                with pytest.raises(typer.Exit) as exc_info:
                    uninstall_plugin_impl(ctx, "test-plugin")
                assert exc_info.value.exit_code == 0

        # Plugin should still exist
        assert plugin.exists()
