"""Tests for plugin loader module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.plugins.interface import (
    Plugin,
    PluginContext,
    PluginError,
    PluginMetadata,
)
from reasoning_mcp.plugins.loader import (
    PluginLoader,
    _is_path_within_directory,
    _validate_entry_point,
)

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings
    from reasoning_mcp.registry import MethodRegistry


class TestIsPathWithinDirectory:
    """Tests for _is_path_within_directory function."""

    def test_path_within_directory(self, tmp_path: Path) -> None:
        """Test path within directory returns True."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file_path = subdir / "file.txt"
        file_path.touch()

        assert _is_path_within_directory(file_path, tmp_path) is True

    def test_path_is_directory(self, tmp_path: Path) -> None:
        """Test directory path returns True."""
        assert _is_path_within_directory(tmp_path, tmp_path) is True

    def test_path_outside_directory(self, tmp_path: Path) -> None:
        """Test path outside directory returns False."""
        outside = tmp_path.parent / "outside.txt"
        # Don't need to create the file, just test the logic
        assert _is_path_within_directory(outside, tmp_path) is False

    def test_path_traversal_attack(self, tmp_path: Path) -> None:
        """Test path traversal attack returns False."""
        attack_path = tmp_path / ".." / ".." / "etc" / "passwd"
        assert _is_path_within_directory(attack_path, tmp_path) is False


class TestValidateEntryPoint:
    """Tests for _validate_entry_point function."""

    def test_valid_entry_point(self) -> None:
        """Test valid entry point passes."""
        _validate_entry_point("my_plugin.plugin.MyPlugin", "test")

    def test_valid_entry_point_with_underscores(self) -> None:
        """Test entry point with underscores passes."""
        _validate_entry_point("my_plugin.sub_module.My_Plugin", "test")

    def test_valid_entry_point_with_numbers(self) -> None:
        """Test entry point with numbers passes."""
        _validate_entry_point("plugin1.v2.Plugin3", "test")

    def test_empty_entry_point_raises(self) -> None:
        """Test empty entry point raises error."""
        with pytest.raises(PluginError) as exc_info:
            _validate_entry_point("", "test")
        assert "no entry_point specified" in str(exc_info.value)

    def test_double_dot_raises(self) -> None:
        """Test entry point with '..' raises error."""
        with pytest.raises(PluginError) as exc_info:
            _validate_entry_point("..evil.module", "test")
        assert "contains '..'" in str(exc_info.value)

    def test_invalid_format_raises(self) -> None:
        """Test invalid entry point format raises error."""
        with pytest.raises(PluginError) as exc_info:
            _validate_entry_point("1invalid.module", "test")
        assert "invalid entry_point format" in str(exc_info.value)

    def test_spaces_in_entry_point_raises(self) -> None:
        """Test entry point with spaces raises error."""
        with pytest.raises(PluginError) as exc_info:
            _validate_entry_point("my plugin.module", "test")
        assert "invalid entry_point format" in str(exc_info.value)


class TestPluginLoader:
    """Tests for PluginLoader class."""

    def create_loader(
        self, plugin_dirs: list[Path], tmp_path: Path | None = None
    ) -> PluginLoader:
        """Create a PluginLoader for testing."""
        registry = MagicMock()
        registry.register = MagicMock()
        registry.unregister = MagicMock()

        settings = MagicMock()
        settings.plugins_dir = tmp_path or Path("/tmp/plugins")

        return PluginLoader(
            plugin_dirs=plugin_dirs,
            registry=registry,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """Test discover returns empty list for empty directory."""
        loader = self.create_loader([tmp_path], tmp_path)
        result = await loader.discover()
        assert result == []

    @pytest.mark.asyncio
    async def test_discover_nonexistent_directory(self) -> None:
        """Test discover handles non-existent directory."""
        loader = self.create_loader([Path("/nonexistent/path")])
        result = await loader.discover()
        assert result == []

    @pytest.mark.asyncio
    async def test_discover_plugin_toml(self, tmp_path: Path) -> None:
        """Test discover finds plugin.toml files."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        plugin_toml = plugin_dir / "plugin.toml"
        plugin_toml.write_text(
            """
[plugin]
name = "test-plugin"
version = "1.0.0"
author = "Test"
description = "Test plugin"
entry_point = "test_plugin.plugin.TestPlugin"
"""
        )

        loader = self.create_loader([tmp_path], tmp_path)
        result = await loader.discover()

        assert len(result) == 1
        assert result[0].name == "test-plugin"
        assert result[0].version == "1.0.0"

    @pytest.mark.asyncio
    async def test_discover_pyproject_toml(self, tmp_path: Path) -> None:
        """Test discover finds pyproject.toml with plugin config."""
        plugin_dir = tmp_path / "package-plugin"
        plugin_dir.mkdir()

        pyproject = plugin_dir / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "package-plugin"
version = "2.0.0"
description = "Package plugin"

[tool.reasoning-mcp.plugin]
name = "package-plugin"
version = "2.0.0"
author = "Package Author"
description = "A package-based plugin"
entry_point = "package_plugin.plugin.PackagePlugin"
"""
        )

        loader = self.create_loader([tmp_path], tmp_path)
        result = await loader.discover()

        assert len(result) == 1
        assert result[0].name == "package-plugin"

    @pytest.mark.asyncio
    async def test_discover_ignores_invalid_toml(self, tmp_path: Path) -> None:
        """Test discover ignores invalid TOML files."""
        plugin_dir = tmp_path / "bad-plugin"
        plugin_dir.mkdir()

        plugin_toml = plugin_dir / "plugin.toml"
        plugin_toml.write_text("not valid toml [[[")

        loader = self.create_loader([tmp_path], tmp_path)
        result = await loader.discover()

        # Should log warning but not fail
        assert result == []

    @pytest.mark.asyncio
    async def test_load_unknown_plugin_raises(self, tmp_path: Path) -> None:
        """Test load raises ValueError for unknown plugin."""
        loader = self.create_loader([tmp_path], tmp_path)

        with pytest.raises(ValueError) as exc_info:
            await loader.load("unknown-plugin")
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_returns_cached_plugin(self, tmp_path: Path) -> None:
        """Test load returns cached plugin on second call."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Manually add a loaded plugin
        mock_plugin = MagicMock(spec=Plugin)
        loader._loaded_plugins["test-plugin"] = mock_plugin

        result = await loader.load("test-plugin")
        assert result is mock_plugin

    @pytest.mark.asyncio
    async def test_load_validates_dependencies(self, tmp_path: Path) -> None:
        """Test load validates plugin dependencies."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Add plugin metadata with missing dependency
        metadata = PluginMetadata(
            name="dep-plugin",
            version="1.0.0",
            author="Test",
            description="Plugin with deps",
            dependencies=["nonexistent_package_xyz"],
            entry_point="dep_plugin.plugin.DepPlugin",
        )
        loader._plugin_metadata["dep-plugin"] = metadata

        with pytest.raises(PluginError) as exc_info:
            await loader.load("dep-plugin")
        assert "missing dependencies" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unload_nonexistent_plugin(self, tmp_path: Path) -> None:
        """Test unload returns False for non-existent plugin."""
        loader = self.create_loader([tmp_path], tmp_path)

        result = await loader.unload("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_unload_plugin(self, tmp_path: Path) -> None:
        """Test unload properly unloads a plugin."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Create mock plugin
        mock_plugin = MagicMock(spec=Plugin)
        mock_plugin.shutdown = AsyncMock()
        mock_plugin.get_methods.return_value = [
            (MagicMock(), MagicMock(identifier="method1")),
        ]

        loader._loaded_plugins["test-plugin"] = mock_plugin

        result = await loader.unload("test-plugin")

        assert result is True
        assert "test-plugin" not in loader._loaded_plugins
        mock_plugin.shutdown.assert_called_once()
        loader._registry.unregister.assert_called_with("method1")

    @pytest.mark.asyncio
    async def test_reload_plugin(self, tmp_path: Path) -> None:
        """Test reload unloads and reloads plugin."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Add metadata
        metadata = PluginMetadata(
            name="reload-plugin",
            version="1.0.0",
            author="Test",
            description="Reload plugin",
            entry_point="reload_plugin.plugin.ReloadPlugin",
        )
        loader._plugin_metadata["reload-plugin"] = metadata

        # Mock the import and plugin instantiation
        mock_plugin = MagicMock(spec=Plugin)
        mock_plugin.initialize = AsyncMock()
        mock_plugin.shutdown = AsyncMock()
        mock_plugin.get_methods.return_value = []

        with patch.object(loader, "_import_plugin", return_value=mock_plugin):
            result = await loader.reload("reload-plugin")

        assert result is mock_plugin
        mock_plugin.initialize.assert_called_once()

    def test_is_loaded(self, tmp_path: Path) -> None:
        """Test is_loaded returns correct status."""
        loader = self.create_loader([tmp_path], tmp_path)

        assert loader.is_loaded("test") is False

        loader._loaded_plugins["test"] = MagicMock()
        assert loader.is_loaded("test") is True

    def test_get_plugin(self, tmp_path: Path) -> None:
        """Test get_plugin returns plugin or None."""
        loader = self.create_loader([tmp_path], tmp_path)

        assert loader.get_plugin("test") is None

        mock_plugin = MagicMock(spec=Plugin)
        loader._loaded_plugins["test"] = mock_plugin
        assert loader.get_plugin("test") is mock_plugin

    @pytest.mark.asyncio
    async def test_load_all(self, tmp_path: Path) -> None:
        """Test load_all loads all discovered plugins."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Add multiple plugin metadata
        for name in ["plugin1", "plugin2"]:
            loader._plugin_metadata[name] = PluginMetadata(
                name=name,
                version="1.0.0",
                author="Test",
                description=f"{name} description",
                entry_point=f"{name}.plugin.Plugin",
            )

        # Mock load to succeed
        mock_plugin = MagicMock(spec=Plugin)
        with patch.object(loader, "load", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_plugin

            result = await loader.load_all()

            assert len(result) == 2
            assert mock_load.call_count == 2

    @pytest.mark.asyncio
    async def test_load_all_continues_on_failure(self, tmp_path: Path) -> None:
        """Test load_all continues loading when one plugin fails."""
        loader = self.create_loader([tmp_path], tmp_path)

        # Add multiple plugin metadata
        for name in ["good-plugin", "bad-plugin", "another-good"]:
            loader._plugin_metadata[name] = PluginMetadata(
                name=name,
                version="1.0.0",
                author="Test",
                description=f"{name} description",
                entry_point=f"{name}.plugin.Plugin",
            )

        mock_plugin = MagicMock(spec=Plugin)

        async def mock_load(name: str):
            if "bad" in name:
                raise PluginError("Load failed")
            return mock_plugin

        with patch.object(loader, "load", side_effect=mock_load):
            result = await loader.load_all()

            # Should have loaded 2 of 3 plugins
            assert len(result) == 2


class TestPluginLoaderMetadataLoading:
    """Tests for PluginLoader metadata loading methods."""

    def create_loader(self, tmp_path: Path) -> PluginLoader:
        """Create a PluginLoader for testing."""
        registry = MagicMock()
        settings = MagicMock()
        return PluginLoader([tmp_path], registry, settings)

    def test_load_metadata_from_toml(self, tmp_path: Path) -> None:
        """Test loading metadata from plugin.toml."""
        loader = self.create_loader(tmp_path)

        plugin_toml = tmp_path / "plugin.toml"
        plugin_toml.write_text(
            """
[plugin]
name = "toml-plugin"
version = "1.2.3"
author = "TOML Author"
description = "A TOML plugin"
dependencies = ["dep1", "dep2"]
entry_point = "toml_plugin.plugin.TomlPlugin"
"""
        )

        metadata = loader._load_metadata_from_toml(plugin_toml)

        assert metadata.name == "toml-plugin"
        assert metadata.version == "1.2.3"
        assert metadata.author == "TOML Author"
        assert metadata.description == "A TOML plugin"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.entry_point == "toml_plugin.plugin.TomlPlugin"

    def test_load_metadata_from_toml_missing_section(self, tmp_path: Path) -> None:
        """Test error when plugin.toml missing [plugin] section."""
        loader = self.create_loader(tmp_path)

        plugin_toml = tmp_path / "plugin.toml"
        plugin_toml.write_text("[other]\nname = 'test'")

        with pytest.raises(PluginError) as exc_info:
            loader._load_metadata_from_toml(plugin_toml)
        assert "Missing [plugin] section" in str(exc_info.value)

    def test_load_metadata_from_pyproject(self, tmp_path: Path) -> None:
        """Test loading metadata from pyproject.toml."""
        loader = self.create_loader(tmp_path)

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "pyproject-plugin"
version = "3.0.0"
description = "Project description"

[tool.reasoning-mcp.plugin]
name = "pyproject-plugin"
version = "3.0.0"
author = "Project Author"
description = "A pyproject plugin"
entry_point = "pyproject_plugin.plugin.Plugin"
"""
        )

        metadata = loader._load_metadata_from_pyproject(pyproject)

        assert metadata is not None
        assert metadata.name == "pyproject-plugin"
        assert metadata.version == "3.0.0"

    def test_load_metadata_from_pyproject_no_plugin(self, tmp_path: Path) -> None:
        """Test pyproject.toml without plugin section returns None."""
        loader = self.create_loader(tmp_path)

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "regular-project"
version = "1.0.0"
"""
        )

        metadata = loader._load_metadata_from_pyproject(pyproject)
        assert metadata is None
