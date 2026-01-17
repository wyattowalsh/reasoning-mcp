"""Plugin loader for discovering and managing plugins.

This module provides the PluginLoader class which handles the complete
lifecycle of plugins: discovery, loading, initialization, and cleanup.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import tomllib
from typing import TYPE_CHECKING

from reasoning_mcp.plugins.interface import (
    Plugin,
    PluginContext,
    PluginError,
    PluginMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

    from reasoning_mcp.config import Settings
    from reasoning_mcp.registry import MethodRegistry

logger = logging.getLogger(__name__)


def _is_path_within_directory(path: "Path", directory: "Path") -> bool:
    """Check if a path is safely within a directory (no traversal attacks).

    Args:
        path: Path to check
        directory: Directory that should contain the path

    Returns:
        True if path is within directory, False otherwise
    """
    try:
        # Resolve to absolute paths to handle symlinks and normalize
        resolved_path = path.resolve()
        resolved_dir = directory.resolve()
        # Check if the resolved path starts with the resolved directory
        return str(resolved_path).startswith(str(resolved_dir) + "/") or resolved_path == resolved_dir
    except (OSError, ValueError):
        return False


def _validate_entry_point(entry_point: str, plugin_name: str) -> None:
    """Validate that an entry point is safe and well-formed.

    Args:
        entry_point: The entry point string (e.g., "my_plugin.plugin.MyPlugin")
        plugin_name: Name of the plugin (for error messages)

    Raises:
        PluginError: If entry point is invalid or potentially unsafe
    """
    if not entry_point:
        raise PluginError(f"Plugin '{plugin_name}' has no entry_point specified")

    # Check for dangerous patterns
    if ".." in entry_point:
        raise PluginError(f"Plugin '{plugin_name}' has invalid entry_point (contains '..')")

    # Validate format: should be valid Python module.class format
    # Pattern: module.submodule.ClassName
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
    if not re.match(pattern, entry_point):
        raise PluginError(
            f"Plugin '{plugin_name}' has invalid entry_point format: {entry_point}. "
            "Expected format: module.submodule.ClassName"
        )


class PluginLoader:
    """Loader for discovering, loading, and managing plugins.

    The PluginLoader handles the complete plugin lifecycle:
    - Discovery: Scanning directories for plugin metadata files
    - Loading: Dynamically importing plugin modules
    - Initialization: Calling plugin initialize() with context
    - Registration: Adding plugin methods to the registry
    - Cleanup: Unloading and shutting down plugins

    This class supports both package-based plugins (with pyproject.toml)
    and single-file plugins (with plugin.toml).

    Example:
        >>> loader = PluginLoader(
        ...     plugin_dirs=[Path("plugins")],
        ...     registry=method_registry,
        ...     settings=app_settings,
        ... )
        >>> plugins_meta = await loader.discover()
        >>> plugins = await loader.load_all()
        >>> # Later...
        >>> await loader.unload("my_plugin")
    """

    def __init__(
        self,
        plugin_dirs: list[Path],
        registry: MethodRegistry,
        settings: Settings,
    ) -> None:
        """Initialize the plugin loader.

        Args:
            plugin_dirs: List of directories to scan for plugins
            registry: Method registry for registering plugin methods
            settings: Application settings
        """
        self._plugin_dirs = plugin_dirs
        self._registry = registry
        self._settings = settings
        self._loaded_plugins: dict[str, Plugin] = {}
        self._plugin_metadata: dict[str, PluginMetadata] = {}

    async def discover(self) -> list[PluginMetadata]:
        """Discover available plugins in configured directories.

        Scans plugin directories for plugin metadata files:
        - plugin.toml: Standalone plugin configuration
        - pyproject.toml: Package-based plugin with [tool.reasoning-mcp.plugin]

        Returns:
            List of discovered plugin metadata

        Example:
            >>> plugins = await loader.discover()
            >>> for plugin in plugins:
            ...     print(f"Found: {plugin.name} v{plugin.version}")
        """
        discovered: list[PluginMetadata] = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue

            logger.info(f"Scanning for plugins in: {plugin_dir}")

            # Scan for plugin.toml files (standalone plugins)
            for plugin_file in plugin_dir.rglob("plugin.toml"):
                try:
                    # Security: Verify file is within the allowed plugin directory
                    if not _is_path_within_directory(plugin_file, plugin_dir):
                        logger.warning(
                            f"Skipping plugin file outside allowed directory: {plugin_file}"
                        )
                        continue

                    metadata = self._load_metadata_from_toml(plugin_file)
                    discovered.append(metadata)
                    self._plugin_metadata[metadata.name] = metadata
                    logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin metadata from {plugin_file}: {e}")

            # Scan for pyproject.toml files (package-based plugins)
            for pyproject_file in plugin_dir.rglob("pyproject.toml"):
                try:
                    # Security: Verify file is within the allowed plugin directory
                    if not _is_path_within_directory(pyproject_file, plugin_dir):
                        logger.warning(
                            f"Skipping pyproject file outside allowed directory: {pyproject_file}"
                        )
                        continue

                    pyproject_metadata = self._load_metadata_from_pyproject(pyproject_file)
                    if pyproject_metadata:
                        discovered.append(pyproject_metadata)
                        self._plugin_metadata[pyproject_metadata.name] = pyproject_metadata
                        logger.info(
                            f"Discovered plugin: {pyproject_metadata.name} "
                            f"v{pyproject_metadata.version}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load plugin metadata from {pyproject_file}: {e}")

        logger.info(f"Discovery complete: {len(discovered)} plugins found")
        return discovered

    async def load(self, name: str) -> Plugin:
        """Load a specific plugin by name.

        Imports the plugin module, instantiates the plugin class,
        initializes it with context, and registers its methods.

        Args:
            name: Plugin name to load

        Returns:
            Loaded and initialized plugin instance

        Raises:
            PluginError: If plugin loading or initialization fails
            ValueError: If plugin name is not found

        Example:
            >>> plugin = await loader.load("advanced_cot")
            >>> print(f"Loaded: {plugin.name}")
        """
        if name in self._loaded_plugins:
            logger.debug(f"Plugin already loaded: {name}")
            return self._loaded_plugins[name]

        if name not in self._plugin_metadata:
            raise ValueError(f"Plugin '{name}' not found. Run discover() first.")

        metadata = self._plugin_metadata[name]
        logger.info(f"Loading plugin: {name}")

        try:
            # Validate dependencies
            self._validate_dependencies(metadata)

            # Import plugin module and instantiate
            plugin = self._import_plugin(metadata)

            # Initialize plugin with context
            context = PluginContext(
                registry=self._registry,
                settings=self._settings,
                logger=logging.getLogger(f"reasoning_mcp.plugins.{name}"),
            )
            await plugin.initialize(context)

            # Register plugin methods
            methods = plugin.get_methods()
            for method, method_metadata in methods:
                self._registry.register(method, method_metadata, replace=False)
                logger.info(
                    f"Registered method '{method_metadata.identifier}' from plugin '{name}'"
                )

            # Store loaded plugin
            self._loaded_plugins[name] = plugin
            logger.info(f"Successfully loaded plugin: {name}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin '{name}': {e}")
            raise PluginError(f"Failed to load plugin '{name}': {e}") from e

    async def load_all(self) -> dict[str, Plugin]:
        """Load all discovered plugins.

        Attempts to load every plugin found during discovery.
        Plugins that fail to load are logged but don't stop the process.

        Returns:
            Dictionary mapping plugin names to loaded plugin instances

        Example:
            >>> plugins = await loader.load_all()
            >>> print(f"Loaded {len(plugins)} plugins")
        """
        logger.info(f"Loading all plugins ({len(self._plugin_metadata)} discovered)")
        results: dict[str, Plugin] = {}

        for name in self._plugin_metadata:
            try:
                plugin = await self.load(name)
                results[name] = plugin
            except Exception as e:
                logger.error(f"Failed to load plugin '{name}': {e}")
                # Continue loading other plugins

        logger.info(f"Loaded {len(results)}/{len(self._plugin_metadata)} plugins")
        return results

    async def unload(self, name: str) -> bool:
        """Unload a specific plugin.

        Shuts down the plugin, unregisters its methods from the registry,
        and removes it from the loaded plugins.

        Args:
            name: Plugin name to unload

        Returns:
            True if plugin was unloaded, False if not found

        Example:
            >>> success = await loader.unload("my_plugin")
            >>> print(f"Unloaded: {success}")
        """
        if name not in self._loaded_plugins:
            logger.warning(f"Plugin not loaded: {name}")
            return False

        plugin = self._loaded_plugins[name]
        logger.info(f"Unloading plugin: {name}")

        try:
            # Get plugin methods before shutdown
            methods = plugin.get_methods()

            # Shutdown plugin
            await plugin.shutdown()

            # Unregister methods
            for _, method_metadata in methods:
                self._registry.unregister(method_metadata.identifier)
                logger.debug(
                    f"Unregistered method '{method_metadata.identifier}' from plugin '{name}'"
                )

            # Remove from loaded plugins
            del self._loaded_plugins[name]
            logger.info(f"Successfully unloaded plugin: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload plugin '{name}': {e}")
            # Still remove from loaded plugins even if shutdown failed
            del self._loaded_plugins[name]
            return False

    async def reload(self, name: str) -> Plugin:
        """Reload a plugin.

        Unloads the plugin if currently loaded, then loads it again.
        Useful for development or when plugin code has been updated.

        Args:
            name: Plugin name to reload

        Returns:
            Reloaded plugin instance

        Raises:
            PluginError: If reload fails

        Example:
            >>> plugin = await loader.reload("my_plugin")
            >>> print(f"Reloaded: {plugin.name}")
        """
        logger.info(f"Reloading plugin: {name}")

        # Unload if currently loaded
        if name in self._loaded_plugins:
            await self.unload(name)

        # Reload module to pick up code changes
        if name in self._plugin_metadata:
            metadata = self._plugin_metadata[name]
            module_name = metadata.entry_point.rsplit(".", 1)[0]
            try:
                if module_name in importlib.sys.modules:
                    importlib.reload(importlib.sys.modules[module_name])
            except Exception as e:
                logger.warning(f"Failed to reload module '{module_name}': {e}")

        # Load plugin again
        return await self.load(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is currently loaded.

        Args:
            name: Plugin name to check

        Returns:
            True if plugin is loaded, False otherwise

        Example:
            >>> if loader.is_loaded("my_plugin"):
            ...     print("Plugin is active")
        """
        return name in self._loaded_plugins

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a loaded plugin instance by name.

        Args:
            name: Plugin name to retrieve

        Returns:
            Plugin instance if loaded, None otherwise

        Example:
            >>> plugin = loader.get_plugin("my_plugin")
            >>> if plugin:
            ...     print(f"Plugin version: {plugin.version}")
        """
        return self._loaded_plugins.get(name)

    def _load_metadata_from_toml(self, path: Path) -> PluginMetadata:
        """Load plugin metadata from a plugin.toml file.

        Args:
            path: Path to plugin.toml file

        Returns:
            Plugin metadata

        Raises:
            PluginError: If metadata is invalid
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if "plugin" not in data:
            raise PluginError(f"Missing [plugin] section in {path}")

        plugin_data = data["plugin"]
        return PluginMetadata(
            name=plugin_data["name"],
            version=plugin_data["version"],
            author=plugin_data.get("author", "Unknown"),
            description=plugin_data.get("description", ""),
            dependencies=plugin_data.get("dependencies", []),
            entry_point=plugin_data.get("entry_point", ""),
        )

    def _load_metadata_from_pyproject(self, path: Path) -> PluginMetadata | None:
        """Load plugin metadata from a pyproject.toml file.

        Only loads if the file contains [tool.reasoning-mcp.plugin] section.

        Args:
            path: Path to pyproject.toml file

        Returns:
            Plugin metadata if present, None otherwise

        Raises:
            PluginError: If metadata is invalid
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Check for plugin metadata in tool section
        if "tool" not in data or "reasoning-mcp" not in data["tool"]:
            return None

        tool_data = data["tool"]["reasoning-mcp"]
        if "plugin" not in tool_data:
            return None

        plugin_data = tool_data["plugin"]

        # Get project metadata for defaults
        project_data = data.get("project", {})

        return PluginMetadata(
            name=plugin_data["name"],
            version=plugin_data.get("version", project_data.get("version", "0.0.0")),
            author=plugin_data.get(
                "author", ", ".join(project_data.get("authors", [{"name": "Unknown"}])[0].values())
            ),
            description=plugin_data.get("description", project_data.get("description", "")),
            dependencies=plugin_data.get("dependencies", []),
            entry_point=plugin_data.get("entry_point", ""),
        )

    def _validate_dependencies(self, metadata: PluginMetadata) -> None:
        """Validate that plugin dependencies are available.

        Args:
            metadata: Plugin metadata with dependencies

        Raises:
            PluginError: If required dependencies are missing
        """
        missing_deps: list[str] = []

        for dep in metadata.dependencies:
            # Parse dependency (e.g., "numpy>=1.24.0" -> "numpy")
            dep_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()

            try:
                importlib.import_module(dep_name)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise PluginError(
                f"Plugin '{metadata.name}' has missing dependencies: {', '.join(missing_deps)}"
            )

    def _import_plugin(self, metadata: PluginMetadata) -> Plugin:
        """Import and instantiate a plugin from its entry point.

        Args:
            metadata: Plugin metadata with entry_point

        Returns:
            Instantiated plugin object

        Raises:
            PluginError: If import or instantiation fails
        """
        # Security: Validate entry point format before importing
        _validate_entry_point(metadata.entry_point, metadata.name)

        try:
            # Parse entry point (e.g., "my_plugin.plugin.MyPlugin")
            module_name, class_name = metadata.entry_point.rsplit(".", 1)

            # Import module
            module = importlib.import_module(module_name)

            # Get plugin class
            plugin_class = getattr(module, class_name)

            # Instantiate plugin
            plugin: Plugin = plugin_class()

            # Validate protocol compliance
            if not isinstance(plugin, Plugin):
                raise PluginError(f"Plugin class '{class_name}' does not implement Plugin protocol")

            return plugin

        except (ValueError, AttributeError, ImportError) as e:
            raise PluginError(
                f"Failed to import plugin '{metadata.name}' from '{metadata.entry_point}': {e}"
            ) from e


__all__ = ["PluginLoader"]
