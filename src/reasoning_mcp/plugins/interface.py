"""Plugin protocol and interfaces for reasoning-mcp.

This module defines the plugin system interfaces, allowing third-party
extensions to register custom reasoning methods with the server.

The plugin system uses Protocol-based interfaces for type safety while
maintaining flexibility for different plugin implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from reasoning_mcp.config import Settings
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.registry import MethodRegistry


class PluginError(Exception):
    """Base exception for plugin-related errors.

    This exception is raised when plugins fail to load, initialize,
    or execute properly. Plugins should raise this or a subclass
    when encountering errors during their lifecycle.

    Example:
        >>> raise PluginError("Failed to load method configuration")
    """
    pass


@dataclass(frozen=True)
class PluginMetadata:
    """Metadata describing a plugin.

    Contains information about the plugin's identity, dependencies,
    and entry point for discovery and loading.

    Attributes:
        name: Unique plugin name (e.g., "my_reasoning_plugin")
        version: Semantic version string (e.g., "1.0.0")
        author: Plugin author name or organization
        description: Brief description of plugin functionality
        dependencies: List of required Python packages
        entry_point: Module path to plugin class (e.g., "my_plugin.plugin.MyPlugin")

    Example:
        >>> metadata = PluginMetadata(
        ...     name="advanced_cot",
        ...     version="1.0.0",
        ...     author="Research Team",
        ...     description="Advanced chain-of-thought methods",
        ...     dependencies=["numpy>=1.24.0", "scipy>=1.10.0"],
        ...     entry_point="advanced_cot.plugin.AdvancedCOTPlugin",
        ... )
    """
    name: str
    version: str
    author: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    entry_point: str = ""

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
        if not self.description:
            raise ValueError("Plugin description cannot be empty")


@dataclass
class PluginContext:
    """Context provided to plugins during initialization.

    This dataclass contains references to core system components
    that plugins need to register methods and access configuration.

    Attributes:
        registry: Method registry for registering reasoning methods
        settings: Application configuration settings
        logger: Plugin-specific logger instance

    Example:
        >>> context = PluginContext(
        ...     registry=method_registry,
        ...     settings=app_settings,
        ...     logger=logging.getLogger("reasoning_mcp.plugins.my_plugin"),
        ... )
        >>> plugin = MyPlugin()
        >>> await plugin.initialize(context)
    """
    registry: MethodRegistry
    settings: Settings
    logger: logging.Logger


class Plugin(Protocol):
    """Protocol defining the interface for reasoning-mcp plugins.

    All plugins must implement this protocol to be loaded and managed
    by the plugin manager. Plugins extend the server's capabilities by
    registering custom reasoning methods.

    The plugin lifecycle:
        1. Discovery: Plugin metadata is read from plugin.json
        2. Loading: Plugin class is imported via entry_point
        3. Initialization: initialize() is called with PluginContext
        4. Registration: get_methods() provides methods to register
        5. Execution: Methods are available via the registry
        6. Shutdown: shutdown() is called during cleanup

    Example Implementation:
        >>> class MyReasoningPlugin:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_reasoning_plugin"
        ...
        ...     @property
        ...     def version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Custom reasoning methods"
        ...
        ...     async def initialize(self, context: PluginContext) -> None:
        ...         self._context = context
        ...         context.logger.info(f"Initializing {self.name}")
        ...
        ...     async def shutdown(self) -> None:
        ...         self._context.logger.info(f"Shutting down {self.name}")
        ...
        ...     def get_methods(self) -> list[tuple[ReasoningMethod, MethodMetadata]]:
        ...         return [(MyMethod(), my_method_metadata)]
    """

    @property
    def name(self) -> str:
        """Unique identifier for the plugin.

        This should match the plugin metadata name and be unique
        across all loaded plugins.

        Returns:
            Plugin name string
        """
        ...

    @property
    def version(self) -> str:
        """Semantic version of the plugin.

        Should follow semantic versioning (e.g., "1.0.0", "2.1.3").

        Returns:
            Version string
        """
        ...

    @property
    def description(self) -> str:
        """Brief description of plugin functionality.

        This should clearly explain what reasoning methods or
        capabilities the plugin provides.

        Returns:
            Description string
        """
        ...

    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with system context.

        This method is called once when the plugin is loaded. Plugins
        should perform setup, validate dependencies, and prepare their
        methods for registration. The context provides access to the
        registry, settings, and logger.

        Args:
            context: Plugin context with system components

        Raises:
            PluginError: If initialization fails
        """
        ...

    async def shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources.

        This method is called when the server is shutting down or when
        the plugin is being unloaded. Plugins should cleanup any
        resources, close connections, and perform final cleanup.

        Raises:
            PluginError: If shutdown fails
        """
        ...

    def get_methods(self) -> list[tuple[ReasoningMethod, MethodMetadata]]:
        """Get reasoning methods provided by this plugin.

        Returns a list of (method, metadata) tuples. Each method must
        implement the ReasoningMethod protocol and have corresponding
        metadata describing its capabilities.

        This method is called after initialize() to discover and
        register the plugin's methods with the registry.

        Returns:
            List of (ReasoningMethod, MethodMetadata) tuples

        Example:
            >>> def get_methods(self):
            ...     return [
            ...         (ChainOfThoughtV2(), cot_v2_metadata),
            ...         (TreeOfThoughts(), tot_metadata),
            ...     ]
        """
        ...


__all__ = [
    "Plugin",
    "PluginContext",
    "PluginMetadata",
    "PluginError",
]
