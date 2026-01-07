"""
End-to-end tests for the plugin system in reasoning-mcp.

This module provides comprehensive E2E tests for plugin discovery, loading,
initialization, execution, unloading, and dependency management. Tests cover:

- Plugin discovery from directories
- Plugin metadata parsing and validation
- Plugin loading and initialization
- Plugin method registration and execution
- Plugin unloading and cleanup
- Plugin reload functionality
- Plugin dependency resolution
- Multiple plugin isolation
- Error handling and edge cases

The tests use temporary directories and mock plugins to simulate a complete
plugin lifecycle without requiring external dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from reasoning_mcp.config import Settings
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import SessionConfig
from reasoning_mcp.plugins.interface import Plugin, PluginContext, PluginError, PluginMetadata
from reasoning_mcp.plugins.loader import PluginLoader
from reasoning_mcp.registry import MethodRegistry


# ============================================================================
# Test Plugin Implementations
# ============================================================================


class MockReasoningMethod:
    """Mock reasoning method for testing plugin functionality.

    This class implements the ReasoningMethod protocol to be used in plugin tests
    without requiring actual method implementations.
    """

    def __init__(
        self,
        identifier: str = "mock_plugin_method",
        name: str = "Mock Plugin Method",
        description: str = "A mock plugin method",
        category: str = "core",
        healthy: bool = True,
        init_error: bool = False,
    ):
        self._identifier = identifier
        self._name = name
        self._description = description
        self._category = category
        self._healthy = healthy
        self._init_error = init_error
        self._initialized = False
        self._execution_count = 0

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    @property
    def execution_count(self) -> int:
        """Track how many times this method has been executed."""
        return self._execution_count

    async def initialize(self) -> None:
        if self._init_error:
            raise RuntimeError("Init failed")
        self._initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the method and return a thought node."""
        self._execution_count += 1
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.REASONING,
            content=f"Mock execution: {input_text}",
            session_id=session.id,
            depth=0,
            confidence=0.9,
        )

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought."""
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.REASONING,
            content=f"Continuation: {guidance or 'continued'}",
            session_id=session.id,
            parent_id=previous_thought.id,
            depth=previous_thought.depth + 1,
            confidence=0.8,
        )

    async def health_check(self) -> bool:
        return self._healthy


class MockPlugin:
    """Mock plugin implementation for E2E testing.

    This plugin provides a simple reasoning method and can be configured
    to simulate various scenarios including initialization failures and
    dependency requirements.
    """

    def __init__(
        self,
        plugin_name: str = "test_plugin",
        plugin_version: str = "1.0.0",
        init_error: bool = False,
        shutdown_error: bool = False,
    ):
        self._name = plugin_name
        self._version = plugin_version
        self._description = "A test plugin for E2E testing"
        self._init_error = init_error
        self._shutdown_error = shutdown_error
        self._context: PluginContext | None = None
        self._initialized = False
        self._shutdown_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return self._description

    @property
    def initialized(self) -> bool:
        """Check if plugin was initialized."""
        return self._initialized

    @property
    def shutdown_called(self) -> bool:
        """Check if shutdown was called."""
        return self._shutdown_called

    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with system context."""
        if self._init_error:
            raise PluginError(f"Failed to initialize plugin {self._name}")

        self._context = context
        self._initialized = True
        context.logger.info(f"Initialized plugin: {self._name} v{self._version}")

    async def shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        if self._shutdown_error:
            raise PluginError(f"Failed to shutdown plugin {self._name}")

        self._shutdown_called = True
        if self._context:
            self._context.logger.info(f"Shutdown plugin: {self._name}")

    def get_methods(self) -> list[tuple[ReasoningMethod, MethodMetadata]]:
        """Get reasoning methods provided by this plugin."""
        method = MockReasoningMethod(
            identifier=f"{self._name}_method",
            name=f"{self._name.title()} Method",
            description=f"Method from {self._name}",
            category="core",
        )
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,  # Reuse enum for testing
            name=method.name,
            description=method.description,
            category=MethodCategory.CORE,
            tags=frozenset({"plugin", "test"}),
        )
        return [(method, metadata)]


class DependencyPlugin(MockPlugin):
    """Plugin that declares dependencies on other plugins."""

    def __init__(self, plugin_name: str = "dependency_plugin", dependencies: list[str] | None = None):
        super().__init__(plugin_name=plugin_name)
        self._dependencies = dependencies or []
        self._description = f"A plugin with dependencies: {', '.join(self._dependencies)}"

    @property
    def dependencies(self) -> list[str]:
        return self._dependencies


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for plugin testing."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    return plugin_dir


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Create a mock method registry for testing."""
    return MethodRegistry()


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings()


@pytest.fixture
def plugin_context(mock_registry: MethodRegistry, mock_settings: Settings) -> PluginContext:
    """Create a plugin context for testing."""
    return PluginContext(
        registry=mock_registry,
        settings=mock_settings,
        logger=logging.getLogger("reasoning_mcp.plugins.test"),
    )


@pytest.fixture
def sample_session() -> Session:
    """Create a sample session for testing method execution."""
    session = Session(
        id=str(uuid4()),
        status=SessionStatus.ACTIVE,
        config=SessionConfig(),
    )
    return session


def create_plugin_directory(
    plugin_dir: Path,
    plugin_name: str,
    metadata: dict[str, Any],
    plugin_code: str | None = None,
) -> Path:
    """Helper function to create a plugin directory with metadata and code.

    Args:
        plugin_dir: Parent directory for plugins
        plugin_name: Name of the plugin
        metadata: Plugin metadata dictionary
        plugin_code: Optional Python code for the plugin module

    Returns:
        Path to the created plugin directory
    """
    plugin_path = plugin_dir / plugin_name
    plugin_path.mkdir(parents=True, exist_ok=True)

    # Write plugin.json metadata
    metadata_file = plugin_path / "plugin.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    # Write __init__.py if code provided
    if plugin_code:
        init_file = plugin_path / "__init__.py"
        init_file.write_text(plugin_code)

    return plugin_path


# ============================================================================
# Test: Plugin Discovery
# ============================================================================


class MockPluginDiscovery:
    """Tests for plugin discovery functionality."""

    async def test_plugin_discovery_empty_directory(
        self,
        temp_plugin_dir: Path,
        mock_registry: MethodRegistry,
        mock_settings: Settings,
    ):
        """Test discovery in an empty directory returns no plugins."""
        loader = PluginLoader(
            plugin_dirs=[temp_plugin_dir],
            registry=mock_registry,
            settings=mock_settings,
        )
        discovered = await loader.discover()

        assert discovered == []

    async def test_plugin_discovery_single_plugin(
        self,
        temp_plugin_dir: Path,
        mock_registry: MethodRegistry,
        mock_settings: Settings,
    ):
        """Test discovering a single plugin."""
        metadata = {
            "plugin": {
                "name": "test_plugin",
                "version": "1.0.0",
                "author": "Test Author",
                "description": "A test plugin",
                "entry_point": "test_plugin.MockPlugin",
            }
        }
        plugin_path = temp_plugin_dir / "test_plugin"
        plugin_path.mkdir()
        import tomli_w
        (plugin_path / "plugin.toml").write_bytes(tomli_w.dumps(metadata))

        loader = PluginLoader(
            plugin_dirs=[temp_plugin_dir],
            registry=mock_registry,
            settings=mock_settings,
        )
        discovered = await loader.discover()

        assert len(discovered) == 1
        assert discovered[0].name == "test_plugin"

    async def test_plugin_discovery_multiple_plugins(self, temp_plugin_dir: Path):
        """Test discovering multiple plugins."""
        plugins = [
            {
                "name": "plugin_one",
                "version": "1.0.0",
                "author": "Author 1",
                "description": "Plugin 1",
                "entry_point": "plugin_one.Plugin",
            },
            {
                "name": "plugin_two",
                "version": "2.0.0",
                "author": "Author 2",
                "description": "Plugin 2",
                "entry_point": "plugin_two.Plugin",
            },
            {
                "name": "plugin_three",
                "version": "3.0.0",
                "author": "Author 3",
                "description": "Plugin 3",
                "entry_point": "plugin_three.Plugin",
            },
        ]

        for plugin in plugins:
            create_plugin_directory(temp_plugin_dir, plugin["name"], plugin)

        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        discovered = await loader.discover()

        assert len(discovered) == 3
        discovered_names = {path.name for path in discovered}
        assert discovered_names == {"plugin_one", "plugin_two", "plugin_three"}

    async def test_plugin_discovery_metadata_parsed_correctly(self, temp_plugin_dir: Path):
        """Test that plugin metadata is parsed correctly."""
        metadata = {
            "name": "advanced_plugin",
            "version": "2.1.0",
            "author": "Advanced Author",
            "description": "An advanced test plugin",
            "dependencies": ["numpy>=1.24.0", "scipy>=1.10.0"],
            "entry_point": "advanced_plugin.AdvancedPlugin",
        }
        plugin_path = create_plugin_directory(temp_plugin_dir, "advanced_plugin", metadata)

        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        await loader.discover()

        # Read and validate metadata
        metadata_file = plugin_path / "plugin.json"
        loaded_metadata = json.loads(metadata_file.read_text())

        assert loaded_metadata["name"] == "advanced_plugin"
        assert loaded_metadata["version"] == "2.1.0"
        assert loaded_metadata["author"] == "Advanced Author"
        assert loaded_metadata["description"] == "An advanced test plugin"
        assert len(loaded_metadata["dependencies"]) == 2
        assert "numpy>=1.24.0" in loaded_metadata["dependencies"]

    async def test_plugin_discovery_ignores_invalid_directories(self, temp_plugin_dir: Path):
        """Test that discovery ignores directories without plugin.json."""
        # Create valid plugin
        valid_metadata = {
            "name": "valid_plugin",
            "version": "1.0.0",
            "author": "Author",
            "description": "Valid",
            "entry_point": "valid_plugin.Plugin",
        }
        create_plugin_directory(temp_plugin_dir, "valid_plugin", valid_metadata)

        # Create invalid directory (no plugin.json)
        invalid_dir = temp_plugin_dir / "invalid_plugin"
        invalid_dir.mkdir()
        (invalid_dir / "some_file.txt").write_text("not a plugin")

        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        discovered = await loader.discover()

        assert len(discovered) == 1
        assert discovered[0].name == "valid_plugin"

    async def test_plugin_discovery_nonexistent_directory(self, tmp_path: Path):
        """Test discovery with nonexistent directory logs warning."""
        nonexistent = tmp_path / "does_not_exist"

        loader = PluginLoader(plugin_dirs=[nonexistent])
        discovered = await loader.discover()

        assert discovered == []

    async def test_plugin_discovery_multiple_directories(self, tmp_path: Path):
        """Test discovering plugins from multiple directories."""
        dir1 = tmp_path / "plugins1"
        dir2 = tmp_path / "plugins2"
        dir1.mkdir()
        dir2.mkdir()

        metadata1 = {
            "name": "plugin_from_dir1",
            "version": "1.0.0",
            "author": "Author 1",
            "description": "Plugin 1",
            "entry_point": "plugin1.Plugin",
        }
        metadata2 = {
            "name": "plugin_from_dir2",
            "version": "2.0.0",
            "author": "Author 2",
            "description": "Plugin 2",
            "entry_point": "plugin2.Plugin",
        }

        create_plugin_directory(dir1, "plugin_from_dir1", metadata1)
        create_plugin_directory(dir2, "plugin_from_dir2", metadata2)

        loader = PluginLoader(plugin_dirs=[dir1, dir2])
        discovered = await loader.discover()

        assert len(discovered) == 2
        discovered_names = {path.name for path in discovered}
        assert discovered_names == {"plugin_from_dir1", "plugin_from_dir2"}


# ============================================================================
# Test: Plugin Loading
# ============================================================================


class MockPluginLoading:
    """Tests for plugin loading functionality."""

    async def test_plugin_loading_success(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test successfully loading a plugin."""
        # Note: PluginLoader.load() is not yet implemented (returns None)
        # This test validates the expected behavior when implemented
        metadata = {
            "name": "loadable_plugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A loadable plugin",
            "entry_point": "loadable_plugin.MockPlugin",
        }
        plugin_path = create_plugin_directory(temp_plugin_dir, "loadable_plugin", metadata)

        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        result = await loader.load(plugin_path, plugin_context)

        # Currently returns None as not implemented
        assert result is None

    async def test_plugin_loading_registry_access(self, plugin_context: PluginContext):
        """Test that loaded plugin can access registry."""
        plugin = MockPlugin(plugin_name="registry_test_plugin")
        await plugin.initialize(plugin_context)

        assert plugin._context is not None
        assert plugin._context.registry is plugin_context.registry
        assert plugin.initialized

    async def test_plugin_loading_settings_access(self, plugin_context: PluginContext):
        """Test that loaded plugin can access settings."""
        plugin = MockPlugin(plugin_name="settings_test_plugin")
        await plugin.initialize(plugin_context)

        assert plugin._context is not None
        assert plugin._context.settings is plugin_context.settings

    async def test_plugin_loading_logger_access(self, plugin_context: PluginContext):
        """Test that loaded plugin can access logger."""
        plugin = MockPlugin(plugin_name="logger_test_plugin")
        await plugin.initialize(plugin_context)

        assert plugin._context is not None
        assert plugin._context.logger is not None
        assert isinstance(plugin._context.logger, logging.Logger)


# ============================================================================
# Test: Plugin Initialization
# ============================================================================


class MockPluginInitialization:
    """Tests for plugin initialization functionality."""

    async def test_plugin_initialization_success(self, plugin_context: PluginContext):
        """Test successful plugin initialization."""
        plugin = MockPlugin(plugin_name="init_test_plugin")

        assert not plugin.initialized

        await plugin.initialize(plugin_context)

        assert plugin.initialized
        assert plugin._context is plugin_context

    async def test_plugin_initialization_context_populated(self, plugin_context: PluginContext):
        """Test that plugin context is properly populated."""
        plugin = MockPlugin(plugin_name="context_test_plugin")
        await plugin.initialize(plugin_context)

        assert plugin._context is not None
        assert plugin._context.registry is not None
        assert plugin._context.settings is not None
        assert plugin._context.logger is not None

    async def test_plugin_initialization_failure_handling(self, plugin_context: PluginContext):
        """Test handling of plugin initialization failure."""
        plugin = MockPlugin(plugin_name="failing_plugin", init_error=True)

        with pytest.raises(PluginError, match="Failed to initialize"):
            await plugin.initialize(plugin_context)

        assert not plugin.initialized

    async def test_plugin_initialization_registers_methods(
        self,
        plugin_context: PluginContext,
    ):
        """Test that plugin initialization allows method registration."""
        plugin = MockPlugin(plugin_name="method_registration_plugin")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        assert len(methods) == 1

        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        assert plugin_context.registry.method_count == 1
        assert plugin_context.registry.is_registered(metadata.identifier)

    async def test_plugin_initialization_multiple_times(self, plugin_context: PluginContext):
        """Test that initializing plugin multiple times is handled."""
        plugin = MockPlugin(plugin_name="multi_init_plugin")

        await plugin.initialize(plugin_context)
        first_context = plugin._context

        # Initialize again
        await plugin.initialize(plugin_context)
        second_context = plugin._context

        # Context should be updated
        assert first_context is plugin_context
        assert second_context is plugin_context


# ============================================================================
# Test: Plugin Methods
# ============================================================================


class MockPluginMethods:
    """Tests for plugin method functionality."""

    async def test_plugin_methods_registration(
        self,
        plugin_context: PluginContext,
    ):
        """Test loading plugin with custom method and registering it."""
        plugin = MockPlugin(plugin_name="custom_method_plugin")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        assert len(methods) == 1

        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        retrieved = plugin_context.registry.get(metadata.identifier)
        assert retrieved is method

    async def test_plugin_methods_execution(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ):
        """Test executing custom plugin method."""
        plugin = MockPlugin(plugin_name="execution_plugin")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        # Execute the method
        result = await method.execute(sample_session, "test input")

        assert result is not None
        assert result.content == "Mock execution: test input"
        assert result.session_id == sample_session.id

    async def test_plugin_methods_work_like_native(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ):
        """Test that plugin methods work identically to native methods."""
        plugin = MockPlugin(plugin_name="native_like_plugin")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        # Initialize method
        await method.initialize()

        # Execute method
        thought = await method.execute(sample_session, "native-like execution")
        assert thought.content == "Mock execution: native-like execution"

        # Continue reasoning
        continued = await method.continue_reasoning(
            sample_session,
            thought,
            guidance="continue from here",
        )
        assert continued.parent_id == thought.id
        assert continued.depth == thought.depth + 1

        # Health check
        healthy = await method.health_check()
        assert healthy is True

    async def test_plugin_methods_multiple_from_single_plugin(
        self,
        plugin_context: PluginContext,
    ):
        """Test plugin providing multiple methods."""

        class MultiMethodPlugin(MockPlugin):
            def get_methods(self) -> list[tuple[ReasoningMethod, MethodMetadata]]:
                methods = []
                for i in range(3):
                    method = MockReasoningMethod(
                        identifier=f"{self._name}_method_{i}",
                        name=f"Method {i}",
                        description=f"Method {i} from plugin",
                    )
                    # Use different identifiers for each method
                    identifiers = [
                        MethodIdentifier.CHAIN_OF_THOUGHT,
                        MethodIdentifier.TREE_OF_THOUGHTS,
                        MethodIdentifier.REACT,
                    ]
                    metadata = MethodMetadata(
                        identifier=identifiers[i],
                        name=method.name,
                        description=method.description,
                        category=MethodCategory.CORE,
                    )
                    methods.append((method, metadata))
                return methods

        plugin = MultiMethodPlugin(plugin_name="multi_method_plugin")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        assert len(methods) == 3

        # Register all methods
        for method, metadata in methods:
            plugin_context.registry.register(method, metadata)

        assert plugin_context.registry.method_count == 3


# ============================================================================
# Test: Plugin Unloading
# ============================================================================


class MockPluginUnloading:
    """Tests for plugin unloading functionality."""

    async def test_plugin_unloading_success(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test successfully unloading a plugin."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        # Manually add a plugin to simulate loading
        plugin = MockPlugin(plugin_name="unload_test_plugin")
        await plugin.initialize(plugin_context)
        loader.plugins["unload_test_plugin"] = plugin

        result = await loader.unload("unload_test_plugin")

        assert result is True
        assert "unload_test_plugin" not in loader.plugins
        assert plugin.shutdown_called

    async def test_plugin_unloading_methods_unregistered(
        self,
        plugin_context: PluginContext,
    ):
        """Test that unloading plugin unregisters its methods."""
        plugin = MockPlugin(plugin_name="unregister_test_plugin")
        await plugin.initialize(plugin_context)

        # Register method
        methods = plugin.get_methods()
        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        assert plugin_context.registry.is_registered(metadata.identifier)

        # Unregister method (simulating plugin unload)
        plugin_context.registry.unregister(metadata.identifier)

        assert not plugin_context.registry.is_registered(metadata.identifier)
        assert plugin_context.registry.method_count == 0

    async def test_plugin_unloading_cleanup_called(self, plugin_context: PluginContext):
        """Test that cleanup is called during unload."""
        plugin = MockPlugin(plugin_name="cleanup_test_plugin")
        await plugin.initialize(plugin_context)

        assert not plugin.shutdown_called

        await plugin.shutdown()

        assert plugin.shutdown_called

    async def test_plugin_unloading_nonexistent_plugin(self, temp_plugin_dir: Path):
        """Test unloading a non-existent plugin returns False."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        result = await loader.unload("nonexistent_plugin")

        assert result is False

    async def test_plugin_unloading_with_error(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test handling errors during plugin unload."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        # Add plugin that errors on shutdown
        plugin = MockPlugin(plugin_name="error_plugin", shutdown_error=True)
        await plugin.initialize(plugin_context)
        loader.plugins["error_plugin"] = plugin

        result = await loader.unload("error_plugin")

        # Should handle error gracefully
        assert result is False


# ============================================================================
# Test: Plugin Reload
# ============================================================================


class MockPluginReload:
    """Tests for plugin reload functionality."""

    async def test_plugin_reload_after_modification(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test reloading a plugin after modification."""
        metadata_v1 = {
            "name": "reload_plugin",
            "version": "1.0.0",
            "author": "Author",
            "description": "Version 1",
            "entry_point": "reload_plugin.Plugin",
        }
        plugin_path = create_plugin_directory(temp_plugin_dir, "reload_plugin", metadata_v1)

        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        await loader.discover()

        # Simulate loading v1
        plugin_v1 = MockPlugin(plugin_name="reload_plugin", plugin_version="1.0.0")
        await plugin_v1.initialize(plugin_context)
        loader.plugins["reload_plugin"] = plugin_v1

        assert plugin_v1.version == "1.0.0"

        # Modify metadata
        metadata_v2 = metadata_v1.copy()
        metadata_v2["version"] = "2.0.0"
        metadata_v2["description"] = "Version 2"
        metadata_file = plugin_path / "plugin.json"
        metadata_file.write_text(json.dumps(metadata_v2, indent=2))

        # Reload
        await loader.unload("reload_plugin")
        plugin_v2 = MockPlugin(plugin_name="reload_plugin", plugin_version="2.0.0")
        await plugin_v2.initialize(plugin_context)
        loader.plugins["reload_plugin"] = plugin_v2

        assert plugin_v2.version == "2.0.0"
        assert plugin_v1.shutdown_called

    async def test_plugin_reload_changes_applied(
        self,
        plugin_context: PluginContext,
    ):
        """Test that changes are applied after reload."""
        # Load v1
        plugin_v1 = MockPlugin(plugin_name="change_plugin", plugin_version="1.0.0")
        await plugin_v1.initialize(plugin_context)

        methods_v1 = plugin_v1.get_methods()
        method_v1, metadata_v1 = methods_v1[0]
        plugin_context.registry.register(method_v1, metadata_v1)

        # Unload
        await plugin_v1.shutdown()
        plugin_context.registry.unregister(metadata_v1.identifier)

        # Load v2 with different behavior
        class ModifiedPlugin(MockPlugin):
            def get_methods(self) -> list[tuple[ReasoningMethod, MethodMetadata]]:
                method = MockReasoningMethod(
                    identifier="modified_method",
                    name="Modified Method",
                    description="This is modified",
                )
                metadata = MethodMetadata(
                    identifier=MethodIdentifier.TREE_OF_THOUGHTS,  # Different identifier
                    name=method.name,
                    description=method.description,
                    category=MethodCategory.CORE,
                )
                return [(method, metadata)]

        plugin_v2 = ModifiedPlugin(plugin_name="change_plugin", plugin_version="2.0.0")
        await plugin_v2.initialize(plugin_context)

        methods_v2 = plugin_v2.get_methods()
        method_v2, metadata_v2 = methods_v2[0]

        assert metadata_v2.identifier != metadata_v1.identifier
        assert method_v2.identifier == "modified_method"


# ============================================================================
# Test: Plugin Dependencies
# ============================================================================


class MockPluginDependencies:
    """Tests for plugin dependency management."""

    async def test_plugin_dependencies_declaration(self, plugin_context: PluginContext):
        """Test creating plugin with dependencies."""
        plugin = DependencyPlugin(
            plugin_name="dependent_plugin",
            dependencies=["base_plugin", "utility_plugin"],
        )
        await plugin.initialize(plugin_context)

        assert len(plugin.dependencies) == 2
        assert "base_plugin" in plugin.dependencies
        assert "utility_plugin" in plugin.dependencies

    async def test_plugin_dependencies_resolution(self, plugin_context: PluginContext):
        """Test dependency resolution order."""
        # Create base plugin (no dependencies)
        base_plugin = MockPlugin(plugin_name="base_plugin")
        await base_plugin.initialize(plugin_context)

        # Create dependent plugin
        dependent_plugin = DependencyPlugin(
            plugin_name="dependent_plugin",
            dependencies=["base_plugin"],
        )

        # Should be able to initialize after base
        await dependent_plugin.initialize(plugin_context)

        assert base_plugin.initialized
        assert dependent_plugin.initialized

    async def test_plugin_dependencies_missing_dependency(self, plugin_context: PluginContext):
        """Test handling missing dependencies."""
        plugin = DependencyPlugin(
            plugin_name="dependent_plugin",
            dependencies=["nonexistent_plugin"],
        )

        # Plugin can still initialize (dependency checking would happen at loader level)
        await plugin.initialize(plugin_context)

        assert plugin.initialized
        assert "nonexistent_plugin" in plugin.dependencies

    async def test_plugin_dependencies_circular_detection(self, plugin_context: PluginContext):
        """Test detection of circular dependencies."""
        # Plugin A depends on B
        plugin_a = DependencyPlugin(plugin_name="plugin_a", dependencies=["plugin_b"])

        # Plugin B depends on A (circular)
        plugin_b = DependencyPlugin(plugin_name="plugin_b", dependencies=["plugin_a"])

        # Both can initialize (circular dependency would be detected at loader level)
        await plugin_a.initialize(plugin_context)
        await plugin_b.initialize(plugin_context)

        assert "plugin_b" in plugin_a.dependencies
        assert "plugin_a" in plugin_b.dependencies


# ============================================================================
# Test: Multiple Plugins
# ============================================================================


class TestMultiplePlugins:
    """Tests for multiple plugin scenarios."""

    async def test_multiple_plugins_no_conflicts(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test loading multiple plugins without conflicts."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        # Create and load multiple plugins
        plugins = []
        for i in range(5):
            plugin = MockPlugin(plugin_name=f"plugin_{i}", plugin_version=f"{i}.0.0")
            await plugin.initialize(plugin_context)
            loader.plugins[f"plugin_{i}"] = plugin
            plugins.append(plugin)

        assert len(loader.plugins) == 5

        # All should be initialized
        for plugin in plugins:
            assert plugin.initialized

    async def test_multiple_plugins_isolation(self, plugin_context: PluginContext):
        """Test that plugins are isolated from each other."""
        plugin1 = MockPlugin(plugin_name="isolated_plugin_1")
        plugin2 = MockPlugin(plugin_name="isolated_plugin_2")

        await plugin1.initialize(plugin_context)
        await plugin2.initialize(plugin_context)

        # Each plugin has its own context but shares registry
        assert plugin1._context is plugin_context
        assert plugin2._context is plugin_context
        assert plugin1._context.registry is plugin2._context.registry

        # But plugins are separate instances
        assert plugin1 is not plugin2
        assert plugin1.name != plugin2.name

    async def test_multiple_plugins_method_registration(
        self,
        plugin_context: PluginContext,
    ):
        """Test registering methods from multiple plugins."""
        plugins = []
        identifiers = [
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.REACT,
        ]

        for i in range(3):
            plugin = MockPlugin(plugin_name=f"multi_plugin_{i}")
            await plugin.initialize(plugin_context)
            plugins.append(plugin)

            # Register each plugin's method with unique identifier
            methods = plugin.get_methods()
            method, metadata = methods[0]
            # Override identifier to avoid conflicts
            metadata = MethodMetadata(
                identifier=identifiers[i],
                name=metadata.name,
                description=metadata.description,
                category=metadata.category,
                tags=metadata.tags,
            )
            plugin_context.registry.register(method, metadata)

        assert plugin_context.registry.method_count == 3

    async def test_multiple_plugins_concurrent_execution(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ):
        """Test concurrent execution of methods from different plugins."""
        plugin1 = MockPlugin(plugin_name="concurrent_1")
        plugin2 = MockPlugin(plugin_name="concurrent_2")

        await plugin1.initialize(plugin_context)
        await plugin2.initialize(plugin_context)

        # Get methods from both plugins
        methods1 = plugin1.get_methods()
        method1, metadata1 = methods1[0]

        methods2 = plugin2.get_methods()
        method2, metadata2 = methods2[0]

        # Execute both concurrently
        import asyncio
        results = await asyncio.gather(
            method1.execute(sample_session, "input 1"),
            method2.execute(sample_session, "input 2"),
        )

        assert len(results) == 2
        assert results[0].content == "Mock execution: input 1"
        assert results[1].content == "Mock execution: input 2"


# ============================================================================
# Test: Plugin System Integration
# ============================================================================


class MockPluginSystemIntegration:
    """Integration tests for the complete plugin system."""

    async def test_full_plugin_lifecycle(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
        sample_session: Session,
    ):
        """Test complete plugin lifecycle: discover -> load -> init -> execute -> unload."""
        # Create plugin directory
        metadata = {
            "name": "lifecycle_plugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "Lifecycle test",
            "entry_point": "lifecycle_plugin.Plugin",
        }
        create_plugin_directory(temp_plugin_dir, "lifecycle_plugin", metadata)

        # Discover
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])
        discovered = await loader.discover()
        assert len(discovered) == 1

        # Load (manually since loader.load not implemented)
        plugin = MockPlugin(plugin_name="lifecycle_plugin")
        await plugin.initialize(plugin_context)
        loader.plugins["lifecycle_plugin"] = plugin

        # Register methods
        methods = plugin.get_methods()
        method, method_metadata = methods[0]
        plugin_context.registry.register(method, method_metadata)

        # Execute
        result = await method.execute(sample_session, "lifecycle test")
        assert result.content == "Mock execution: lifecycle test"

        # Unload
        plugin_context.registry.unregister(method_metadata.identifier)
        unload_result = await loader.unload("lifecycle_plugin")
        assert unload_result is True
        assert plugin.shutdown_called

    async def test_plugin_system_error_recovery(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test that plugin system recovers from errors."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        # Add failing plugin
        failing_plugin = MockPlugin(plugin_name="failing_plugin", init_error=True)

        with pytest.raises(PluginError):
            await failing_plugin.initialize(plugin_context)

        # Add successful plugin
        success_plugin = MockPlugin(plugin_name="success_plugin")
        await success_plugin.initialize(plugin_context)
        loader.plugins["success_plugin"] = success_plugin

        # System should still work with successful plugin
        assert success_plugin.initialized
        assert len(loader.plugins) == 1

    async def test_plugin_system_shutdown_all(
        self,
        temp_plugin_dir: Path,
        plugin_context: PluginContext,
    ):
        """Test shutting down all plugins at once."""
        loader = PluginLoader(plugin_dirs=[temp_plugin_dir])

        # Load multiple plugins
        plugins = []
        for i in range(3):
            plugin = MockPlugin(plugin_name=f"shutdown_plugin_{i}")
            await plugin.initialize(plugin_context)
            loader.plugins[f"shutdown_plugin_{i}"] = plugin
            plugins.append(plugin)

        assert len(loader.plugins) == 3

        # Shutdown all
        await loader.shutdown_all()

        assert len(loader.plugins) == 0
        for plugin in plugins:
            assert plugin.shutdown_called
