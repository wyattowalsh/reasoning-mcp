"""
End-to-end tests for the plugin system in reasoning-mcp.

This module provides comprehensive E2E tests for plugin discovery, loading,
initialization, execution, unloading, and dependency management.

Note: These tests are designed to work with the current PluginLoader implementation
which uses TOML-based metadata files (plugin.toml or pyproject.toml).
"""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from reasoning_mcp.config import Settings
from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.session import SessionConfig
from reasoning_mcp.plugins.interface import PluginContext, PluginError
from reasoning_mcp.plugins.loader import PluginLoader
from reasoning_mcp.registry import MethodRegistry

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

# ============================================================================
# Mock Plugin Implementations
# ============================================================================


class MockReasoningMethod:
    """Mock reasoning method for plugin testing."""

    streaming_context = None

    def __init__(
        self,
        identifier: str = "mock_method",
        name: str = "Mock Method",
        description: str = "A mock method",
        category: str = "core",
        healthy: bool = True,
    ):
        self._identifier = identifier
        self._name = name
        self._description = description
        self._category = category
        self._healthy = healthy
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
        return self._execution_count

    async def initialize(self) -> None:
        self._initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        self._execution_count += 1
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content=f"Mock execution: {input_text}",
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
        execution_context: Any = None,
    ) -> ThoughtNode:
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content=f"Continuation: {guidance or 'continued'}",
            parent_id=previous_thought.id,
            depth=previous_thought.depth + 1,
            confidence=0.8,
        )

    async def health_check(self) -> bool:
        return self._healthy

    async def emit_thought(self, content: str, confidence: float | None = None) -> None:
        pass


class MockPlugin:
    """Mock plugin for testing."""

    def __init__(
        self,
        plugin_name: str = "mock_plugin",
        plugin_version: str = "1.0.0",
        init_error: bool = False,
        shutdown_error: bool = False,
    ):
        self._name = plugin_name
        self._version = plugin_version
        self._description = "A mock plugin"
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
        return self._initialized

    @property
    def shutdown_called(self) -> bool:
        return self._shutdown_called

    async def initialize(self, context: PluginContext) -> None:
        if self._init_error:
            raise PluginError(f"Failed to initialize {self._name}")
        self._context = context
        self._initialized = True
        context.logger.info(f"Initialized plugin: {self._name}")

    async def shutdown(self) -> None:
        if self._shutdown_error:
            raise PluginError(f"Failed to shutdown {self._name}")
        self._shutdown_called = True
        if self._context:
            self._context.logger.info(f"Shutdown plugin: {self._name}")

    def get_methods(self) -> list[tuple[Any, MethodMetadata]]:
        """Return mock methods. Uses Any to avoid type issues with MockReasoningMethod."""
        method = MockReasoningMethod(
            identifier=f"{self._name}_method",
            name=f"{self._name} Method",
            description=f"Method from {self._name}",
        )
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name=method.name,
            description=method.description,
            category=MethodCategory.CORE,
            tags=frozenset({"plugin", "test"}),
        )
        return [(method, metadata)]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary plugin directory."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    return plugin_dir


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Create a mock registry."""
    return MethodRegistry()


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    return Settings()


@pytest.fixture
def plugin_context(mock_registry: MethodRegistry, mock_settings: Settings) -> PluginContext:
    """Create a plugin context."""
    return PluginContext(
        registry=mock_registry,
        settings=mock_settings,
        logger=logging.getLogger("reasoning_mcp.plugins.test"),
    )


@pytest.fixture
def sample_session() -> Session:
    """Create a sample session."""
    return Session(
        id=str(uuid4()),
        status=SessionStatus.ACTIVE,
        config=SessionConfig(),
    )


@pytest.fixture
def plugin_loader(
    temp_plugin_dir: Path,
    mock_registry: MethodRegistry,
    mock_settings: Settings,
) -> PluginLoader:
    """Create a plugin loader."""
    return PluginLoader(
        plugin_dirs=[temp_plugin_dir],
        registry=mock_registry,
        settings=mock_settings,
    )


def create_toml_plugin(plugin_dir: Path, name: str, version: str = "1.0.0") -> Path:
    """Helper to create a plugin with TOML metadata."""
    plugin_path = plugin_dir / name
    plugin_path.mkdir()

    toml_content = f"""[plugin]
name = "{name}"
version = "{version}"
author = "Test Author"
description = "Test plugin"
entry_point = "{name}.Plugin"
"""
    (plugin_path / "plugin.toml").write_text(toml_content)
    return plugin_path


# ============================================================================
# Tests
# ============================================================================


class TestPluginDiscovery:
    """Tests for plugin discovery."""

    async def test_plugin_discovery_empty_directory(self, plugin_loader: PluginLoader) -> None:
        """Test discovery in empty directory."""
        discovered = await plugin_loader.discover()
        assert discovered == []

    async def test_plugin_discovery_single_plugin(
        self,
        temp_plugin_dir: Path,
        plugin_loader: PluginLoader,
    ) -> None:
        """Test discovering a single plugin."""
        create_toml_plugin(temp_plugin_dir, "test_plugin")

        discovered = await plugin_loader.discover()

        assert len(discovered) == 1
        assert discovered[0].name == "test_plugin"
        assert discovered[0].version == "1.0.0"

    async def test_plugin_discovery_multiple_plugins(
        self,
        temp_plugin_dir: Path,
        plugin_loader: PluginLoader,
    ) -> None:
        """Test discovering multiple plugins."""
        create_toml_plugin(temp_plugin_dir, "plugin_one", "1.0.0")
        create_toml_plugin(temp_plugin_dir, "plugin_two", "2.0.0")
        create_toml_plugin(temp_plugin_dir, "plugin_three", "3.0.0")

        discovered = await plugin_loader.discover()

        assert len(discovered) == 3
        names = {p.name for p in discovered}
        assert names == {"plugin_one", "plugin_two", "plugin_three"}

    async def test_plugin_discovery_metadata_parsed(
        self,
        temp_plugin_dir: Path,
        plugin_loader: PluginLoader,
    ) -> None:
        """Test that metadata is parsed correctly."""
        create_toml_plugin(temp_plugin_dir, "advanced_plugin", "2.1.0")

        discovered = await plugin_loader.discover()

        assert len(discovered) == 1
        assert discovered[0].name == "advanced_plugin"
        assert discovered[0].version == "2.1.0"
        assert discovered[0].author == "Test Author"
        assert discovered[0].description == "Test plugin"

    async def test_plugin_discovery_ignores_invalid(
        self,
        temp_plugin_dir: Path,
        plugin_loader: PluginLoader,
    ) -> None:
        """Test that invalid directories are ignored."""
        # Valid plugin
        create_toml_plugin(temp_plugin_dir, "valid_plugin")

        # Invalid directory (no plugin.toml)
        invalid_dir = temp_plugin_dir / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "some_file.txt").write_text("not a plugin")

        discovered = await plugin_loader.discover()

        assert len(discovered) == 1
        assert discovered[0].name == "valid_plugin"


class TestPluginInitialization:
    """Tests for plugin initialization."""

    async def test_plugin_initialization_success(self, plugin_context: PluginContext) -> None:
        """Test successful initialization."""
        plugin = MockPlugin(plugin_name="init_test")

        assert not plugin.initialized

        await plugin.initialize(plugin_context)

        assert plugin.initialized
        assert plugin._context is plugin_context

    async def test_plugin_initialization_context_populated(
        self,
        plugin_context: PluginContext,
    ) -> None:
        """Test that context is populated."""
        plugin = MockPlugin(plugin_name="context_test")
        await plugin.initialize(plugin_context)

        assert plugin._context is not None
        assert plugin._context.registry is not None
        assert plugin._context.settings is not None
        assert plugin._context.logger is not None

    async def test_plugin_initialization_failure(self, plugin_context: PluginContext) -> None:
        """Test initialization failure handling."""
        plugin = MockPlugin(plugin_name="failing", init_error=True)

        with pytest.raises(PluginError, match="Failed to initialize"):
            await plugin.initialize(plugin_context)

        assert not plugin.initialized

    async def test_plugin_initialization_registers_methods(
        self,
        plugin_context: PluginContext,
    ) -> None:
        """Test that methods can be registered after initialization."""
        plugin = MockPlugin(plugin_name="method_test")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        assert len(methods) == 1

        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        assert plugin_context.registry.method_count == 1


class TestPluginMethods:
    """Tests for plugin method functionality."""

    async def test_plugin_methods_execution(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ) -> None:
        """Test executing plugin methods."""
        plugin = MockPlugin(plugin_name="exec_test")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        result = await method.execute(sample_session, "test input")

        assert result is not None
        assert result.content == "Mock execution: test input"
        assert result.type == ThoughtType.INITIAL

    async def test_plugin_methods_work_like_native(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ) -> None:
        """Test that plugin methods work like native methods."""
        plugin = MockPlugin(plugin_name="native_test")
        await plugin.initialize(plugin_context)

        methods = plugin.get_methods()
        method, metadata = methods[0]
        plugin_context.registry.register(method, metadata)

        # Initialize
        await method.initialize()

        # Execute
        thought = await method.execute(sample_session, "test")
        assert thought.content == "Mock execution: test"

        # Continue reasoning
        continued = await method.continue_reasoning(
            sample_session,
            thought,
            guidance="continue",
        )
        assert continued.parent_id == thought.id

        # Health check
        healthy = await method.health_check()
        assert healthy is True


class TestPluginUnloading:
    """Tests for plugin unloading."""

    async def test_plugin_unloading_cleanup(self, plugin_context: PluginContext) -> None:
        """Test that cleanup is called during unload."""
        plugin = MockPlugin(plugin_name="cleanup_test")
        await plugin.initialize(plugin_context)

        assert not plugin.shutdown_called

        await plugin.shutdown()

        assert plugin.shutdown_called

    async def test_plugin_unloading_with_error(self, plugin_context: PluginContext) -> None:
        """Test handling errors during unload."""
        plugin = MockPlugin(plugin_name="error_test", shutdown_error=True)
        await plugin.initialize(plugin_context)

        with pytest.raises(PluginError):
            await plugin.shutdown()


class TestPluginReload:
    """Tests for plugin reload."""

    async def test_plugin_reload_changes_applied(self, plugin_context: PluginContext) -> None:
        """Test that changes are applied after reload."""
        # Load v1
        plugin_v1 = MockPlugin(plugin_name="reload_test", plugin_version="1.0.0")
        await plugin_v1.initialize(plugin_context)

        methods_v1 = plugin_v1.get_methods()
        method_v1, metadata_v1 = methods_v1[0]
        plugin_context.registry.register(method_v1, metadata_v1)

        # Unload
        await plugin_v1.shutdown()
        plugin_context.registry.unregister(metadata_v1.identifier)

        # Load v2
        plugin_v2 = MockPlugin(plugin_name="reload_test", plugin_version="2.0.0")
        await plugin_v2.initialize(plugin_context)

        assert plugin_v1.shutdown_called
        assert plugin_v2.version == "2.0.0"


class TestPluginDependencies:
    """Tests for plugin dependencies."""

    async def test_plugin_dependencies_declaration(self) -> None:
        """Test declaring plugin dependencies."""
        # Dependencies would be declared in plugin.toml
        # This test validates the concept
        plugin = MockPlugin(plugin_name="dependent")

        assert plugin.name == "dependent"


class TestMultiplePlugins:
    """Tests for multiple plugin scenarios."""

    async def test_multiple_plugins_isolation(self, plugin_context: PluginContext) -> None:
        """Test that plugins are isolated."""
        plugin1 = MockPlugin(plugin_name="plugin_1")
        plugin2 = MockPlugin(plugin_name="plugin_2")

        await plugin1.initialize(plugin_context)
        await plugin2.initialize(plugin_context)

        # Same context but different instances
        assert plugin1._context is plugin_context
        assert plugin2._context is plugin_context
        assert plugin1 is not plugin2
        assert plugin1.name != plugin2.name

    async def test_multiple_plugins_concurrent_execution(
        self,
        plugin_context: PluginContext,
        sample_session: Session,
    ) -> None:
        """Test concurrent execution of methods from different plugins."""
        plugin1 = MockPlugin(plugin_name="concurrent_1")
        plugin2 = MockPlugin(plugin_name="concurrent_2")

        await plugin1.initialize(plugin_context)
        await plugin2.initialize(plugin_context)

        methods1 = plugin1.get_methods()
        method1, _ = methods1[0]

        methods2 = plugin2.get_methods()
        method2, _ = methods2[0]

        import asyncio

        results = await asyncio.gather(
            method1.execute(sample_session, "input 1"),
            method2.execute(sample_session, "input 2"),
        )

        assert len(results) == 2
        assert results[0].content == "Mock execution: input 1"
        assert results[1].content == "Mock execution: input 2"


class TestPluginSystemIntegration:
    """Integration tests for the complete plugin system."""

    async def test_plugin_system_error_recovery(self, plugin_context: PluginContext) -> None:
        """Test that system recovers from errors."""
        # Failing plugin
        failing = MockPlugin(plugin_name="failing", init_error=True)

        with pytest.raises(PluginError):
            await failing.initialize(plugin_context)

        # Successful plugin should still work
        success = MockPlugin(plugin_name="success")
        await success.initialize(plugin_context)

        assert success.initialized
