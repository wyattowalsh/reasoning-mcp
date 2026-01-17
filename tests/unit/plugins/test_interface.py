"""Tests for plugin interface module."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from reasoning_mcp.plugins.interface import (
    Plugin,
    PluginContext,
    PluginError,
    PluginMetadata,
)

if TYPE_CHECKING:
    from reasoning_mcp.config import Settings
    from reasoning_mcp.registry import MethodRegistry


class TestPluginError:
    """Tests for PluginError exception."""

    def test_plugin_error_is_exception(self) -> None:
        """Test PluginError is an Exception."""
        err = PluginError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_plugin_error_with_cause(self) -> None:
        """Test PluginError with a cause."""
        cause = ValueError("original error")
        err = PluginError("wrapper error")
        err.__cause__ = cause
        assert err.__cause__ is cause


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_valid_metadata_creation(self) -> None:
        """Test creating valid PluginMetadata."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            dependencies=["numpy>=1.24.0"],
            entry_point="test_plugin.plugin.TestPlugin",
        )

        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "A test plugin"
        assert metadata.dependencies == ["numpy>=1.24.0"]
        assert metadata.entry_point == "test_plugin.plugin.TestPlugin"

    def test_metadata_default_dependencies(self) -> None:
        """Test PluginMetadata with default dependencies."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
        )

        assert metadata.dependencies == []
        assert metadata.entry_point == ""

    def test_metadata_is_frozen(self) -> None:
        """Test PluginMetadata is immutable (frozen)."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
        )

        with pytest.raises(AttributeError):
            metadata.name = "new-name"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PluginMetadata(
                name="",
                version="1.0.0",
                author="Test Author",
                description="A test plugin",
            )
        assert "name cannot be empty" in str(exc_info.value)

    def test_empty_version_raises(self) -> None:
        """Test empty version raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PluginMetadata(
                name="test-plugin",
                version="",
                author="Test Author",
                description="A test plugin",
            )
        assert "version cannot be empty" in str(exc_info.value)

    def test_empty_description_raises(self) -> None:
        """Test empty description raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PluginMetadata(
                name="test-plugin",
                version="1.0.0",
                author="Test Author",
                description="",
            )
        assert "description cannot be empty" in str(exc_info.value)


class TestPluginContext:
    """Tests for PluginContext dataclass."""

    def test_plugin_context_creation(self) -> None:
        """Test creating PluginContext."""
        registry = MagicMock()
        settings = MagicMock()
        logger = MagicMock(spec=logging.Logger)

        context = PluginContext(
            registry=registry,
            settings=settings,
            logger=logger,
        )

        assert context.registry is registry
        assert context.settings is settings
        assert context.logger is logger


class TestPluginProtocol:
    """Tests for Plugin protocol."""

    def test_minimal_plugin_implementation(self) -> None:
        """Test a minimal plugin implementation satisfies Protocol."""

        class MinimalPlugin:
            """Minimal plugin implementation."""

            @property
            def name(self) -> str:
                return "minimal"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Minimal plugin"

            async def initialize(self, context: PluginContext) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            def get_methods(self) -> list:
                return []

        plugin = MinimalPlugin()

        # Verify protocol compliance at runtime
        assert isinstance(plugin, Plugin)
        assert plugin.name == "minimal"
        assert plugin.version == "1.0.0"
        assert plugin.description == "Minimal plugin"
        assert plugin.get_methods() == []

    def test_full_plugin_implementation(self) -> None:
        """Test a full plugin implementation with methods."""

        class FullPlugin:
            """Full plugin implementation."""

            def __init__(self) -> None:
                self._context: PluginContext | None = None
                self._initialized = False

            @property
            def name(self) -> str:
                return "full-plugin"

            @property
            def version(self) -> str:
                return "2.0.0"

            @property
            def description(self) -> str:
                return "A fully featured plugin"

            async def initialize(self, context: PluginContext) -> None:
                self._context = context
                self._initialized = True
                context.logger.info(f"Initialized {self.name}")

            async def shutdown(self) -> None:
                if self._context:
                    self._context.logger.info(f"Shutting down {self.name}")
                self._initialized = False

            def get_methods(self) -> list:
                # Return mock methods for testing
                return [("method1", "metadata1"), ("method2", "metadata2")]

        plugin = FullPlugin()

        assert isinstance(plugin, Plugin)
        assert len(plugin.get_methods()) == 2

    def test_non_conforming_class_fails_protocol_check(self) -> None:
        """Test that a non-conforming class fails Protocol check."""

        class IncompletePlugin:
            """Plugin missing required methods."""

            @property
            def name(self) -> str:
                return "incomplete"

            # Missing other required members

        plugin = IncompletePlugin()

        # Should not be recognized as implementing Plugin protocol
        assert not isinstance(plugin, Plugin)

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self) -> None:
        """Test plugin lifecycle: initialize -> use -> shutdown."""

        class LifecyclePlugin:
            """Plugin to test lifecycle."""

            def __init__(self) -> None:
                self._context: PluginContext | None = None
                self._log: list[str] = []

            @property
            def name(self) -> str:
                return "lifecycle"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Lifecycle test plugin"

            async def initialize(self, context: PluginContext) -> None:
                self._context = context
                self._log.append("initialized")

            async def shutdown(self) -> None:
                self._log.append("shutdown")
                self._context = None

            def get_methods(self) -> list:
                self._log.append("get_methods")
                return []

        plugin = LifecyclePlugin()

        # Create mock context
        context = PluginContext(
            registry=MagicMock(),
            settings=MagicMock(),
            logger=MagicMock(spec=logging.Logger),
        )

        # Test lifecycle
        await plugin.initialize(context)
        assert "initialized" in plugin._log

        plugin.get_methods()
        assert "get_methods" in plugin._log

        await plugin.shutdown()
        assert "shutdown" in plugin._log
        assert plugin._context is None

    def test_plugin_with_error_handling(self) -> None:
        """Test plugin that raises PluginError."""

        class ErrorPlugin:
            """Plugin that raises errors."""

            @property
            def name(self) -> str:
                return "error-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Plugin that raises errors"

            async def initialize(self, context: PluginContext) -> None:
                raise PluginError("Initialization failed")

            async def shutdown(self) -> None:
                raise PluginError("Shutdown failed")

            def get_methods(self) -> list:
                return []

        plugin = ErrorPlugin()
        assert isinstance(plugin, Plugin)
