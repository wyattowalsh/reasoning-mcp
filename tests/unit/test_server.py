"""Tests for reasoning_mcp.server module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.server import (
    AppContext,
    app_lifespan,
    get_auth_provider,
    get_sampling_handler,
    mcp,
)


class TestServerInstance:
    """Tests for FastMCP server instance."""

    def test_server_exists(self):
        """Test that mcp server instance exists."""
        assert mcp is not None

    def test_server_name(self):
        """Test that server has correct name."""
        assert mcp.name == "reasoning-mcp"

    def test_server_has_lifespan(self):
        """Test that server has lifespan configured."""
        # FastMCP doesn't expose lifespan as a public attribute,
        # but we can verify it was passed by checking the server works
        # and that app_lifespan exists and is callable
        assert callable(app_lifespan)


class TestAppContext:
    """Tests for AppContext dataclass."""

    def test_appcontext_creation(self):
        """Test AppContext can be created with required fields."""
        # Create mock objects for the required fields
        mock_registry = MagicMock()
        mock_session_manager = MagicMock()
        mock_settings = MagicMock()

        ctx = AppContext(
            registry=mock_registry,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

        assert ctx.registry is mock_registry
        assert ctx.session_manager is mock_session_manager
        assert ctx.settings is mock_settings
        assert ctx.initialized is False

    def test_appcontext_initialized_default(self):
        """Test AppContext initialized field defaults to False."""
        mock_registry = MagicMock()
        mock_session_manager = MagicMock()
        mock_settings = MagicMock()

        ctx = AppContext(
            registry=mock_registry,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

        assert ctx.initialized is False

    def test_appcontext_initialized_explicit(self):
        """Test AppContext initialized field can be set explicitly."""
        mock_registry = MagicMock()
        mock_session_manager = MagicMock()
        mock_settings = MagicMock()

        ctx = AppContext(
            registry=mock_registry,
            session_manager=mock_session_manager,
            settings=mock_settings,
            initialized=True,
        )

        assert ctx.initialized is True

    def test_appcontext_has_required_attributes(self):
        """Test AppContext has all expected attributes."""
        mock_registry = MagicMock()
        mock_session_manager = MagicMock()
        mock_settings = MagicMock()

        ctx = AppContext(
            registry=mock_registry,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

        assert hasattr(ctx, "registry")
        assert hasattr(ctx, "session_manager")
        assert hasattr(ctx, "settings")
        assert hasattr(ctx, "initialized")


class TestAppLifespan:
    """Tests for app_lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_creates_context(self):
        """Test that lifespan creates and yields AppContext."""
        mock_server = MagicMock()

        async with app_lifespan(mock_server) as ctx:
            assert isinstance(ctx, AppContext)
            assert ctx.registry is not None
            assert ctx.session_manager is not None
            assert ctx.settings is not None

    @pytest.mark.asyncio
    async def test_lifespan_initializes_context(self):
        """Test that lifespan initializes the context."""
        mock_server = MagicMock()

        async with app_lifespan(mock_server) as ctx:
            assert ctx.initialized is True

    @pytest.mark.asyncio
    async def test_lifespan_initializes_registry(self):
        """Test that lifespan initializes the registry."""
        mock_server = MagicMock()

        with patch("reasoning_mcp.registry.MethodRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.initialize = AsyncMock()
            mock_registry_class.return_value = mock_registry

            async with app_lifespan(mock_server) as ctx:
                # Verify initialize was called
                mock_registry.initialize.assert_awaited_once()
                assert ctx.registry is mock_registry

    @pytest.mark.asyncio
    async def test_lifespan_creates_session_manager(self):
        """Test that lifespan creates session manager with settings."""
        mock_server = MagicMock()

        with patch("reasoning_mcp.server.Settings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.max_sessions = 100
            mock_settings.session_cleanup_interval = 300
            mock_settings_class.return_value = mock_settings

            with patch("reasoning_mcp.sessions.SessionManager") as mock_sm_class:
                mock_sm = MagicMock()
                mock_sm.clear = AsyncMock()  # Need AsyncMock for cleanup
                mock_sm_class.return_value = mock_sm

                async with app_lifespan(mock_server) as ctx:
                    # Verify SessionManager was created with correct args
                    mock_sm_class.assert_called_once_with(
                        max_sessions=100,
                        cleanup_interval=300,
                    )
                    assert ctx.session_manager is mock_sm

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_clears_sessions(self):
        """Test that lifespan cleanup clears sessions."""
        mock_server = MagicMock()

        with patch("reasoning_mcp.sessions.SessionManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm.clear = AsyncMock()
            mock_sm_class.return_value = mock_sm

            async with app_lifespan(mock_server):
                pass  # Just enter and exit the context

            # Verify cleanup was called
            mock_sm.clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_marks_uninitialized(self):
        """Test that lifespan cleanup marks context as uninitialized."""
        mock_server = MagicMock()

        # We need to capture the context to check it after cleanup
        captured_ctx = None

        with patch("reasoning_mcp.sessions.SessionManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm.clear = AsyncMock()
            mock_sm_class.return_value = mock_sm

            async with app_lifespan(mock_server) as ctx:
                captured_ctx = ctx
                assert ctx.initialized is True

            # After exiting, should be marked as uninitialized
            assert captured_ctx.initialized is False

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_on_exception(self):
        """Test that lifespan cleanup happens even on exception."""
        mock_server = MagicMock()

        with patch("reasoning_mcp.sessions.SessionManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm.clear = AsyncMock()
            mock_sm_class.return_value = mock_sm

            try:
                async with app_lifespan(mock_server):
                    raise RuntimeError("Test exception")
            except RuntimeError:
                pass  # Expected

            # Verify cleanup still happened
            mock_sm.clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_sets_initialized_before_yielding(self):
        """Test that context is marked initialized before yielding."""
        mock_server = MagicMock()

        async with app_lifespan(mock_server) as ctx:
            # Should be initialized when we receive the context
            assert ctx.initialized is True


class TestGetSamplingHandler:
    """Tests for get_sampling_handler function."""

    def test_get_sampling_handler_exists(self):
        """Test that get_sampling_handler function exists."""
        assert callable(get_sampling_handler)

    def test_get_sampling_handler_returns_config(self):
        """Test that get_sampling_handler returns sampling configuration."""
        result = get_sampling_handler()
        assert result is not None
        assert isinstance(result, dict)
        assert "enabled" in result
        assert "provider" in result

    def test_get_sampling_handler_no_parameters(self):
        """Test that get_sampling_handler takes no parameters."""
        # This should work without arguments
        get_sampling_handler()


class TestGetAuthProvider:
    """Tests for get_auth_provider function."""

    def test_get_auth_provider_exists(self):
        """Test that get_auth_provider function exists."""
        assert callable(get_auth_provider)

    def test_get_auth_provider_returns_none(self):
        """Test that get_auth_provider returns None."""
        result = get_auth_provider()
        assert result is None

    def test_get_auth_provider_no_parameters(self):
        """Test that get_auth_provider takes no parameters."""
        # This should work without arguments
        get_auth_provider()


class TestGetAppContext:
    """Tests for get_app_context function."""

    def test_get_app_context_raises_when_not_initialized(self):
        """Test that get_app_context raises error when not in lifespan."""
        from reasoning_mcp.server import (
            AppContextNotInitializedError,
            get_app_context,
        )

        with pytest.raises(AppContextNotInitializedError):
            get_app_context()

    def test_app_context_not_initialized_error_message(self):
        """Test that AppContextNotInitializedError has informative message."""
        from reasoning_mcp.server import AppContextNotInitializedError

        error = AppContextNotInitializedError()
        assert "AppContext not initialized" in str(error)
        assert "lifespan" in str(error)

    def test_app_context_not_initialized_error_is_runtime_error(self):
        """Test that AppContextNotInitializedError is a RuntimeError."""
        from reasoning_mcp.server import AppContextNotInitializedError

        assert issubclass(AppContextNotInitializedError, RuntimeError)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_module_exports_mcp(self):
        """Test that module exports mcp."""
        from reasoning_mcp import server

        assert "mcp" in server.__all__
        assert hasattr(server, "mcp")

    def test_module_exports_appcontext(self):
        """Test that module exports AppContext."""
        from reasoning_mcp import server

        assert "AppContext" in server.__all__
        assert hasattr(server, "AppContext")

    def test_module_exports_app_lifespan(self):
        """Test that module exports app_lifespan."""
        from reasoning_mcp import server

        assert "app_lifespan" in server.__all__
        assert hasattr(server, "app_lifespan")

    def test_module_exports_get_sampling_handler(self):
        """Test that module exports get_sampling_handler."""
        from reasoning_mcp import server

        assert "get_sampling_handler" in server.__all__
        assert hasattr(server, "get_sampling_handler")

    def test_module_exports_get_auth_provider(self):
        """Test that module exports get_auth_provider."""
        from reasoning_mcp import server

        assert "get_auth_provider" in server.__all__
        assert hasattr(server, "get_auth_provider")

    def test_module_exports_get_app_context(self):
        """Test that module exports get_app_context."""
        from reasoning_mcp import server

        assert "get_app_context" in server.__all__
        assert hasattr(server, "get_app_context")

    def test_module_exports_app_context_not_initialized_error(self):
        """Test that module exports AppContextNotInitializedError."""
        from reasoning_mcp import server

        assert "AppContextNotInitializedError" in server.__all__
        assert hasattr(server, "AppContextNotInitializedError")

    def test_module_exports_count(self):
        """Test that module exports exactly 7 items."""
        from reasoning_mcp import server

        assert len(server.__all__) == 7

    def test_all_exports_are_valid(self):
        """Test that all items in __all__ actually exist in module."""
        from reasoning_mcp import server

        for name in server.__all__:
            assert hasattr(server, name), f"{name} is in __all__ but not in module"
