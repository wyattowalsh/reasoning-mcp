"""Unit tests for health check tools."""

from __future__ import annotations

import pytest

from reasoning_mcp.tools.health import (
    HealthCheckOutput,
    LivenessCheckOutput,
    ReadinessCheckOutput,
)


class TestHealthCheckOutputModel:
    """Tests for HealthCheckOutput model."""

    def test_create_healthy_output(self):
        """Test creating a healthy output."""
        output = HealthCheckOutput(
            status="healthy",
            version="0.1.0",
            components=[
                {
                    "name": "registry",
                    "status": "healthy",
                    "message": "10 methods registered",
                    "latency_ms": 1.5,
                }
            ],
            startup_errors=[],
        )

        assert output.status == "healthy"
        assert output.version == "0.1.0"
        assert len(output.components) == 1
        assert output.components[0]["name"] == "registry"
        assert output.startup_errors == []

    def test_create_degraded_output(self):
        """Test creating a degraded output with startup errors."""
        output = HealthCheckOutput(
            status="degraded",
            version="0.1.0",
            components=[],
            startup_errors=["Telemetry failed to initialize"],
        )

        assert output.status == "degraded"
        assert len(output.startup_errors) == 1


class TestReadinessCheckOutputModel:
    """Tests for ReadinessCheckOutput model."""

    def test_create_ready_output(self):
        """Test creating a ready output."""
        output = ReadinessCheckOutput(
            ready=True,
            reason=None,
            components_ready={
                "context": True,
                "registry": True,
                "session_manager": True,
            },
        )

        assert output.ready is True
        assert output.reason is None
        assert all(output.components_ready.values())

    def test_create_not_ready_output(self):
        """Test creating a not ready output."""
        output = ReadinessCheckOutput(
            ready=False,
            reason="Registry error: no methods registered",
            components_ready={
                "context": True,
                "registry": False,
                "session_manager": True,
            },
        )

        assert output.ready is False
        assert "Registry error" in output.reason
        assert output.components_ready["registry"] is False


class TestLivenessCheckOutputModel:
    """Tests for LivenessCheckOutput model."""

    def test_create_alive_output(self):
        """Test creating an alive output."""
        output = LivenessCheckOutput(
            alive=True,
            uptime_seconds=None,
        )

        assert output.alive is True

    def test_create_alive_with_uptime(self):
        """Test creating an alive output with uptime."""
        output = LivenessCheckOutput(
            alive=True,
            uptime_seconds=3600.5,
        )

        assert output.alive is True
        assert output.uptime_seconds == 3600.5


class TestHealthCheckHandlers:
    """Tests for health check handler functions."""

    @pytest.fixture
    def mock_app_context(self):
        """Create a mock app context for testing."""
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        ctx.initialized = True
        ctx.startup_errors = []

        # Mock registry
        ctx.registry = MagicMock()
        ctx.registry.list_all.return_value = ["method1", "method2"]

        # Mock session manager
        ctx.session_manager = MagicMock()
        ctx.session_manager.list_sessions = AsyncMock(return_value=[])

        # Mock hybrid session manager (not configured)
        ctx.hybrid_session_manager = None

        # Mock middleware (not configured)
        ctx.middleware = None

        # Mock cache (not configured)
        ctx.cache = None

        return ctx

    @pytest.mark.asyncio
    async def test_health_check_handler(self, mock_app_context):
        """Test health_check_handler returns proper output."""
        from reasoning_mcp.tools.health import health_check_handler

        result = await health_check_handler(mock_app_context)

        assert isinstance(result, HealthCheckOutput)
        assert result.status in ("healthy", "degraded", "unhealthy")
        assert result.version == "0.1.0"

    @pytest.mark.asyncio
    async def test_readiness_check_handler(self, mock_app_context):
        """Test readiness_check_handler returns proper output."""
        from reasoning_mcp.tools.health import readiness_check_handler

        result = await readiness_check_handler(mock_app_context)

        assert isinstance(result, ReadinessCheckOutput)
        assert result.ready is True
        assert result.components_ready["context"] is True

    @pytest.mark.asyncio
    async def test_readiness_not_initialized(self, mock_app_context):
        """Test readiness when context not initialized."""
        from reasoning_mcp.tools.health import readiness_check_handler

        mock_app_context.initialized = False

        result = await readiness_check_handler(mock_app_context)

        assert result.ready is False
        assert "not initialized" in result.reason

    @pytest.mark.asyncio
    async def test_liveness_check_handler(self, mock_app_context):
        """Test liveness_check_handler returns proper output."""
        from reasoning_mcp.tools.health import liveness_check_handler

        result = await liveness_check_handler(mock_app_context)

        assert isinstance(result, LivenessCheckOutput)
        assert result.alive is True


class TestHealthCheckWithComponents:
    """Tests for health check with various component configurations."""

    @pytest.fixture
    def full_mock_context(self):
        """Create a mock context with all components."""
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        ctx.initialized = True
        ctx.startup_errors = []

        # Mock registry
        ctx.registry = MagicMock()
        ctx.registry.list_all.return_value = ["method1", "method2", "method3"]

        # Mock session manager
        ctx.session_manager = MagicMock()
        ctx.session_manager.list_sessions = AsyncMock(return_value=["session1"])

        # Mock hybrid session manager
        ctx.hybrid_session_manager = MagicMock()
        ctx.hybrid_session_manager.list_sessions = AsyncMock(return_value=[])

        # Mock middleware with metrics
        ctx.middleware = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.tool_calls = 100
        mock_metrics.errors = 2
        ctx.middleware.get_metrics.return_value = mock_metrics

        # Mock cache with metrics
        ctx.cache = MagicMock()
        cache_metrics = MagicMock()
        cache_metrics.hit_rate = 0.85
        ctx.cache.get_metrics.return_value = cache_metrics

        return ctx

    @pytest.mark.asyncio
    async def test_health_with_all_components(self, full_mock_context):
        """Test health check reports all components."""
        from reasoning_mcp.tools.health import health_check_handler

        result = await health_check_handler(full_mock_context)

        assert result.status == "healthy"
        component_names = [c["name"] for c in result.components]
        assert "registry" in component_names
        assert "session_manager" in component_names
        assert "persistence" in component_names
        assert "middleware" in component_names
        assert "cache" in component_names

    @pytest.mark.asyncio
    async def test_health_reports_startup_errors(self, full_mock_context):
        """Test health check includes startup errors."""
        from reasoning_mcp.tools.health import health_check_handler

        full_mock_context.startup_errors = ["Telemetry init failed"]

        result = await health_check_handler(full_mock_context)

        assert len(result.startup_errors) == 1
        assert "Telemetry" in result.startup_errors[0]
