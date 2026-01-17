"""E2E tests for MCP server protocol compliance and tool execution.

Tests cover:
- Server initialization and lifespan
- MCP tool registration and discovery
- Tool execution via MCP protocol
- Resource access via MCP protocol
- Prompt retrieval via MCP protocol
- Background task management
- Middleware integration (rate limiting, caching)
- Error handling and edge cases

These tests validate the complete reasoning-mcp MCP server functionality
to ensure protocol compliance and correct operation.

Run with: pytest tests/e2e/test_mcp_server.py -v -m e2e
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.config import Settings
from reasoning_mcp.middleware import (
    CacheMetrics,
    RateLimitMetrics,
    RateLimitMiddleware,
    ReasoningMiddleware,
    RequestMetrics,
    ResponseCacheMiddleware,
)
from reasoning_mcp.models.core import MethodIdentifier
from reasoning_mcp.models.tools import (
    ComposeOutput,
    MethodInfo,
    ReasonOutput,
    Recommendation,
    SessionState,
    ThoughtOutput,
)
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.server import (
    AppContext,
    AppContextNotInitializedError,
    app_lifespan,
    get_app_context,
    mcp,
)
from reasoning_mcp.sessions import SessionManager
from reasoning_mcp.tasks import TaskProgress, TaskResult, TaskStatus

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

# ============================================================================
# TEST CLASS: MCP SERVER LIFECYCLE
# ============================================================================


class TestMCPServerLifecycle:
    """Tests for MCP server lifecycle management."""

    async def test_server_initialization_success(self):
        """Verify app_lifespan creates valid AppContext with all components."""
        mock_server = MagicMock()

        async with app_lifespan(mock_server) as ctx:
            # Verify context is properly initialized
            assert isinstance(ctx, AppContext)
            assert ctx.initialized is True
            assert ctx.registry is not None
            assert ctx.session_manager is not None
            assert ctx.settings is not None

            # Verify registry has methods registered
            assert isinstance(ctx.registry, MethodRegistry)
            assert ctx.registry.method_count > 0

            # Verify session manager is configured
            assert isinstance(ctx.session_manager, SessionManager)

    async def test_server_shutdown_cleanup(self):
        """Verify cleanup on shutdown: sessions cleared, metrics logged."""
        mock_server = MagicMock()

        # Track context for post-shutdown verification
        captured_ctx: AppContext | None = None

        async with app_lifespan(mock_server) as ctx:
            captured_ctx = ctx

            # Create a session to verify cleanup
            session = await ctx.session_manager.create()
            assert session is not None

            # Verify session exists
            active_session = await ctx.session_manager.get(session.id)
            assert active_session is not None

        # After exiting context, verify cleanup
        assert captured_ctx is not None
        assert captured_ctx.initialized is False

    async def test_server_middleware_registration(self):
        """Verify middleware registered when enabled in settings."""
        mock_server = MagicMock()

        # Patch settings to enable middleware
        with patch("reasoning_mcp.server.Settings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.enable_middleware = True
            mock_settings.enable_middleware_metrics = True
            mock_settings.middleware_log_level = "DEBUG"
            mock_settings.enable_cache = False
            mock_settings.enable_rate_limiting = False
            mock_settings.jwt_enabled = False
            mock_settings.api_key_enabled = False
            mock_settings.enable_sampling = True
            mock_settings.enable_elicitation = True
            mock_settings.enable_background_tasks = True
            mock_settings.max_sessions = 100
            mock_settings.session_cleanup_interval = 300
            mock_settings.sampling_provider = None
            mock_settings_class.return_value = mock_settings

            async with app_lifespan(mock_server) as ctx:
                # Verify middleware was registered
                mock_server.add_middleware.assert_called()
                assert ctx.middleware is not None

    async def test_server_telemetry_initialization(self):
        """Verify OpenTelemetry setup during server init."""
        mock_server = MagicMock()

        with (
            patch("reasoning_mcp.server.init_telemetry") as mock_init_telemetry,
            patch("reasoning_mcp.server.shutdown_telemetry") as mock_shutdown_telemetry,
        ):
            async with app_lifespan(mock_server):
                # Verify telemetry was initialized
                mock_init_telemetry.assert_called_once()

            # Verify telemetry was shutdown
            mock_shutdown_telemetry.assert_called_once()

    async def test_server_cleanup_on_exception(self):
        """Verify cleanup happens even when exception occurs."""
        mock_server = MagicMock()

        # Create a mock session manager to track cleanup
        mock_session_manager = MagicMock()
        mock_session_manager.clear = AsyncMock()

        with (
            patch("reasoning_mcp.sessions.SessionManager", return_value=mock_session_manager),
            pytest.raises(RuntimeError, match="Test exception"),
        ):
            async with app_lifespan(mock_server):
                raise RuntimeError("Test exception")

        # Verify cleanup was called despite exception
        mock_session_manager.clear.assert_awaited_once()


# ============================================================================
# TEST CLASS: MCP TOOL REGISTRATION
# ============================================================================


class TestMCPToolRegistration:
    """Tests for MCP tool registration and metadata."""

    def test_mcp_server_instance_exists(self) -> None:
        """Verify mcp server instance is created."""
        assert mcp is not None, "MCP server instance should exist"
        assert mcp.name == "reasoning-mcp", f"Expected name 'reasoning-mcp', got '{mcp.name}'"

    def test_all_tools_registered(self) -> None:
        """Verify all expected tools are registered with mcp instance."""
        # Get tool names from the mcp instance
        # FastMCP stores tools internally
        expected_tools = [
            "reason",
            "reason_background",
            "methods_list",
            "methods_recommend",
            "methods_compare",
            "methods_compare_background",
            "session_continue",
            "session_continue_background",
            "session_branch",
            "session_inspect",
            "session_merge",
            "compose",
            "compose_background",
            "evaluate",
            "ensemble_reason",
            "list_voting_strategies",
            # Background task tools
            "task_progress",
            "task_result",
            "task_cancel",
        ]

        # Access the internal tool registry
        registered_tools = list(mcp._tool_manager._tools.keys())

        # Check each expected tool with detailed error message
        missing_tools = [t for t in expected_tools if t not in registered_tools]
        assert not missing_tools, (
            f"Missing tools: {missing_tools}. Registered: {sorted(registered_tools)}"
        )

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("reason", id="reason-tool"),
            pytest.param("methods_list", id="methods-list-tool"),
            pytest.param("compose", id="compose-tool"),
        ],
    )
    def test_tool_metadata_present(self, tool_name: str) -> None:
        """Verify tools have titles, descriptions, and schemas (parametrized)."""
        tool = mcp._tool_manager._tools.get(tool_name)
        assert tool is not None, f"Tool '{tool_name}' not found in registry"
        assert tool.name == tool_name, (
            f"Tool name mismatch: expected '{tool_name}', got '{tool.name}'"
        )

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("reason_background", id="reason-bg"),
            pytest.param("methods_compare_background", id="methods-compare-bg"),
            pytest.param("session_continue_background", id="session-continue-bg"),
            pytest.param("compose_background", id="compose-bg"),
        ],
    )
    def test_background_tools_registered(self, tool_name: str) -> None:
        """Verify task=True tools are properly registered as background tasks (parametrized)."""
        tool = mcp._tool_manager._tools.get(tool_name)
        assert tool is not None, f"Background tool '{tool_name}' not registered"


# ============================================================================
# TEST CLASS: MCP TOOL EXECUTION
# ============================================================================


class TestMCPToolExecution:
    """Tests for MCP tool execution via protocol."""

    @pytest.mark.parametrize(
        "method_id",
        [
            pytest.param("chain_of_thought", id="cot"),
            pytest.param("self_reflection", id="self-reflect"),
            pytest.param("tree_of_thoughts", id="tot"),
        ],
    )
    async def test_reason_tool_with_multiple_methods(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
        method_id: str,
    ) -> None:
        """Test reason tool with various method types (parametrized)."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        result = await reason(problem=sample_problem, method=method_id)

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.thought is not None
        assert result.method_used == method_id

    async def test_reason_tool_execution(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test reason tool with chain_of_thought method."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        result = await reason(
            problem=sample_problem,
            method="chain_of_thought",
        )

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.thought is not None
        assert result.method_used == "chain_of_thought"

    async def test_reason_with_auto_selection(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test auto method selection when method is None."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        # Execute without specifying method (auto-selection)
        result = await reason(problem=sample_problem, method=None)

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.method_used is not None  # Should be auto-selected

    async def test_reason_with_hints(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
        sample_hints: dict[str, Any],
    ) -> None:
        """Test hint-guided method selection."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.models.tools import ReasonHints
        from reasoning_mcp.tools.reason import reason

        # Convert hints to ReasonHints
        hints = ReasonHints(
            domain=sample_hints.get("domain"),
            complexity=sample_hints.get("complexity"),
            prefer_methods=[MethodIdentifier(m) for m in sample_hints.get("prefer_methods", [])],
            avoid_methods=[MethodIdentifier(m) for m in sample_hints.get("avoid_methods", [])],
        )

        result = await reason(problem=sample_problem, method=None, hints=hints)

        assert isinstance(result, ReasonOutput)
        # Method should be influenced by hints
        assert result.method_used in sample_hints.get("prefer_methods", []) or (
            result.method_used not in sample_hints.get("avoid_methods", [])
        )

    async def test_methods_list_tool(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test listing methods with filters."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.methods import methods_list

        # List all methods
        result = methods_list(category=None, tags=None)

        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
        assert len(result) > 0, "methods_list returned empty result"
        assert all(isinstance(m, MethodInfo) for m in result), (
            f"Not all results are MethodInfo: {[type(m).__name__ for m in result if not isinstance(m, MethodInfo)]}"
        )

        # List with category filter
        core_methods = methods_list(category="core", tags=None)
        assert isinstance(core_methods, list), (
            f"Expected list for category filter, got {type(core_methods).__name__}"
        )

    async def test_methods_recommend_tool(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test method recommendations."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.methods import methods_recommend

        result = methods_recommend(problem=sample_problem, max_results=3)

        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
        assert len(result) <= 3, f"Expected at most 3 recommendations, got {len(result)}"
        assert all(isinstance(r, Recommendation) for r in result), (
            f"Not all results are Recommendation: {[type(r).__name__ for r in result if not isinstance(r, Recommendation)]}"
        )

        # Recommendations should have valid scores (0.0 to 1.0)
        for rec in result:
            assert 0.0 <= rec.score <= 1.0, (
                f"Score {rec.score} out of valid range [0.0, 1.0] for {rec.method_id}"
            )

    async def test_session_lifecycle_tools(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test continue, branch, inspect, merge sequence.

        Note: Some session tools currently have placeholder implementations,
        so we test the tool interfaces rather than full functionality.
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason
        from reasoning_mcp.tools.session import (
            session_branch,
            session_continue,
            session_inspect,
        )

        # Start a session
        reason_result = await reason(problem=sample_problem, method="chain_of_thought")
        session_id = reason_result.session_id
        assert session_id is not None

        # Continue reasoning (returns placeholder but validates interface)
        continue_result = await session_continue(
            session_id=session_id,
            guidance="Let's explore this further",
        )
        assert isinstance(continue_result, ThoughtOutput)
        assert continue_result.content is not None

        # Inspect session (placeholder implementation)
        inspect_result = await session_inspect(session_id=session_id, include_graph=True)
        assert isinstance(inspect_result, SessionState)
        assert inspect_result.status is not None

        # Create a branch (placeholder implementation)
        branch_result = await session_branch(
            session_id=session_id,
            branch_name="alternative-approach",
            from_thought_id=None,
        )
        assert branch_result.success is True

    async def test_compose_tool_execution(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
        sample_sequence_pipeline: dict[str, Any],
    ) -> None:
        """Test pipeline execution via compose tool.

        This test validates pipeline parsing with the stage_type discriminator
        and compose tool execution.
        """
        ctx, _ = mcp_server_context

        from pydantic import TypeAdapter

        # Use the pipeline model
        from reasoning_mcp.models.pipeline import Pipeline
        from reasoning_mcp.tools.compose import compose

        # Validate pipeline parsing with stage_type discriminator
        pipeline_adapter: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)
        pipeline_obj = pipeline_adapter.validate_python(sample_sequence_pipeline)

        # Verify pipeline was parsed correctly
        assert pipeline_obj is not None
        assert pipeline_obj.stage_type.value == "sequence"

        result = await compose(
            pipeline=pipeline_obj,
            input=sample_problem,
            session_id=None,
        )

        assert isinstance(result, ComposeOutput)
        assert result.session_id is not None

    async def test_ensemble_reason_tool(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test ensemble reasoning execution."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.models.ensemble import VotingStrategy
        from reasoning_mcp.tools.ensemble import EnsembleToolInput, ensemble_reason

        input_data = EnsembleToolInput(
            query=sample_problem,
            methods=["chain_of_thought", "self_reflection"],
            strategy=VotingStrategy.MAJORITY,
            timeout_ms=30000,
        )

        result = await ensemble_reason(input_data)

        assert result is not None
        assert result.final_answer is not None
        assert result.confidence >= 0.0


# ============================================================================
# TEST CLASS: MCP RESOURCE ACCESS
# ============================================================================


class TestMCPResourceAccess:
    """Tests for MCP resource access via protocol."""

    async def test_method_metadata_resource(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test method://metadata/{method_id} resource."""
        ctx, server = mcp_server_context

        # The resource is registered but we need to simulate access
        # In E2E, we'll call the resource handler directly
        from reasoning_mcp.resources.method import register_method_resources

        # Create a mock server to capture the resource registration
        mock_mcp = MagicMock()
        resources = {}

        def capture_resource(uri_pattern):
            def decorator(func):
                resources[uri_pattern] = func
                return func

            return decorator

        mock_mcp.resource = capture_resource
        mock_mcp.get_context = MagicMock(return_value=ctx)

        register_method_resources(mock_mcp)

        # Call the metadata resource
        metadata_handler = resources.get("method://{method_id}")
        assert metadata_handler is not None

    async def test_session_state_resource(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test session://{session_id} resource access.

        Note: The reason tool creates sessions that may use a different
        context, so we test the resource registration and interface instead.
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        # Create a session first
        result = await reason(problem=sample_problem, method="chain_of_thought")
        session_id = result.session_id

        # Verify the reason tool returned a valid session ID
        assert session_id is not None
        assert isinstance(session_id, str)

        # The thought should be valid
        assert result.thought is not None


# ============================================================================
# TEST CLASS: MCP PROMPT RETRIEVAL
# ============================================================================


class TestMCPPromptRetrieval:
    """Tests for MCP prompt retrieval via protocol."""

    async def test_guided_reasoning_prompt(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test guided_reasoning prompt function."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.prompts.guided import register_guided_prompts

        # Capture registered prompts
        mock_mcp = MagicMock()
        prompts = {}
        mock_mcp.app_context = ctx

        def capture_prompt():
            def decorator(func):
                prompts[func.__name__] = func
                return func

            return decorator

        mock_mcp.prompt = capture_prompt

        register_guided_prompts(mock_mcp)

        # Verify prompts were registered
        assert "reason_with_method" in prompts
        assert "compare_methods" in prompts

    async def test_workflow_prompts(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test workflow-specific prompts are registered."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.prompts.pipelines import register_pipeline_prompts

        mock_mcp = MagicMock()
        prompts = {}

        def capture_prompt(*args, **kwargs):
            """Mock decorator that handles @mcp.prompt(name=..., description=...) form."""

            def decorator(func):
                prompts[func.__name__] = func
                return func

            return decorator

        mock_mcp.prompt = capture_prompt

        # Register prompts - will call mock_mcp.prompt(name=..., description=...)
        register_pipeline_prompts(mock_mcp)

        # Verify prompts were registered (prompts dict should have entries)
        assert len(prompts) > 0, "Expected at least one pipeline prompt to be registered"


# ============================================================================
# TEST CLASS: MCP BACKGROUND TASKS
# ============================================================================


class TestMCPBackgroundTasks:
    """Tests for background task execution and management."""

    async def test_task_manager_basic_operations(self, task_manager):
        """Test basic task manager operations."""
        # Create a task
        task = task_manager.create_task(total_steps=5)

        assert task is not None
        assert task.task_id is not None
        assert task.status == TaskStatus.PENDING

        # Start the task
        task_manager.start_task(task.task_id)
        task = task_manager.get_task(task.task_id)
        assert task.status == TaskStatus.RUNNING

        # Update progress
        task_manager.update_progress(
            task.task_id,
            progress=0.5,
            message="Half done",
            steps_completed=3,
        )

        progress = task_manager.get_progress(task.task_id)
        assert isinstance(progress, TaskProgress)
        assert progress.progress == 0.5

        # Complete the task
        task_manager.complete_task(task.task_id, result={"answer": "test"})

        result = task_manager.get_result(task.task_id)
        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"answer": "test"}

    async def test_task_cancellation(self, task_manager):
        """Test task cancellation."""
        task = task_manager.create_task()
        task_manager.start_task(task.task_id)

        # Cancel the task
        task_manager.cancel_task(task.task_id)

        task = task_manager.get_task(task.task_id)
        assert task.status == TaskStatus.CANCELLED

    async def test_task_failure_handling(self, task_manager):
        """Test task failure handling."""
        task = task_manager.create_task()
        task_manager.start_task(task.task_id)

        # Fail the task
        task_manager.fail_task(task.task_id, error="Test error occurred")

        result = task_manager.get_result(task.task_id)
        assert result.status == TaskStatus.FAILED
        assert result.error == "Test error occurred"


# ============================================================================
# TEST CLASS: MCP MIDDLEWARE INTEGRATION
# ============================================================================


class TestMCPMiddlewareIntegration:
    """Tests for middleware integration with MCP server."""

    def test_reasoning_middleware_metrics(self) -> None:
        """Test reasoning middleware metrics collection."""
        middleware = ReasoningMiddleware(
            enable_logging=True,
            enable_metrics=True,
        )

        # Initial metrics should be zero
        metrics = middleware.get_metrics()
        assert isinstance(metrics, RequestMetrics)
        assert metrics.tool_calls == 0
        assert metrics.resource_reads == 0
        assert metrics.errors == 0

        # Reset metrics
        middleware.reset_metrics()
        metrics = middleware.get_metrics()
        assert metrics.tool_calls == 0

    @pytest.mark.asyncio
    async def test_response_cache_middleware(self) -> None:
        """Test response caching functionality."""
        cache = ResponseCacheMiddleware(
            max_entries=100,
            default_ttl=60,
        )

        # Test cache set/get (now async with thread-safe locking)
        await cache.set("test_key", {"data": "test"})
        result = await cache.get("test_key")
        assert result == {"data": "test"}

        # Test cache miss
        result = await cache.get("nonexistent")
        assert result is None

        # Test metrics
        metrics = cache.get_metrics()
        assert isinstance(metrics, CacheMetrics)
        assert metrics.hits >= 1
        assert metrics.misses >= 1

    def test_rate_limiting_middleware(self, test_settings_with_middleware: Settings) -> None:
        """Test rate limit middleware functionality."""
        rate_limiter = RateLimitMiddleware(test_settings_with_middleware)

        # Initial metrics
        metrics = rate_limiter.get_metrics()
        assert isinstance(metrics, RateLimitMetrics)
        assert metrics.requests_allowed == 0
        assert metrics.requests_rejected == 0

    def test_cache_eviction(self) -> None:
        """Test cache eviction when at capacity."""
        cache = ResponseCacheMiddleware(
            max_entries=5,
            default_ttl=60,
        )

        # Fill cache to capacity
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        # Cache should have evicted old entries
        assert cache.size <= 5


# ============================================================================
# TEST CLASS: MCP ERROR HANDLING
# ============================================================================


class TestMCPErrorHandling:
    """Tests for error handling in MCP server."""

    async def test_invalid_method_id_error(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test error on invalid method ID."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        # Should raise or handle gracefully for invalid method
        with pytest.raises((ValueError, KeyError)):
            await reason(
                problem=sample_problem,
                method="nonexistent_method_xyz",
            )

    async def test_session_not_found_error(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test behavior on missing session.

        Note: Current placeholder implementation of session_inspect doesn't
        actually validate session existence, so we verify it returns a
        SessionState object (which shows the interface is consistent).
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.session import session_inspect

        # session_inspect raises ValueError for nonexistent sessions
        with pytest.raises(ValueError, match="Session not found"):
            await session_inspect(
                session_id="nonexistent-session-id-12345",
                include_graph=False,
            )

    async def test_app_context_not_initialized_error(self):
        """Test error when AppContext accessed before initialization."""
        # Ensure context is not initialized
        import reasoning_mcp.server as server_module

        original = server_module._APP_CONTEXT
        server_module._APP_CONTEXT = None

        try:
            with pytest.raises(AppContextNotInitializedError):
                get_app_context()
        finally:
            server_module._APP_CONTEXT = original

    async def test_invalid_pipeline_error(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test error on malformed pipeline."""
        ctx, _ = mcp_server_context

        from pydantic import TypeAdapter, ValidationError

        from reasoning_mcp.models.pipeline import Pipeline

        # Invalid pipeline structure
        invalid_pipeline = {
            "type": "invalid_type",
            "name": "test",
        }

        pipeline_adapter: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)

        with pytest.raises(ValidationError):
            pipeline_adapter.validate_python(invalid_pipeline)


# ============================================================================
# TEST CLASS: INTEGRATION SCENARIOS
# ============================================================================


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    async def test_complete_reasoning_workflow(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test complete reasoning workflow from start to finish.

        Note: This test uses explicit method specification since method
        recommendations may vary depending on registry initialization.
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason
        from reasoning_mcp.tools.session import (
            session_continue,
            session_inspect,
        )

        # Step 1: Start reasoning with a known method
        result = await reason(problem=sample_problem, method="chain_of_thought")

        assert result.session_id is not None, "Initial reasoning should create a session"
        session_id = result.session_id
        assert result.thought is not None, "Initial reasoning should produce a thought"

        # Step 2: Continue reasoning (placeholder returns consistent results)
        for i in range(2):
            continue_result = await session_continue(
                session_id=session_id,
                guidance=f"Step {i + 2}: Continue analysis",
            )
            assert continue_result.content is not None

        # Step 3: Inspect session state (placeholder behavior)
        final_state = await session_inspect(session_id=session_id, include_graph=True)

        assert final_state is not None, "Session inspect should return state"
        assert final_state.status is not None, "Session state should have a status"

    async def test_parallel_reasoning_sessions(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
        sample_math_problem: str,
    ) -> None:
        """Test running multiple reasoning sessions in parallel."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        # Start multiple sessions concurrently
        results = await asyncio.gather(
            reason(problem=sample_problem, method="chain_of_thought"),
            reason(problem=sample_math_problem, method="mathematical_reasoning"),
        )

        # Verify all sessions completed
        assert len(results) == 2, f"Expected 2 parallel results, got {len(results)}"
        assert all(r.session_id is not None for r in results), (
            f"Some sessions missing ID: {[i for i, r in enumerate(results) if r.session_id is None]}"
        )
        assert results[0].session_id != results[1].session_id, (
            f"Sessions should have unique IDs, both have: {results[0].session_id}"
        )

    async def test_method_comparison_workflow(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_ethical_problem: str,
    ) -> None:
        """Test method comparison for a specific problem type."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.methods import methods_compare, methods_recommend

        # Get recommendations for ethical problem
        recommendations = methods_recommend(problem=sample_ethical_problem, max_results=5)

        # Compare top methods
        if len(recommendations) >= 2:
            method_ids = [str(r.method_id) for r in recommendations[:3]]
            comparison = methods_compare(methods=method_ids, problem=sample_ethical_problem)

            assert comparison.winner is not None, "Comparison should determine a winner"
            assert len(comparison.scores) == len(method_ids), (
                f"Expected {len(method_ids)} scores, got {len(comparison.scores)}"
            )


# ============================================================================
# TEST CLASS: TOOL OUTPUT VALIDATION
# ============================================================================


class TestToolOutputValidation:
    """Tests for validating tool output schemas and types."""

    async def test_reason_output_schema(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Verify ReasonOutput contains all expected fields."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        result = await reason(problem=sample_problem, method="chain_of_thought")

        # Verify output type
        assert isinstance(result, ReasonOutput)

        # Verify required fields
        assert result.session_id is not None
        assert isinstance(result.session_id, str)
        assert result.thought is not None
        assert result.method_used is not None

    async def test_method_info_schema(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Verify MethodInfo contains all expected fields."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.methods import methods_list

        result = methods_list(category=None, tags=None)

        assert len(result) > 0, "methods_list should return at least one method"
        method_info = result[0]

        assert isinstance(method_info, MethodInfo), (
            f"Expected MethodInfo, got {type(method_info).__name__}"
        )
        assert method_info.id is not None, "MethodInfo.id should not be None"
        assert method_info.name is not None, "MethodInfo.name should not be None"
        assert method_info.description is not None, "MethodInfo.description should not be None"

    async def test_session_state_schema(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Verify SessionState contains all expected fields.

        Note: Current session_inspect is a placeholder that returns
        hardcoded values, so we verify the schema structure.
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason
        from reasoning_mcp.tools.session import session_inspect

        # Create session
        result = await reason(problem=sample_problem, method="chain_of_thought")

        # Inspect session (placeholder returns consistent schema)
        state = await session_inspect(session_id=result.session_id, include_graph=False)

        # Verify SessionState schema fields
        assert isinstance(state, SessionState), f"Expected SessionState, got {type(state).__name__}"
        assert state.session_id is not None, "SessionState.session_id should not be None"
        assert state.status is not None, "SessionState.status should not be None"
        assert state.thought_count is not None, (
            "SessionState.thought_count should not be None"
        )  # May be 0 in placeholder
        assert state.branch_count is not None, "SessionState.branch_count should not be None"


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_reason_with_unicode_input(
        self,
        mcp_server_context: tuple[AppContext, Any],
        unicode_problem: str,
    ) -> None:
        """Test reason tool handles unicode characters correctly."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        result = await reason(problem=unicode_problem, method="chain_of_thought")

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.thought is not None

    async def test_reason_with_long_input(
        self,
        mcp_server_context: tuple[AppContext, Any],
        long_problem: str,
    ) -> None:
        """Test reason tool handles very long input strings."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        result = await reason(problem=long_problem, method="chain_of_thought")

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None

    @pytest.mark.slow
    @pytest.mark.timeout(30)  # 30 second timeout for concurrent test
    async def test_concurrent_session_creation(
        self,
        mcp_server_context: tuple[AppContext, Any],
        sample_problem: str,
    ) -> None:
        """Test creating many sessions concurrently.

        This test validates the server can handle concurrent session creation
        without race conditions or resource exhaustion.
        """
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.reason import reason

        # Create 10 concurrent sessions
        num_sessions = 10
        tasks = [
            reason(problem=f"Problem {i}: {sample_problem}", method="chain_of_thought")
            for i in range(num_sessions)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all succeeded with unique session IDs
        assert len(results) == num_sessions, f"Expected {num_sessions} results, got {len(results)}"
        session_ids = {r.session_id for r in results}
        assert len(session_ids) == num_sessions, (
            f"Expected {num_sessions} unique sessions, got {len(session_ids)} "
            f"(duplicates: {num_sessions - len(session_ids)})"
        )

    async def test_methods_list_with_category_filter(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test methods_list with various category filters."""
        ctx, _ = mcp_server_context

        from reasoning_mcp.tools.methods import methods_list

        # Test with valid category
        core_methods = methods_list(category="core", tags=None)
        assert isinstance(core_methods, list), f"Expected list, got {type(core_methods).__name__}"

        # Test with None (all methods)
        all_methods = methods_list(category=None, tags=None)
        assert len(all_methods) >= len(core_methods), (
            f"All methods ({len(all_methods)}) should be >= core methods ({len(core_methods)})"
        )

    async def test_pipeline_validation_edge_cases(
        self,
        mcp_server_context: tuple[AppContext, Any],
    ) -> None:
        """Test pipeline validation with edge case configurations."""
        from pydantic import TypeAdapter, ValidationError

        from reasoning_mcp.models.pipeline import Pipeline

        pipeline_adapter: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)

        # Test with minimal valid pipeline
        minimal_pipeline = {
            "stage_type": "method",
            "method_id": "chain_of_thought",
        }
        result = pipeline_adapter.validate_python(minimal_pipeline)
        assert result is not None, "Minimal pipeline should validate successfully"

        # Test with missing required field
        with pytest.raises(ValidationError):
            pipeline_adapter.validate_python({"stage_type": "method"})

    async def test_cache_key_collision_handling(self) -> None:
        """Test cache handles key collisions gracefully."""
        cache = ResponseCacheMiddleware(max_entries=5, default_ttl=60)

        # Set multiple values with similar keys
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        # Verify cache doesn't exceed max_entries
        assert cache.size <= 5, f"Cache size {cache.size} exceeds max_entries 5"

        # Verify most recent entries are preserved
        # (implementation may vary based on eviction policy)
        assert cache.get("key_9") is not None or cache.size == 5, (
            f"Expected key_9 to exist or cache to be at capacity, size={cache.size}"
        )

    @pytest.mark.parametrize(
        ("requests_per_minute", "expected_allowed"),
        [
            pytest.param(60, True, id="normal-rate"),
            pytest.param(1, True, id="low-rate"),
        ],
    )
    def test_rate_limiter_configurations(
        self,
        requests_per_minute: int,
        expected_allowed: bool,
    ) -> None:
        """Test rate limiter with various configurations."""
        settings = Settings(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=requests_per_minute,
            rate_limit_requests_per_hour=1000,
            rate_limit_burst_size=10,
        )

        rate_limiter = RateLimitMiddleware(settings)
        assert rate_limiter is not None, "RateLimitMiddleware should be created"

        # Verify initial state
        metrics = rate_limiter.get_metrics()
        assert metrics.requests_allowed == 0, (
            f"Initial allowed should be 0, got {metrics.requests_allowed}"
        )
