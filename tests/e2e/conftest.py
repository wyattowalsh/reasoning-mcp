"""E2E test fixtures for MCP server testing.

This module provides comprehensive fixtures for end-to-end testing of the
reasoning-mcp MCP server, including:
- Server lifecycle management
- MCP context and session fixtures
- Mock LLM providers for deterministic testing
- Tool and resource testing utilities
- Reusable assertion helpers

Fixture Scopes:
- Session: Registry initialization (expensive, cached)
- Function: Server context, sessions (isolated per test)
- Module: Settings (shared within module)

Markers:
- @pytest.mark.e2e: All E2E tests
- @pytest.mark.slow: Tests taking >1s
- @pytest.mark.integration: Tests requiring real components
- @pytest.mark.timeout(seconds): Test timeout in seconds

Pytest-asyncio Configuration:
- Uses `asyncio_mode = "auto"` (no @pytest.mark.asyncio needed)
- Function-scoped event loops by default for test isolation
- Module-scoped fixtures use `scope="module"` with matching loop scope
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from reasoning_mcp.config import Settings
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.server import AppContext, app_lifespan, mcp
from reasoning_mcp.sessions import SessionManager
from reasoning_mcp.tasks import TaskManager

logger = logging.getLogger(__name__)

# Type variables for generic fixtures
T = TypeVar("T")


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for E2E tests."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow (>1s)")
    config.addinivalue_line("markers", "integration: mark test as requiring real components")
    config.addinivalue_line("markers", "timeout(seconds): set test timeout in seconds")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def assert_valid_session_id(session_id: str | None, context: str = "") -> None:
    """Assert that a session ID is valid (not None and non-empty).

    Args:
        session_id: The session ID to validate
        context: Optional context for error message

    Raises:
        AssertionError: If session_id is None or empty
    """
    ctx_msg = f" [{context}]" if context else ""
    assert session_id is not None, f"Session ID should not be None{ctx_msg}"
    assert isinstance(session_id, str), (
        f"Session ID should be string, got {type(session_id).__name__}{ctx_msg}"
    )
    assert len(session_id) > 0, f"Session ID should not be empty{ctx_msg}"


def assert_valid_thought(thought: Any, context: str = "") -> None:
    """Assert that a thought result is valid.

    Args:
        thought: The thought to validate
        context: Optional context for error message

    Raises:
        AssertionError: If thought is None or invalid
    """
    ctx_msg = f" [{context}]" if context else ""
    assert thought is not None, f"Thought should not be None{ctx_msg}"


def assert_list_of_type(
    result: Any,
    expected_type: type,
    min_length: int = 0,
    max_length: int | None = None,
    context: str = "",
) -> None:
    """Assert that result is a list of expected type with length constraints.

    Args:
        result: The result to validate
        expected_type: Expected type for list items
        min_length: Minimum list length (default: 0)
        max_length: Maximum list length (optional)
        context: Optional context for error message

    Raises:
        AssertionError: If validation fails
    """
    ctx_msg = f" [{context}]" if context else ""
    assert isinstance(result, list), f"Expected list, got {type(result).__name__}{ctx_msg}"
    assert len(result) >= min_length, (
        f"Expected at least {min_length} items, got {len(result)}{ctx_msg}"
    )
    if max_length is not None:
        assert len(result) <= max_length, (
            f"Expected at most {max_length} items, got {len(result)}{ctx_msg}"
        )

    invalid_items = [type(item).__name__ for item in result if not isinstance(item, expected_type)]
    assert not invalid_items, (
        f"Expected all {expected_type.__name__}, found: {invalid_items}{ctx_msg}"
    )


async def assert_completes_within(
    coro: Any,
    timeout_seconds: float,
    context: str = "",
) -> Any:
    """Assert that an async operation completes within a time limit.

    Args:
        coro: The coroutine to execute
        timeout_seconds: Maximum time allowed
        context: Optional context for error message

    Returns:
        The result of the coroutine

    Raises:
        AssertionError: If operation times out
    """
    import asyncio

    ctx_msg = f" [{context}]" if context else ""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError:
        raise AssertionError(f"Operation timed out after {timeout_seconds}s{ctx_msg}") from None


def assert_valid_uuid(value: str | None, context: str = "") -> None:
    """Assert that a value is a valid UUID string.

    Args:
        value: The value to validate
        context: Optional context for error message

    Raises:
        AssertionError: If value is not a valid UUID
    """
    import re

    ctx_msg = f" [{context}]" if context else ""
    assert value is not None, f"UUID should not be None{ctx_msg}"
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    assert uuid_pattern.match(value), f"Invalid UUID format: {value}{ctx_msg}"


def create_mock_decorator(
    capture_dict: dict[str, Callable[..., Any]],
) -> Callable[..., Callable[[Callable[..., T]], Callable[..., T]]]:
    """Create a mock decorator that captures registered functions.

    This helper creates decorators that mimic FastMCP's @mcp.tool(),
    @mcp.resource(), and @mcp.prompt() decorators for testing.

    Args:
        capture_dict: Dictionary to store captured functions

    Returns:
        A decorator factory function
    """

    def decorator_factory(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            capture_dict[func.__name__] = func
            return func

        return decorator

    return decorator_factory


# ============================================================================
# MCP SERVER CONTEXT FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def mcp_server_context() -> AsyncGenerator[tuple[AppContext, MagicMock], None]:
    """Create a fully initialized MCP server context for E2E testing.

    This fixture initializes the complete MCP server lifecycle, including:
    - Registry with native methods registered
    - Session manager for session lifecycle
    - Optional middleware based on settings
    - Telemetry initialization

    Yields:
        Tuple of (AppContext, FastMCP server instance)

    Example:
        >>> async def test_server(mcp_server_context):
        ...     ctx, server = mcp_server_context
        ...     assert ctx.initialized
        ...     assert ctx.registry.method_count > 0
    """
    mock_server = MagicMock()
    mock_server.add_middleware = MagicMock()

    async with app_lifespan(mock_server) as ctx:
        yield ctx, mock_server


@pytest_asyncio.fixture
async def initialized_app_context() -> AsyncGenerator[AppContext, None]:
    """Create an initialized AppContext without full server lifecycle.

    This provides a lighter-weight fixture for tests that only need
    the AppContext components without the full MCP server machinery.

    Yields:
        Initialized AppContext with registry, session manager, and settings
    """
    settings = Settings()
    registry = MethodRegistry()
    session_manager = SessionManager(
        max_sessions=settings.max_sessions,
        cleanup_interval=settings.session_cleanup_interval,
    )

    # Register native methods
    from reasoning_mcp.methods.native import register_all_native_methods

    register_all_native_methods(registry)

    # Initialize registry
    await registry.initialize()

    ctx = AppContext(
        registry=registry,
        session_manager=session_manager,
        settings=settings,
        initialized=True,
    )

    try:
        yield ctx
    finally:
        await session_manager.clear()


@pytest_asyncio.fixture(scope="module")
async def shared_registry() -> AsyncGenerator[MethodRegistry, None]:
    """Provide a shared, initialized registry for module-level tests.

    This fixture is module-scoped to avoid repeated expensive initialization
    of native methods. Use this when tests don't modify the registry.

    Yields:
        Initialized MethodRegistry with native methods registered
    """
    registry = MethodRegistry()

    from reasoning_mcp.methods.native import register_all_native_methods

    register_all_native_methods(registry)
    await registry.initialize()

    yield registry


@pytest.fixture
def test_settings() -> Settings:
    """Provide test-specific settings with sensible defaults.

    Returns:
        Settings instance configured for testing
    """
    return Settings(
        enable_middleware=False,  # Disable for faster tests
        enable_cache=False,  # Disable for deterministic results
        enable_rate_limiting=False,  # Disable rate limiting
        enable_sampling=True,  # Enable for sampling tests
        enable_elicitation=True,  # Enable for elicitation tests
        enable_background_tasks=True,  # Enable background tasks
        max_sessions=100,
        session_cleanup_interval=3600,
    )


@pytest.fixture
def test_settings_with_middleware() -> Settings:
    """Provide settings with middleware enabled for middleware tests.

    Returns:
        Settings instance with middleware features enabled
    """
    return Settings(
        enable_middleware=True,
        enable_middleware_metrics=True,
        middleware_log_level="DEBUG",
        enable_cache=True,
        cache_max_entries=100,
        cache_default_ttl_seconds=60,
        enable_rate_limiting=True,
        rate_limit_requests_per_minute=60,
        rate_limit_requests_per_hour=1000,
        rate_limit_burst_size=10,
    )


# ============================================================================
# SESSION FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def test_session(initialized_app_context: AppContext) -> Session:
    """Provide a test session created through the session manager.

    Args:
        initialized_app_context: The initialized app context

    Returns:
        A new Session instance ready for testing
    """
    session = await initialized_app_context.session_manager.create()
    return session


@pytest_asyncio.fixture
async def active_reasoning_session(
    initialized_app_context: AppContext,
) -> AsyncGenerator[Session, None]:
    """Provide an active session with initial reasoning started.

    Creates a session and adds an initial thought to simulate
    an active reasoning process.

    Args:
        initialized_app_context: The initialized app context

    Yields:
        Session with initial thought added
    """
    session = await initialized_app_context.session_manager.create()
    session.start()

    # Add initial thought
    initial_thought = ThoughtNode(
        id=str(uuid4()),
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Let's analyze this problem step by step.",
        confidence=0.8,
        depth=0,
    )
    session.add_thought(initial_thought)

    try:
        yield session
    finally:
        # Cleanup session
        await initialized_app_context.session_manager.delete(session.id)


@pytest_asyncio.fixture
async def session_with_branches(
    initialized_app_context: AppContext,
) -> AsyncGenerator[Session, None]:
    """Provide a session with multiple branches for merge testing.

    Creates a session with an initial thought and two branches
    for testing branch/merge operations.

    Args:
        initialized_app_context: The initialized app context

    Yields:
        Session with branches for testing
    """
    session = await initialized_app_context.session_manager.create()
    session.start()

    # Add root thought
    root = ThoughtNode(
        id="root-thought",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Analyzing problem from multiple perspectives.",
        confidence=0.85,
        depth=0,
    )
    session.add_thought(root)

    # Add branch 1
    branch1 = ThoughtNode(
        id="branch1-thought",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Approach 1: Direct solution path.",
        parent_id="root-thought",
        branch_id="branch-1",
        confidence=0.75,
        depth=1,
    )
    session.add_thought(branch1)

    # Add branch 2
    branch2 = ThoughtNode(
        id="branch2-thought",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Approach 2: Alternative exploration.",
        parent_id="root-thought",
        branch_id="branch-2",
        confidence=0.70,
        depth=1,
    )
    session.add_thought(branch2)

    try:
        yield session
    finally:
        await initialized_app_context.session_manager.delete(session.id)


# ============================================================================
# MOCK REASONING METHOD FIXTURES
# ============================================================================


@pytest.fixture
def mock_reasoning_method() -> ReasoningMethod:
    """Provide a mock reasoning method for testing.

    Returns:
        Mock ReasoningMethod that returns deterministic results
    """
    mock = MagicMock(spec=ReasoningMethod)
    mock.identifier = "mock_method"
    mock.name = "Mock Method"
    mock.description = "A mock method for testing"
    mock.category = "core"
    mock.streaming_context = None

    async def mock_initialize() -> None:
        pass

    async def mock_execute(
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content=f"Mock reasoning result for: {input_text[:100]}",
            confidence=0.85,
            depth=0,
        )

    async def mock_continue_reasoning(
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
            content=f"Continued reasoning: {guidance or 'next step'}",
            parent_id=previous_thought.id,
            confidence=0.80,
            depth=previous_thought.depth + 1,
        )

    async def mock_health_check() -> bool:
        return True

    mock.initialize = AsyncMock(side_effect=mock_initialize)
    mock.execute = AsyncMock(side_effect=mock_execute)
    mock.continue_reasoning = AsyncMock(side_effect=mock_continue_reasoning)
    mock.health_check = AsyncMock(side_effect=mock_health_check)

    return mock


@pytest.fixture
def mock_method_metadata() -> MethodMetadata:
    """Provide mock method metadata for registration.

    Returns:
        MethodMetadata for mock method registration
    """
    return MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Mock Chain of Thought",
        description="Mock CoT method for testing",
        category=MethodCategory.CORE,
        tags=frozenset({"test", "mock", "reasoning"}),
        complexity=3,
        supports_branching=False,
        supports_revision=True,
        min_thoughts=1,
        max_thoughts=10,
    )


# ============================================================================
# LLM MOCK FIXTURES (for deterministic E2E testing)
# ============================================================================


@pytest.fixture
def mock_llm_response() -> str:
    """Provide a standard mock LLM response.

    Returns:
        Mock response string for LLM calls
    """
    return """Based on my analysis, the solution involves:

1. Understanding the core problem
2. Breaking it down into manageable parts
3. Addressing each part systematically
4. Synthesizing the results

Confidence: 0.85"""


@pytest.fixture
def mock_sampling_context():
    """Provide a mock sampling context for Context.sample() calls.

    Returns:
        Mock that simulates FastMCP Context.sample() behavior
    """
    mock_ctx = MagicMock()

    async def mock_sample(
        prompt: str,
        *,
        tools: list[Any] | None = None,
        max_tokens: int = 1000,
    ) -> str:
        return f"Mock LLM response to: {prompt[:50]}..."

    mock_ctx.sample = AsyncMock(side_effect=mock_sample)
    return mock_ctx


@pytest.fixture
def patch_llm_calls(mock_llm_response: str) -> Any:
    """Context manager to patch all LLM calls with deterministic responses.

    This ensures E2E tests are reproducible without actual LLM calls.

    Args:
        mock_llm_response: The mock response to use

    Returns:
        Patch context manager
    """
    return patch(
        "reasoning_mcp.methods.base.ReasoningMethod.sample_llm",
        new_callable=AsyncMock,
        return_value=mock_llm_response,
    )


# ============================================================================
# TOOL EXECUTION FIXTURES
# ============================================================================


@pytest.fixture
def sample_problem() -> str:
    """Provide a sample problem for reasoning tests.

    Returns:
        Sample problem string
    """
    return (
        "What are the key factors to consider when designing a scalable microservices architecture?"
    )


@pytest.fixture
def empty_problem() -> str:
    """Provide an empty problem string for edge case testing.

    Returns:
        Empty string
    """
    return ""


@pytest.fixture
def unicode_problem() -> str:
    """Provide a problem with unicode characters for edge case testing.

    Returns:
        Problem string with various unicode characters
    """
    return "分析这个问题：如何优化系统性能？ — Using émojis 🤔 and spëcial çharacters"


@pytest.fixture
def long_problem() -> str:
    """Provide a very long problem string for edge case testing.

    Returns:
        Long problem string (10KB+)
    """
    base = "Consider the following complex scenario involving multiple factors. "
    return base * 200  # ~12KB


@pytest.fixture
def sample_math_problem() -> str:
    """Provide a sample mathematical problem.

    Returns:
        Math problem string
    """
    return "If a train travels at 60 mph for 2.5 hours, how far does it travel?"


@pytest.fixture
def sample_ethical_problem() -> str:
    """Provide a sample ethical dilemma.

    Returns:
        Ethical problem string
    """
    return "Should autonomous vehicles prioritize passenger safety over pedestrian safety in unavoidable accident scenarios?"


@pytest.fixture
def sample_code_problem() -> str:
    """Provide a sample code analysis problem.

    Returns:
        Code problem string
    """
    return """Analyze this Python function for potential issues:

def process_data(items):
    result = []
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i] > items[j]:
                result.append((i, j))
    return result
"""


# ============================================================================
# PIPELINE FIXTURES
# ============================================================================


@pytest.fixture
def sample_sequence_pipeline() -> dict[str, Any]:
    """Provide a sample sequence pipeline configuration.

    Returns:
        Pipeline dict for compose tool (using stage_type discriminator)
    """
    return {
        "stage_type": "sequence",
        "name": "test_sequence",
        "stages": [
            {
                "stage_type": "method",
                "method_id": "chain_of_thought",
                "name": "initial_analysis",
            },
            {
                "stage_type": "method",
                "method_id": "self_reflection",
                "name": "reflection",
            },
        ],
    }


@pytest.fixture
def sample_parallel_pipeline() -> dict[str, Any]:
    """Provide a sample parallel pipeline configuration.

    Returns:
        Pipeline dict for compose tool with parallel execution
    """
    return {
        "stage_type": "parallel",
        "name": "test_parallel",
        "branches": [
            {"stage_type": "method", "method_id": "ethical_reasoning"},
            {"stage_type": "method", "method_id": "causal_reasoning"},
        ],
        "merge_strategy": {"name": "vote", "selection_criteria": "highest_confidence"},
        "max_concurrency": 2,
    }


@pytest.fixture
def sample_conditional_pipeline() -> dict[str, Any]:
    """Provide a sample conditional pipeline configuration.

    Returns:
        Pipeline dict with conditional branching
    """
    return {
        "stage_type": "conditional",
        "name": "test_conditional",
        "condition": {
            "name": "complexity_check",
            "expression": "complexity > 0.7",
            "operator": ">",
            "threshold": 0.7,
            "field": "complexity",
        },
        "if_true": {"stage_type": "method", "method_id": "tree_of_thoughts"},
        "if_false": {"stage_type": "method", "method_id": "chain_of_thought"},
    }


# ============================================================================
# RESOURCE AND PROMPT FIXTURES
# ============================================================================


@pytest.fixture
def sample_method_ids() -> list[str]:
    """Provide sample method IDs for testing.

    Returns:
        List of valid method identifiers
    """
    return [
        "chain_of_thought",
        "tree_of_thoughts",
        "self_reflection",
        "ethical_reasoning",
        "react",
    ]


@pytest.fixture
def sample_hints() -> dict[str, Any]:
    """Provide sample hints for method selection.

    Returns:
        Hints dict for reason tool
    """
    return {
        "domain": "software_engineering",
        "complexity": "medium",
        "prefer_methods": ["chain_of_thought", "react"],
        "avoid_methods": ["ethical_reasoning"],
        "custom_hints": {"context": "backend development"},
    }


# ============================================================================
# BACKGROUND TASK FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def task_manager() -> AsyncGenerator[TaskManager, None]:
    """Provide the global task manager for background task testing.

    Yields:
        TaskManager instance with automatic cleanup
    """
    from reasoning_mcp.tasks import get_task_manager

    manager = get_task_manager()
    yield manager

    # Clean up any test tasks
    manager.tasks.clear()


# ============================================================================
# CLEANUP AND UTILITY FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_global_state() -> Generator[None, None, None]:
    """Reset global state before each test.

    This ensures tests are isolated and don't affect each other.
    """
    # Reset global app context
    import reasoning_mcp.server as server_module

    original_context = server_module._APP_CONTEXT
    server_module._APP_CONTEXT = None

    yield

    # Restore original state if needed
    server_module._APP_CONTEXT = original_context


@pytest.fixture
def capture_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Configure log capturing for test assertions.

    Args:
        caplog: pytest log capture fixture

    Returns:
        Configured log capture
    """
    caplog.set_level(logging.DEBUG, logger="reasoning_mcp")
    return caplog


# ============================================================================
# MCP SERVER INSTANCE FIXTURE
# ============================================================================


@pytest.fixture
def mcp_instance() -> Any:
    """Provide the actual MCP server instance for inspection.

    Returns:
        The reasoning-mcp FastMCP server instance
    """
    return mcp
