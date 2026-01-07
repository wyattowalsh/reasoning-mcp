"""
Pytest configuration and shared fixtures for reasoning-mcp tests.

This module provides comprehensive fixtures for integration testing:
- Session fixtures with various configurations
- Registry fixtures for method management
- Server fixtures for MCP testing
- Pipeline fixtures for workflow testing
- Plugin fixtures for extensibility testing
- Data fixtures for sample problems and graphs
"""

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from anyio import create_task_group

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    PipelineStageType,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.pipeline import (
    Accumulator,
    Condition,
    ConditionalPipeline,
    ErrorHandler,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    SequencePipeline,
    SwitchPipeline,
    Transform,
)
from reasoning_mcp.models.session import Session, SessionConfig, SessionMetrics
from reasoning_mcp.models.thought import ThoughtEdge, ThoughtGraph, ThoughtNode
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.server import AppContext, mcp
from reasoning_mcp.sessions import SessionManager


# ============================================================================
# SESSION FIXTURES
# ============================================================================


@pytest.fixture
def default_session_config() -> SessionConfig:
    """Provide a default session configuration for testing.

    Returns:
        A SessionConfig with standard default values.
    """
    return SessionConfig()


@pytest.fixture
def custom_session_config() -> SessionConfig:
    """Provide a custom session configuration with modified parameters.

    Returns:
        A SessionConfig with custom values for testing edge cases.
    """
    return SessionConfig(
        max_depth=20,
        max_thoughts=200,
        timeout_seconds=600.0,
        enable_branching=True,
        max_branches=10,
        auto_prune=True,
        min_confidence_threshold=0.5,
        metadata={"test": "custom_config"},
    )


@pytest_asyncio.fixture
async def test_session() -> Session:
    """Provide a basic test session with default configuration.

    Returns:
        A new Session instance ready for testing.
    """
    return Session()


@pytest_asyncio.fixture
async def active_session() -> Session:
    """Provide an active session that has been started.

    Returns:
        A Session instance in ACTIVE status.
    """
    session = Session()
    session.start()
    return session


@pytest_asyncio.fixture
async def session_with_thoughts() -> Session:
    """Provide a session pre-populated with sample thoughts.

    Returns:
        A Session containing a small graph of related thoughts.
    """
    session = Session().start()

    # Add initial thought
    initial = ThoughtNode(
        id="thought-initial",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Let's analyze this problem step by step.",
        confidence=0.8,
        depth=0,
    )
    session.add_thought(initial)

    # Add continuation
    continuation = ThoughtNode(
        id="thought-cont-1",
        type=ThoughtType.CONTINUATION,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="First, we need to understand the requirements.",
        parent_id="thought-initial",
        confidence=0.85,
        depth=1,
    )
    session.add_thought(continuation)

    # Add branch
    branch = ThoughtNode(
        id="thought-branch-1",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Alternative approach: consider edge cases first.",
        parent_id="thought-initial",
        branch_id="branch-1",
        confidence=0.75,
        depth=1,
    )
    session.add_thought(branch)

    return session


# ============================================================================
# REGISTRY FIXTURES
# ============================================================================


@pytest.fixture
def empty_registry() -> MethodRegistry:
    """Provide an empty method registry.

    Returns:
        A MethodRegistry with no methods registered.
    """
    return MethodRegistry()


@pytest.fixture
def mock_reasoning_method() -> ReasoningMethod:
    """Provide a mock reasoning method for testing.

    Returns:
        A mock ReasoningMethod that satisfies the protocol.
    """
    mock = MagicMock(spec=ReasoningMethod)
    mock.identifier = "mock_method"
    mock.name = "Mock Method"
    mock.description = "A mock method for testing"
    mock.category = "core"
    mock.initialize = AsyncMock(return_value=None)
    mock.execute = AsyncMock(
        return_value=ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Mock thought",
        )
    )
    mock.continue_reasoning = AsyncMock(
        return_value=ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Mock continuation",
        )
    )
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def populated_registry(mock_reasoning_method: ReasoningMethod) -> MethodRegistry:
    """Provide a registry with mock methods registered.

    Args:
        mock_reasoning_method: A mock reasoning method to register.

    Returns:
        A MethodRegistry with several mock methods registered.
    """
    registry = MethodRegistry()

    # Register mock method
    metadata = MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Mock Chain of Thought",
        description="Mock CoT for testing",
        category=MethodCategory.CORE,
        tags=frozenset({"test", "mock"}),
    )
    registry.register(mock_reasoning_method, metadata)

    return registry


# ============================================================================
# SERVER FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def test_server_context() -> AsyncGenerator[AppContext, None]:
    """Provide a test server application context.

    Yields:
        An AppContext with initialized components.
    """
    from reasoning_mcp.config import Settings

    settings = Settings()
    registry = MethodRegistry()
    session_manager = SessionManager(max_sessions=10, cleanup_interval=3600)

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


@pytest_asyncio.fixture
async def session_manager() -> AsyncGenerator[SessionManager, None]:
    """Provide a test session manager.

    Yields:
        A SessionManager instance for testing.
    """
    manager = SessionManager(max_sessions=50, cleanup_interval=1800)
    try:
        yield manager
    finally:
        await manager.clear()


# ============================================================================
# PIPELINE FIXTURES
# ============================================================================


@pytest.fixture
def sample_method_stage() -> MethodStage:
    """Provide a sample method stage for pipeline testing.

    Returns:
        A MethodStage configured for testing.
    """
    return MethodStage(
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="test_stage",
        max_thoughts=10,
        timeout_seconds=60.0,
    )


@pytest.fixture
def sample_sequence_pipeline() -> SequencePipeline:
    """Provide a sample sequence pipeline.

    Returns:
        A SequencePipeline with multiple stages.
    """
    return SequencePipeline(
        name="test_sequence",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="stage_1"
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION, name="stage_2"
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="stage_3"
            ),
        ],
    )


@pytest.fixture
def sample_parallel_pipeline() -> ParallelPipeline:
    """Provide a sample parallel pipeline.

    Returns:
        A ParallelPipeline with multiple branches.
    """
    return ParallelPipeline(
        name="test_parallel",
        branches=[
            MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
            MethodStage(method_id=MethodIdentifier.CAUSAL_REASONING),
            MethodStage(method_id=MethodIdentifier.COUNTERFACTUAL),
        ],
        merge_strategy=MergeStrategy(
            name="vote", selection_criteria="most_common_conclusion"
        ),
        max_concurrency=3,
    )


@pytest.fixture
def sample_conditional_pipeline() -> ConditionalPipeline:
    """Provide a sample conditional pipeline.

    Returns:
        A ConditionalPipeline with branching logic.
    """
    return ConditionalPipeline(
        name="test_conditional",
        condition=Condition(
            name="confidence_check",
            expression="confidence > 0.8",
            operator=">",
            threshold=0.8,
            field="confidence",
        ),
        if_true=MethodStage(method_id=MethodIdentifier.SELF_REFLECTION),
        if_false=SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.SOCRATIC),
                MethodStage(method_id=MethodIdentifier.DIALECTIC),
            ]
        ),
    )


@pytest.fixture
def sample_loop_pipeline() -> LoopPipeline:
    """Provide a sample loop pipeline.

    Returns:
        A LoopPipeline with iteration logic.
    """
    return LoopPipeline(
        name="test_loop",
        body=MethodStage(method_id=MethodIdentifier.SELF_REFLECTION),
        condition=Condition(
            name="quality_threshold",
            expression="quality_score > 0.9",
            operator=">",
            threshold=0.9,
            field="quality_score",
        ),
        max_iterations=5,
        accumulator=Accumulator(
            name="improvement_tracker",
            initial_value=[],
            operation="append",
            field="content",
        ),
    )


@pytest.fixture
def sample_switch_pipeline() -> SwitchPipeline:
    """Provide a sample switch pipeline.

    Returns:
        A SwitchPipeline with multiple cases.
    """
    return SwitchPipeline(
        name="test_switch",
        expression="problem_type",
        cases={
            "ethical": MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
            "mathematical": MethodStage(
                method_id=MethodIdentifier.MATHEMATICAL_REASONING
            ),
            "code": MethodStage(method_id=MethodIdentifier.CODE_REASONING),
        },
        default=MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
    )


# ============================================================================
# DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_mathematical_problem() -> dict[str, Any]:
    """Provide a sample mathematical problem for testing.

    Returns:
        A dictionary containing a mathematical problem and context.
    """
    return {
        "problem": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "type": "mathematical",
        "difficulty": "easy",
        "expected_answer": "150 miles",
        "context": {"units": "imperial", "domain": "physics"},
    }


@pytest.fixture
def sample_ethical_problem() -> dict[str, Any]:
    """Provide a sample ethical problem for testing.

    Returns:
        A dictionary containing an ethical dilemma and context.
    """
    return {
        "problem": "Should autonomous vehicles prioritize passenger safety over pedestrian safety?",
        "type": "ethical",
        "difficulty": "hard",
        "context": {
            "stakeholders": ["passengers", "pedestrians", "manufacturers"],
            "principles": ["safety", "fairness", "transparency"],
        },
    }


@pytest.fixture
def sample_code_problem() -> dict[str, Any]:
    """Provide a sample code analysis problem for testing.

    Returns:
        A dictionary containing a code problem and context.
    """
    return {
        "problem": "Optimize this function to reduce time complexity from O(n^2) to O(n log n)",
        "type": "code",
        "difficulty": "medium",
        "code": "def sort_list(lst):\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] > lst[j]:\n                lst[i], lst[j] = lst[j], lst[i]\n    return lst",
        "language": "python",
        "context": {"optimization_target": "time_complexity"},
    }


@pytest.fixture
def sample_thought_graph() -> ThoughtGraph:
    """Provide a sample thought graph for testing.

    Returns:
        A ThoughtGraph with interconnected thoughts.
    """
    graph = ThoughtGraph()

    # Create root thought
    root = ThoughtNode(
        id="root",
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Root analysis",
        confidence=0.8,
        depth=0,
    )
    graph.add_thought(root)

    # Create child thoughts
    child1 = ThoughtNode(
        id="child1",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Branch 1 exploration",
        parent_id="root",
        branch_id="branch-1",
        confidence=0.75,
        depth=1,
    )
    graph.add_thought(child1)

    child2 = ThoughtNode(
        id="child2",
        type=ThoughtType.BRANCH,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Branch 2 exploration",
        parent_id="root",
        branch_id="branch-2",
        confidence=0.85,
        depth=1,
    )
    graph.add_thought(child2)

    # Create synthesis thought
    synthesis = ThoughtNode(
        id="synthesis",
        type=ThoughtType.SYNTHESIS,
        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        content="Combined insights from both branches",
        confidence=0.9,
        depth=2,
    )
    graph.add_thought(synthesis)

    return graph


@pytest.fixture
async def async_task_group() -> AsyncGenerator:
    """Provide an anyio task group for testing concurrent operations.

    Yields:
        An anyio TaskGroup for managing concurrent tasks in tests.
    """
    async with create_task_group() as tg:
        yield tg


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers and settings.

    This function is called before test collection begins.

    Args:
        config: The pytest configuration object.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring external services"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as fast unit tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Modify test items during collection.

    This function can be used to add markers to tests based on their location
    or other criteria.

    Args:
        config: The pytest configuration object.
        items: List of collected test items.
    """
    # Automatically mark tests in integration/ subdirectory
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        else:
            # Default to unit test if not otherwise marked
            if not any(marker.name in ["integration", "e2e", "slow"]
                      for marker in item.iter_markers()):
                item.add_marker(pytest.mark.unit)
