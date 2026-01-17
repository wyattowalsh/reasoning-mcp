"""Tests for MCTS method with elicitation support.

This module tests the elicitation integration in the MCTS reasoning method.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.mcts import MCTS
from reasoning_mcp.models.core import SessionStatus, ThoughtType
from reasoning_mcp.models.session import Session


@pytest.fixture
def method() -> MCTS:
    """Provide an MCTS method instance for testing.

    Returns:
        MCTS instance with elicitation enabled.
    """
    return MCTS(
        num_iterations=5,  # Small number for testing
        max_depth=2,
        branching_factor=2,
        enable_elicitation=True,
    )


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.mark.asyncio
async def test_mcts_initialization_with_elicitation():
    """Test MCTS initialization with elicitation enabled."""
    mcts = MCTS(enable_elicitation=True)
    await mcts.initialize()
    assert mcts.enable_elicitation is True
    assert mcts._execution_context is None
    assert await mcts.health_check()


@pytest.mark.asyncio
async def test_mcts_initialization_without_elicitation():
    """Test MCTS initialization with elicitation disabled."""
    mcts = MCTS(enable_elicitation=False)
    await mcts.initialize()
    assert mcts.enable_elicitation is False
    assert await mcts.health_check()


@pytest.mark.asyncio
async def test_mcts_execute_without_execution_context(method: MCTS, session: Session):
    """Test MCTS execution without execution context (no elicitation)."""
    await method.initialize()

    result = await method.execute(
        session=session,
        input_text="Test decision problem",
        context=None,
        execution_context=None,
    )

    # Verify basic execution works
    assert result is not None
    assert result.type == ThoughtType.SYNTHESIS
    assert session.is_active
    assert session.status == SessionStatus.ACTIVE

    # Verify no elicitations were made (no execution context)
    assert session.metrics.elicitations_made == 0


@pytest.mark.asyncio
async def test_mcts_disable_elicitation_flag(session: Session):
    """Test MCTS with elicitation disabled via flag."""
    mcts = MCTS(
        num_iterations=5,
        max_depth=2,
        branching_factor=2,
        enable_elicitation=False,  # Disabled
    )
    await mcts.initialize()

    result = await mcts.execute(
        session=session,
        input_text="Test decision problem",
        execution_context=None,
    )

    assert result is not None
    # No elicitations should be made when disabled
    assert session.metrics.elicitations_made == 0


@pytest.mark.asyncio
async def test_mcts_phases_execute(method: MCTS, session: Session):
    """Test that all MCTS phases execute correctly."""
    await method.initialize()

    result = await method.execute(
        session=session,
        input_text="Optimize resource allocation",
        context={"num_iterations": 3, "max_depth": 2},
    )

    # Verify phases completed
    assert result is not None
    assert result.type == ThoughtType.SYNTHESIS
    assert "MCTS Decision Complete" in result.content
    assert "Best decision path found" in result.content

    # Verify metadata
    assert result.metadata.get("is_final") is True
    assert result.metadata.get("total_iterations") == 3


@pytest.mark.asyncio
async def test_mcts_selection_without_elicitation(method: MCTS, session: Session):
    """Test MCTS selection phase without elicitation."""
    await method.initialize()

    # Execute with minimal iterations
    result = await method.execute(
        session=session,
        input_text="Test problem",
        context={"num_iterations": 2},
    )

    # Should complete successfully without elicitation
    assert result is not None
    assert session.metrics.elicitations_made == 0


@pytest.mark.asyncio
async def test_mcts_simulation_without_elicitation(method: MCTS, session: Session):
    """Test MCTS simulation phase without elicitation."""
    await method.initialize()

    result = await method.execute(
        session=session,
        input_text="Test problem",
        context={"num_iterations": 5, "simulation_depth": 2},
    )

    # Simulation should work without user ratings
    assert result is not None
    assert result.metadata.get("total_iterations") == 5


@pytest.mark.asyncio
async def test_mcts_continue_reasoning(method: MCTS, session: Session):
    """Test MCTS continue_reasoning functionality."""
    await method.initialize()

    initial_result = await method.execute(
        session=session,
        input_text="Test decision",
        context={"num_iterations": 3},
    )

    # Continue reasoning from the result
    continued = await method.continue_reasoning(
        session=session,
        previous_thought=initial_result,
        guidance="Focus on cost optimization",
        context={"num_iterations": 2},
    )

    assert continued is not None
    assert continued.type == ThoughtType.CONTINUATION
    assert continued.parent_id == initial_result.id
    assert "cost optimization" in continued.content.lower()
