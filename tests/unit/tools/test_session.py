"""
Comprehensive tests for session management tools in reasoning_mcp.tools.session.

This module provides complete test coverage for all session tools:
- session_continue: Continue reasoning with optional guidance
- session_branch: Create new branches with optional parent thought
- session_inspect: Inspect session state with optional graph
- session_merge: Merge branches with different strategies

Each tool is tested for:
1. Minimal and full parameter usage
2. Return type correctness
3. Async function behavior
4. Default parameter values
5. Edge cases and variations
"""

import pytest

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.tools import BranchOutput, MergeOutput, SessionState, ThoughtOutput
from reasoning_mcp.tools.session import (
    session_branch,
    session_continue,
    session_inspect,
    session_merge,
)


# ============================================================================
# Test session_continue
# ============================================================================


class TestSessionContinue:
    """Test suite for session_continue tool."""

    @pytest.mark.asyncio
    async def test_session_continue_without_guidance(self):
        """Test session_continue without providing guidance."""
        result = await session_continue(session_id="session-123")

        # Verify return type
        assert isinstance(result, ThoughtOutput)

        # Verify required fields are present
        assert result.id is not None
        assert result.content is not None
        assert result.thought_type is not None

        # Verify placeholder implementation details
        assert result.id == "placeholder-thought-id"
        assert "session-123" in result.content
        assert "None" in result.content
        assert result.thought_type == ThoughtType.CONTINUATION

    @pytest.mark.asyncio
    async def test_session_continue_with_guidance(self):
        """Test session_continue with guidance parameter."""
        guidance = "Focus on ethical implications"
        result = await session_continue(
            session_id="session-456",
            guidance=guidance,
        )

        # Verify return type
        assert isinstance(result, ThoughtOutput)

        # Verify required fields
        assert result.id is not None
        assert result.content is not None
        assert result.thought_type is not None

        # Verify guidance is reflected in content
        assert "session-456" in result.content
        assert guidance in result.content

    @pytest.mark.asyncio
    async def test_session_continue_with_empty_guidance(self):
        """Test session_continue with empty string guidance."""
        result = await session_continue(
            session_id="session-789",
            guidance="",
        )

        # Should handle empty string gracefully
        assert isinstance(result, ThoughtOutput)
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_session_continue_returns_thought_output(self):
        """Test that session_continue returns ThoughtOutput type."""
        result = await session_continue(session_id="test-session")

        # Verify exact type
        assert type(result).__name__ == "ThoughtOutput"
        assert hasattr(result, "id")
        assert hasattr(result, "content")
        assert hasattr(result, "thought_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "step_number")

    @pytest.mark.asyncio
    async def test_session_continue_is_async(self):
        """Test that session_continue is an async function."""
        import inspect

        assert inspect.iscoroutinefunction(session_continue)

    @pytest.mark.asyncio
    async def test_session_continue_thought_output_fields(self):
        """Test that session_continue returns ThoughtOutput with expected field values."""
        result = await session_continue(
            session_id="field-test",
            guidance="test guidance",
        )

        # Verify ThoughtOutput fields
        assert isinstance(result.id, str)
        assert isinstance(result.content, str)
        assert isinstance(result.thought_type, ThoughtType)
        assert isinstance(result.confidence, (float, type(None)))
        assert isinstance(result.step_number, (int, type(None)))

        # Verify confidence is in valid range if set
        if result.confidence is not None:
            assert 0.0 <= result.confidence <= 1.0

        # Verify step_number is non-negative if set
        if result.step_number is not None:
            assert result.step_number >= 0


# ============================================================================
# Test session_branch
# ============================================================================


class TestSessionBranch:
    """Test suite for session_branch tool."""

    @pytest.mark.asyncio
    async def test_session_branch_without_from_thought_id(self):
        """Test session_branch without specifying from_thought_id (branches from current head)."""
        result = await session_branch(
            session_id="session-123",
            branch_name="alternative-approach",
        )

        # Verify return type
        assert isinstance(result, BranchOutput)

        # Verify required fields
        assert result.branch_id is not None
        assert result.parent_thought_id is not None
        assert result.session_id is not None
        assert isinstance(result.success, bool)

        # Verify placeholder implementation
        assert "alternative-approach" in result.branch_id
        assert result.parent_thought_id == "current-head"
        assert result.session_id == "session-123"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_branch_with_from_thought_id(self):
        """Test session_branch with specific from_thought_id."""
        result = await session_branch(
            session_id="session-456",
            branch_name="explore-ethics",
            from_thought_id="thought-789",
        )

        # Verify return type
        assert isinstance(result, BranchOutput)

        # Verify from_thought_id is used
        assert result.parent_thought_id == "thought-789"
        assert result.session_id == "session-456"
        assert "explore-ethics" in result.branch_id

    @pytest.mark.asyncio
    async def test_session_branch_with_none_from_thought_id(self):
        """Test session_branch with explicitly None from_thought_id."""
        result = await session_branch(
            session_id="session-000",
            branch_name="test-branch",
            from_thought_id=None,
        )

        # Should use current head when from_thought_id is None
        assert isinstance(result, BranchOutput)
        assert result.parent_thought_id == "current-head"

    @pytest.mark.asyncio
    async def test_session_branch_returns_branch_output(self):
        """Test that session_branch returns BranchOutput type."""
        result = await session_branch(
            session_id="type-test",
            branch_name="test",
        )

        # Verify exact type
        assert type(result).__name__ == "BranchOutput"
        assert hasattr(result, "branch_id")
        assert hasattr(result, "parent_thought_id")
        assert hasattr(result, "session_id")
        assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_session_branch_is_async(self):
        """Test that session_branch is an async function."""
        import inspect

        assert inspect.iscoroutinefunction(session_branch)

    @pytest.mark.asyncio
    async def test_session_branch_output_fields(self):
        """Test that session_branch returns BranchOutput with expected field types."""
        result = await session_branch(
            session_id="field-test",
            branch_name="test-branch",
            from_thought_id="thought-123",
        )

        # Verify BranchOutput field types
        assert isinstance(result.branch_id, str)
        assert isinstance(result.parent_thought_id, str)
        assert isinstance(result.session_id, str)
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_session_branch_different_branch_names(self):
        """Test session_branch with various branch names."""
        branch_names = [
            "main",
            "feature-123",
            "ethical-analysis",
            "experiment_v2",
            "test",
        ]

        for name in branch_names:
            result = await session_branch(
                session_id="multi-test",
                branch_name=name,
            )
            assert isinstance(result, BranchOutput)
            assert name in result.branch_id


# ============================================================================
# Test session_inspect
# ============================================================================


class TestSessionInspect:
    """Test suite for session_inspect tool."""

    @pytest.mark.asyncio
    async def test_session_inspect_without_include_graph(self):
        """Test session_inspect without include_graph (default False)."""
        result = await session_inspect(session_id="session-123")

        # Verify return type
        assert isinstance(result, SessionState)

        # Verify required fields
        assert result.session_id is not None
        assert result.status is not None
        assert isinstance(result.thought_count, int)
        assert isinstance(result.branch_count, int)

        # Verify placeholder implementation
        assert result.session_id == "session-123"
        assert result.status == SessionStatus.ACTIVE
        assert result.thought_count == 0
        assert result.branch_count == 0
        assert result.current_method == MethodIdentifier.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_session_inspect_with_include_graph_true(self):
        """Test session_inspect with include_graph=True."""
        result = await session_inspect(
            session_id="session-456",
            include_graph=True,
        )

        # Verify return type
        assert isinstance(result, SessionState)
        assert result.session_id == "session-456"

        # Graph data would be in metadata if implemented
        # For now, just verify the function accepts the parameter

    @pytest.mark.asyncio
    async def test_session_inspect_with_include_graph_false(self):
        """Test session_inspect with include_graph=False explicitly."""
        result = await session_inspect(
            session_id="session-789",
            include_graph=False,
        )

        # Verify return type
        assert isinstance(result, SessionState)
        assert result.session_id == "session-789"

    @pytest.mark.asyncio
    async def test_session_inspect_returns_session_state(self):
        """Test that session_inspect returns SessionState type."""
        result = await session_inspect(session_id="type-test")

        # Verify exact type
        assert type(result).__name__ == "SessionState"
        assert hasattr(result, "session_id")
        assert hasattr(result, "status")
        assert hasattr(result, "thought_count")
        assert hasattr(result, "branch_count")
        assert hasattr(result, "current_method")
        assert hasattr(result, "started_at")
        assert hasattr(result, "updated_at")

    @pytest.mark.asyncio
    async def test_session_inspect_is_async(self):
        """Test that session_inspect is an async function."""
        import inspect

        assert inspect.iscoroutinefunction(session_inspect)

    @pytest.mark.asyncio
    async def test_session_inspect_output_fields(self):
        """Test that session_inspect returns SessionState with expected field types."""
        result = await session_inspect(
            session_id="field-test",
            include_graph=True,
        )

        # Verify SessionState field types
        assert isinstance(result.session_id, str)
        assert isinstance(result.status, SessionStatus)
        assert isinstance(result.thought_count, int)
        assert isinstance(result.branch_count, int)

        # Verify counts are non-negative
        assert result.thought_count >= 0
        assert result.branch_count >= 0

        # Optional fields can be None or specific types
        assert result.current_method is None or isinstance(
            result.current_method, MethodIdentifier
        )

    @pytest.mark.asyncio
    async def test_session_inspect_datetime_fields(self):
        """Test that session_inspect includes datetime fields."""
        from datetime import datetime

        result = await session_inspect(session_id="datetime-test")

        # Verify datetime fields exist and have correct types if set
        assert hasattr(result, "started_at")
        assert hasattr(result, "updated_at")

        if result.started_at is not None:
            assert isinstance(result.started_at, datetime)

        if result.updated_at is not None:
            assert isinstance(result.updated_at, datetime)


# ============================================================================
# Test session_merge
# ============================================================================


class TestSessionMerge:
    """Test suite for session_merge tool."""

    @pytest.mark.asyncio
    async def test_session_merge_default_strategy(self):
        """Test session_merge with default strategy (latest)."""
        result = await session_merge(
            session_id="session-123",
            source_branch="branch-alt",
            target_branch="main",
        )

        # Verify return type
        assert isinstance(result, MergeOutput)

        # Verify required fields
        assert result.merged_thought_id is not None
        assert result.source_branch_ids is not None
        assert result.session_id is not None
        assert isinstance(result.success, bool)

        # Verify placeholder implementation
        assert result.session_id == "session-123"
        assert "branch-alt" in result.merged_thought_id
        assert "main" in result.merged_thought_id
        assert result.success is True
        assert len(result.source_branch_ids) > 0

    @pytest.mark.asyncio
    async def test_session_merge_latest_strategy(self):
        """Test session_merge with 'latest' strategy explicitly."""
        result = await session_merge(
            session_id="session-456",
            source_branch="branch-ethical",
            target_branch="branch-practical",
            strategy="latest",
        )

        # Verify return type and success
        assert isinstance(result, MergeOutput)
        assert result.session_id == "session-456"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_merge_highest_confidence_strategy(self):
        """Test session_merge with 'highest_confidence' strategy."""
        result = await session_merge(
            session_id="session-789",
            source_branch="branch-a",
            target_branch="branch-b",
            strategy="highest_confidence",
        )

        # Verify the function accepts this strategy
        assert isinstance(result, MergeOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_merge_synthesis_strategy(self):
        """Test session_merge with 'synthesis' strategy."""
        result = await session_merge(
            session_id="session-abc",
            source_branch="branch-creative",
            target_branch="branch-analytical",
            strategy="synthesis",
        )

        # Verify the function accepts this strategy
        assert isinstance(result, MergeOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_merge_sequential_strategy(self):
        """Test session_merge with 'sequential' strategy."""
        result = await session_merge(
            session_id="session-def",
            source_branch="branch-step1",
            target_branch="branch-step2",
            strategy="sequential",
        )

        # Verify the function accepts this strategy
        assert isinstance(result, MergeOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_session_merge_returns_merge_output(self):
        """Test that session_merge returns MergeOutput type."""
        result = await session_merge(
            session_id="type-test",
            source_branch="src",
            target_branch="tgt",
        )

        # Verify exact type
        assert type(result).__name__ == "MergeOutput"
        assert hasattr(result, "merged_thought_id")
        assert hasattr(result, "source_branch_ids")
        assert hasattr(result, "session_id")
        assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_session_merge_is_async(self):
        """Test that session_merge is an async function."""
        import inspect

        assert inspect.iscoroutinefunction(session_merge)

    @pytest.mark.asyncio
    async def test_session_merge_output_fields(self):
        """Test that session_merge returns MergeOutput with expected field types."""
        result = await session_merge(
            session_id="field-test",
            source_branch="branch-src",
            target_branch="branch-tgt",
            strategy="latest",
        )

        # Verify MergeOutput field types
        assert isinstance(result.merged_thought_id, str)
        assert isinstance(result.source_branch_ids, list)
        assert isinstance(result.session_id, str)
        assert isinstance(result.success, bool)

        # Verify source_branch_ids contains strings
        for branch_id in result.source_branch_ids:
            assert isinstance(branch_id, str)

    @pytest.mark.asyncio
    async def test_session_merge_different_strategies(self):
        """Test session_merge with all documented merge strategies."""
        strategies = ["latest", "highest_confidence", "synthesis", "sequential"]

        for strategy in strategies:
            result = await session_merge(
                session_id="multi-strategy-test",
                source_branch="source",
                target_branch="target",
                strategy=strategy,
            )
            assert isinstance(result, MergeOutput)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_session_merge_source_branch_ids_populated(self):
        """Test that session_merge populates source_branch_ids."""
        result = await session_merge(
            session_id="session-xyz",
            source_branch="branch-one",
            target_branch="branch-two",
        )

        # Verify source_branch_ids is populated
        assert isinstance(result.source_branch_ids, list)
        assert len(result.source_branch_ids) >= 2

        # In placeholder implementation, both source and target are included
        assert "branch-one" in result.source_branch_ids
        assert "branch-two" in result.source_branch_ids


# ============================================================================
# Integration Tests
# ============================================================================


class TestSessionToolsIntegration:
    """Integration tests for session management tools."""

    @pytest.mark.asyncio
    async def test_all_session_tools_are_async(self):
        """Test that all session tools are async functions."""
        import inspect

        assert inspect.iscoroutinefunction(session_continue)
        assert inspect.iscoroutinefunction(session_branch)
        assert inspect.iscoroutinefunction(session_inspect)
        assert inspect.iscoroutinefunction(session_merge)

    @pytest.mark.asyncio
    async def test_session_workflow_integration(self):
        """Test a typical session workflow using multiple tools."""
        session_id = "integration-test-session"

        # 1. Inspect initial state
        state = await session_inspect(session_id)
        assert isinstance(state, SessionState)
        assert state.session_id == session_id

        # 2. Continue reasoning
        thought = await session_continue(
            session_id=session_id,
            guidance="Begin analysis",
        )
        assert isinstance(thought, ThoughtOutput)

        # 3. Create a branch
        branch = await session_branch(
            session_id=session_id,
            branch_name="alternative",
        )
        assert isinstance(branch, BranchOutput)
        assert branch.session_id == session_id

        # 4. Merge branches
        merge = await session_merge(
            session_id=session_id,
            source_branch="alternative",
            target_branch="main",
            strategy="synthesis",
        )
        assert isinstance(merge, MergeOutput)
        assert merge.session_id == session_id

    @pytest.mark.asyncio
    async def test_return_types_are_frozen(self):
        """Test that all return types are frozen (immutable)."""
        from pydantic import ValidationError

        # Test ThoughtOutput is frozen
        thought = await session_continue(session_id="test")
        with pytest.raises(ValidationError):
            thought.content = "modified"  # type: ignore[misc]

        # Test BranchOutput is frozen
        branch = await session_branch(session_id="test", branch_name="test")
        with pytest.raises(ValidationError):
            branch.success = False  # type: ignore[misc]

        # Test SessionState is frozen
        state = await session_inspect(session_id="test")
        with pytest.raises(ValidationError):
            state.thought_count = 999  # type: ignore[misc]

        # Test MergeOutput is frozen
        merge = await session_merge(
            session_id="test",
            source_branch="a",
            target_branch="b",
        )
        with pytest.raises(ValidationError):
            merge.success = False  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_consistent_session_id_usage(self):
        """Test that session_id is consistently used across tools."""
        session_id = "consistency-test"

        # All tools should accept and return the same session_id
        thought = await session_continue(session_id=session_id)
        assert session_id in thought.content  # Placeholder includes session_id

        branch = await session_branch(session_id=session_id, branch_name="test")
        assert branch.session_id == session_id

        state = await session_inspect(session_id=session_id)
        assert state.session_id == session_id

        merge = await session_merge(
            session_id=session_id,
            source_branch="a",
            target_branch="b",
        )
        assert merge.session_id == session_id
