"""Session management tools for reasoning-mcp.

This module provides MCP tool functions for managing reasoning sessions,
including continuing reasoning, branching, inspection, merging branches,
listing sessions, getting session details, and deleting sessions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import (
    BranchOutput,
    MergeOutput,
    SessionDeleteOutput,
    SessionGetOutput,
    SessionListOutput,
    SessionState,
    SessionSummary,
    ThoughtOutput,
)

if TYPE_CHECKING:
    from reasoning_mcp.methods.base import ReasoningMethod
    from reasoning_mcp.models.session import Session
    from reasoning_mcp.registry import MethodRegistry
    from reasoning_mcp.sessions import SessionManager

logger = logging.getLogger(__name__)


def _get_session_manager() -> SessionManager:
    """Get the session manager from AppContext.

    Returns:
        SessionManager instance from the application context

    Raises:
        RuntimeError: If AppContext is not initialized
    """
    from reasoning_mcp.server import get_app_context

    return get_app_context().session_manager


def _get_method_registry() -> MethodRegistry:
    """Get the method registry from AppContext.

    Returns:
        MethodRegistry instance from the application context

    Raises:
        RuntimeError: If AppContext is not initialized
    """
    from reasoning_mcp.server import get_app_context

    return get_app_context().registry


def _get_reasoning_method(method_id: MethodIdentifier | str) -> ReasoningMethod | None:
    """Get a reasoning method from the registry.

    Args:
        method_id: The method identifier to look up

    Returns:
        ReasoningMethod instance if found, None otherwise
    """
    registry = _get_method_registry()
    return registry.get(str(method_id))


def _session_to_summary(session: Session) -> SessionSummary:
    """Convert a Session to a SessionSummary.

    Args:
        session: The Session object to convert

    Returns:
        SessionSummary with essential session metadata
    """
    return SessionSummary(
        session_id=session.id,
        status=session.status,
        thought_count=session.thought_count,
        branch_count=session.graph.branch_count,
        current_method=session.current_method,
        created_at=session.created_at,
        updated_at=session.metrics.last_updated if session.metrics else None,
    )


def _thought_to_output(thought: ThoughtNode) -> ThoughtOutput:
    """Convert a ThoughtNode to a ThoughtOutput.

    Args:
        thought: The ThoughtNode to convert

    Returns:
        ThoughtOutput with essential thought data
    """
    return ThoughtOutput(
        id=thought.id,
        content=thought.content,
        thought_type=thought.type,
        confidence=thought.confidence,
        step_number=thought.step_number,
    )


async def session_continue(
    session_id: str,
    guidance: str | None = None,
) -> ThoughtOutput:
    """Continue reasoning in an existing session.

    Generates the next thought in a reasoning session, optionally guided by
    provided hints or directions. This is the primary tool for incremental
    reasoning progression.

    Args:
        session_id: UUID of the session to continue
        guidance: Optional guidance or direction for the next reasoning step

    Returns:
        ThoughtOutput containing the newly generated thought

    Examples:
        Continue without guidance:
        >>> output = await session_continue("session-123")
        >>> assert output.id is not None
        >>> assert output.content != ""

        Continue with guidance:
        >>> output = await session_continue(
        ...     "session-123",
        ...     guidance="Focus on ethical implications"
        ... )
        >>> assert "ethical" in output.content.lower()

    Raises:
        ValueError: If session_id is invalid or session is not active
        RuntimeError: If reasoning engine encounters an error
    """
    manager = _get_session_manager()
    session = await manager.get(session_id)

    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    if session.is_complete:
        raise ValueError(f"Session is already complete: {session_id}")

    # Validate session is in a valid state for continuation
    if session.status not in (SessionStatus.ACTIVE, SessionStatus.CREATED):
        raise ValueError(
            f"Session is not active (status: {session.status}). "
            f"Resume the session before continuing."
        )

    # Determine the method to use (continue with current method or default)
    method_id = session.current_method or MethodIdentifier.CHAIN_OF_THOUGHT

    # Get the reasoning method from the registry
    method = _get_reasoning_method(method_id)

    # Get previous thought if session has any thoughts
    previous_thought: ThoughtNode | None = None
    recent_thoughts = session.get_recent_thoughts(n=1)
    if recent_thoughts:
        previous_thought = recent_thoughts[0]

    # Try to use the actual reasoning method if available
    if method is not None and previous_thought is not None:
        try:
            # Use the method's continue_reasoning implementation
            thought = await method.continue_reasoning(
                session,
                previous_thought,
                guidance=guidance,
                context={"session_id": session_id},
            )
            logger.debug(
                "Continued reasoning using method %s for session %s",
                method_id,
                session_id,
            )
        except (AttributeError, NotImplementedError) as method_error:
            # Method doesn't support continue_reasoning - fall back to default
            logger.warning(
                "Method %s does not support continue_reasoning: %s. Using fallback.",
                method_id,
                method_error,
            )
            thought = _create_fallback_thought(
                session, method_id, guidance, previous_thought
            )
    elif method is not None and previous_thought is None:
        # Session has no previous thoughts - execute as initial reasoning
        try:
            input_text = guidance or "Continue reasoning"
            thought = await method.execute(
                session,
                input_text,
                context={"session_id": session_id},
            )
            logger.debug(
                "Executed initial reasoning using method %s for session %s",
                method_id,
                session_id,
            )
        except (AttributeError, NotImplementedError) as method_error:
            # Method execution failed - fall back to default
            logger.warning(
                "Method %s execution failed: %s. Using fallback.",
                method_id,
                method_error,
            )
            thought = _create_fallback_thought(
                session, method_id, guidance, previous_thought
            )
    else:
        # No method available - use fallback implementation
        logger.debug(
            "Method %s not found in registry. Using fallback implementation.",
            method_id,
        )
        thought = _create_fallback_thought(
            session, method_id, guidance, previous_thought
        )

    # Update session in manager
    await manager.update(session_id, session)

    return _thought_to_output(thought)


def _create_fallback_thought(
    session: Session,
    method_id: MethodIdentifier,
    guidance: str | None,
    previous_thought: ThoughtNode | None,
) -> ThoughtNode:
    """Create a fallback thought when reasoning method is unavailable.

    This function creates a placeholder thought node when the actual reasoning
    method is not available or doesn't support continuation.

    Args:
        session: The current reasoning session
        method_id: The method identifier to use
        guidance: Optional guidance for the thought
        previous_thought: The previous thought to continue from (if any)

    Returns:
        A new ThoughtNode with fallback content
    """
    # Calculate step and depth
    current_step = session.thought_count + 1
    if previous_thought is not None:
        current_depth = previous_thought.depth + 1
        parent_id = previous_thought.id
    else:
        current_depth = 0
        parent_id = None

    # Generate content
    content = f"Continuing reasoning for session {session.id}."
    if guidance:
        content = f"Guided continuation: {guidance}"

    thought = ThoughtNode(
        id=str(uuid4()),
        type=ThoughtType.CONTINUATION if previous_thought else ThoughtType.INITIAL,
        method_id=method_id,
        content=content,
        confidence=0.7,
        step_number=current_step,
        depth=current_depth,
        parent_id=parent_id,
        created_at=datetime.now(),
        metadata={
            "guidance": guidance,
            "session_id": session.id,
            "fallback": True,
        },
    )

    # Add thought to session
    session.add_thought(thought)

    return thought


async def session_branch(
    session_id: str,
    branch_name: str,
    from_thought_id: str | None = None,
) -> BranchOutput:
    """Create a new branch in the reasoning session.

    Creates a new reasoning branch, allowing exploration of alternative paths
    or parallel reasoning strategies. Branches can originate from any thought
    in the session.

    Args:
        session_id: UUID of the session to branch
        branch_name: Human-readable name for the branch
        from_thought_id: Optional ID of the thought to branch from.
            If None, branches from the current head.

    Returns:
        BranchOutput containing information about the created branch

    Examples:
        Create branch from current head:
        >>> output = await session_branch(
        ...     "session-123",
        ...     "alternative-approach"
        ... )
        >>> assert output.success is True
        >>> assert output.branch_id != ""

        Branch from specific thought:
        >>> output = await session_branch(
        ...     "session-123",
        ...     "explore-ethics",
        ...     from_thought_id="thought-456"
        ... )
        >>> assert output.parent_thought_id == "thought-456"

    Raises:
        ValueError: If session_id or from_thought_id is invalid
        RuntimeError: If branching is disabled or max branches exceeded
    """
    manager = _get_session_manager()
    session = await manager.get(session_id)

    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    if not session.config.enable_branching:
        raise RuntimeError(f"Branching is disabled for session: {session_id}")

    if session.graph.branch_count >= session.config.max_branches:
        raise RuntimeError(
            f"Maximum branches ({session.config.max_branches}) exceeded for session: {session_id}"
        )

    # Determine the parent thought
    if from_thought_id is not None:
        parent_thought = session.graph.get_node(from_thought_id)
        if parent_thought is None:
            raise ValueError(f"Thought not found: {from_thought_id}")
        parent_id = from_thought_id
    else:
        # Use the most recent thought as parent
        recent_thoughts = session.get_recent_thoughts(n=1)
        if recent_thoughts:
            parent_id = recent_thoughts[0].id
        else:
            parent_id = session.graph.root_id or "root"

    # Generate branch ID
    branch_id = f"branch-{branch_name}-{uuid4().hex[:8]}"

    # Create a branch thought
    method_id = session.current_method or MethodIdentifier.CHAIN_OF_THOUGHT
    branch_thought = ThoughtNode(
        id=str(uuid4()),
        type=ThoughtType.BRANCH,
        method_id=method_id,
        content=f"Branch '{branch_name}' created from thought {parent_id}",
        confidence=0.7,
        step_number=session.thought_count + 1,
        depth=session.current_depth + 1,
        parent_id=parent_id,
        branch_id=branch_id,
        created_at=datetime.now(),
        metadata={
            "branch_name": branch_name,
            "parent_thought_id": parent_id,
        },
    )

    # Add branch thought to session
    session.add_thought(branch_thought)

    # Update session's active branch
    session.active_branch_id = branch_id

    # Update metrics
    session.metrics.branches_created += 1

    # Update session in manager
    await manager.update(session_id, session)

    return BranchOutput(
        branch_id=branch_id,
        parent_thought_id=parent_id,
        session_id=session_id,
        success=True,
    )


async def session_inspect(
    session_id: str,
    include_graph: bool = False,
) -> SessionState:
    """Inspect the current state of a reasoning session.

    Retrieves comprehensive information about a session's current status,
    including thought counts, metrics, and optionally a graph visualization.

    Args:
        session_id: UUID of the session to inspect
        include_graph: If True, includes graph visualization data in metadata.
            Default is False to reduce payload size.

    Returns:
        SessionState containing session status and metrics

    Examples:
        Basic inspection:
        >>> state = await session_inspect("session-123")
        >>> assert state.session_id == "session-123"
        >>> assert state.thought_count >= 0
        >>> assert state.branch_count >= 0

        Inspection with graph:
        >>> state = await session_inspect(
        ...     "session-123",
        ...     include_graph=True
        ... )
        >>> # Graph data would be in state metadata if implemented

    Raises:
        ValueError: If session_id is invalid or session does not exist
    """
    manager = _get_session_manager()
    session = await manager.get(session_id)

    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    return SessionState(
        session_id=session.id,
        status=session.status,
        thought_count=session.thought_count,
        branch_count=session.graph.branch_count,
        current_method=session.current_method,
        started_at=session.started_at,
        updated_at=session.metrics.last_updated if session.metrics else None,
    )


async def session_merge(
    session_id: str,
    source_branch: str,
    target_branch: str,
    strategy: str = "latest",
) -> MergeOutput:
    """Merge branches in a reasoning session.

    Combines insights from multiple branches back into a unified reasoning path.
    Different merge strategies determine how conflicts and divergent thoughts
    are reconciled.

    Args:
        session_id: UUID of the session containing the branches
        source_branch: ID or name of the branch to merge from
        target_branch: ID or name of the branch to merge into
        strategy: Merge strategy to use. Options:
            - "latest": Use the most recent thought from each branch
            - "highest_confidence": Prefer thoughts with higher confidence
            - "synthesis": Create a new thought synthesizing both branches
            - "sequential": Append source branch thoughts after target
            Default is "latest".

    Returns:
        MergeOutput containing the result of the merge operation

    Examples:
        Simple merge using latest strategy:
        >>> output = await session_merge(
        ...     "session-123",
        ...     "branch-alt",
        ...     "main"
        ... )
        >>> assert output.success is True
        >>> assert output.merged_thought_id != ""

        Merge with synthesis strategy:
        >>> output = await session_merge(
        ...     "session-123",
        ...     "branch-ethical",
        ...     "branch-practical",
        ...     strategy="synthesis"
        ... )
        >>> assert len(output.source_branch_ids) >= 2

    Raises:
        ValueError: If session_id, branch IDs, or strategy is invalid
        RuntimeError: If branches cannot be merged (e.g., conflicting states)
    """
    valid_strategies = {"latest", "highest_confidence", "synthesis", "sequential"}
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid merge strategy: {strategy}. Must be one of {valid_strategies}")

    manager = _get_session_manager()
    session = await manager.get(session_id)

    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    # Get thoughts from source and target branches
    source_thoughts = session.graph.get_branch(source_branch)
    target_thoughts = session.graph.get_branch(target_branch)

    if not source_thoughts and not target_thoughts:
        # If no branch-specific thoughts found, try to find by branch name pattern
        # This handles cases where branch_id might be a name rather than full ID
        all_thoughts = list(session.graph.nodes.values())
        source_thoughts = [t for t in all_thoughts if t.branch_id and source_branch in t.branch_id]
        target_thoughts = [t for t in all_thoughts if t.branch_id and target_branch in t.branch_id]

    # Create merged thought based on strategy
    method_id = session.current_method or MethodIdentifier.CHAIN_OF_THOUGHT

    if strategy == "latest":
        # Use the most recent thought from source branch
        if source_thoughts:
            source_latest = max(source_thoughts, key=lambda t: t.created_at)
            merge_content = f"Merged from {source_branch}: {source_latest.content}"
            confidence = source_latest.confidence
        else:
            merge_content = f"Merged branches {source_branch} and {target_branch}"
            confidence = 0.7
    elif strategy == "highest_confidence":
        # Use the highest confidence thought
        all_branch_thoughts = source_thoughts + target_thoughts
        if all_branch_thoughts:
            best_thought = max(all_branch_thoughts, key=lambda t: t.confidence)
            merge_content = f"Best insight: {best_thought.content}"
            confidence = best_thought.confidence
        else:
            merge_content = f"Merged branches {source_branch} and {target_branch}"
            confidence = 0.7
    elif strategy == "synthesis":
        # Create a synthesis of both branches
        source_summary = (
            source_thoughts[-1].content[:100] if source_thoughts else "No source thoughts"
        )
        target_summary = (
            target_thoughts[-1].content[:100] if target_thoughts else "No target thoughts"
        )
        merge_content = f"Synthesis of {source_branch} ({source_summary}...) and {target_branch} ({target_summary}...)"
        confidence = 0.8
    else:  # sequential
        # Append source after target
        merge_content = f"Sequential merge: {source_branch} appended to {target_branch}"
        confidence = 0.7

    merged_thought = ThoughtNode(
        id=str(uuid4()),
        type=ThoughtType.SYNTHESIS,
        method_id=method_id,
        content=merge_content,
        confidence=confidence,
        step_number=session.thought_count + 1,
        depth=session.current_depth + 1,
        created_at=datetime.now(),
        metadata={
            "merge_strategy": strategy,
            "source_branch": source_branch,
            "target_branch": target_branch,
            "source_thought_count": len(source_thoughts),
            "target_thought_count": len(target_thoughts),
        },
    )

    # Add merged thought to session
    session.add_thought(merged_thought)

    # Update metrics
    session.metrics.branches_merged += 1

    # Update session in manager
    await manager.update(session_id, session)

    return MergeOutput(
        merged_thought_id=merged_thought.id,
        source_branch_ids=[source_branch, target_branch],
        session_id=session_id,
        success=True,
    )


async def session_list(
    limit: int = 100,
    offset: int = 0,
    status: SessionStatus | None = None,
) -> SessionListOutput:
    """List all active reasoning sessions.

    Returns a paginated list of sessions with basic metadata.
    Sessions can be filtered by status.

    Args:
        limit: Maximum number of sessions to return (default: 100)
        offset: Number of sessions to skip for pagination (default: 0)
        status: Optional status filter to only return sessions with this status

    Returns:
        SessionListOutput containing list of session summaries and pagination info

    Examples:
        List all sessions:
        >>> output = await session_list()
        >>> assert output.total >= 0
        >>> assert len(output.sessions) <= output.limit

        List with pagination:
        >>> output = await session_list(limit=10, offset=20)
        >>> assert output.limit == 10
        >>> assert output.offset == 20

        List only active sessions:
        >>> output = await session_list(status=SessionStatus.ACTIVE)
        >>> for session in output.sessions:
        ...     assert session.status == SessionStatus.ACTIVE

    Raises:
        ValueError: If limit or offset is invalid
    """
    if limit < 1:
        raise ValueError("Limit must be at least 1")
    if offset < 0:
        raise ValueError("Offset must be non-negative")

    manager = _get_session_manager()

    # Get sessions with optional status filter
    sessions = await manager.list_sessions(status=status, limit=limit + offset)

    # Apply offset
    paginated_sessions = sessions[offset : offset + limit]

    # Convert to summaries
    summaries = [_session_to_summary(s) for s in paginated_sessions]

    return SessionListOutput(
        sessions=summaries,
        total=len(sessions),
        limit=limit,
        offset=offset,
    )


async def session_get(
    session_id: str,
    include_recent_thoughts: bool = True,
    recent_thoughts_count: int = 5,
) -> SessionGetOutput:
    """Get detailed information about a specific session.

    Retrieves comprehensive information about a session including its
    configuration, metrics, and optionally recent thoughts.

    Args:
        session_id: UUID of the session to retrieve
        include_recent_thoughts: Whether to include recent thoughts (default: True)
        recent_thoughts_count: Number of recent thoughts to include (default: 5)

    Returns:
        SessionGetOutput containing detailed session information

    Examples:
        Get session details:
        >>> output = await session_get("session-123")
        >>> assert output.session_id == "session-123"
        >>> assert output.config is not None

        Get session without recent thoughts:
        >>> output = await session_get("session-123", include_recent_thoughts=False)
        >>> assert len(output.recent_thoughts) == 0

    Raises:
        ValueError: If session_id is invalid or session does not exist
    """
    manager = _get_session_manager()
    session = await manager.get(session_id)

    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    # Get recent thoughts if requested
    recent_thoughts: list[ThoughtOutput] = []
    if include_recent_thoughts:
        thoughts = session.get_recent_thoughts(n=recent_thoughts_count)
        recent_thoughts = [_thought_to_output(t) for t in thoughts]

    # Convert config to dict
    config_dict = session.config.model_dump(mode="json")

    # Convert metrics to dict
    metrics_dict = session.metrics.model_dump(mode="json") if session.metrics else {}

    return SessionGetOutput(
        session_id=session.id,
        status=session.status,
        thought_count=session.thought_count,
        branch_count=session.graph.branch_count,
        current_method=session.current_method,
        active_branch_id=session.active_branch_id,
        created_at=session.created_at,
        started_at=session.started_at,
        completed_at=session.completed_at,
        error=session.error,
        config=config_dict,
        metrics=metrics_dict,
        recent_thoughts=recent_thoughts,
    )


async def session_delete(
    session_id: str,
) -> SessionDeleteOutput:
    """Delete a reasoning session.

    Permanently removes a session and all its associated data including
    thoughts, branches, and metrics.

    Args:
        session_id: UUID of the session to delete

    Returns:
        SessionDeleteOutput indicating whether deletion was successful

    Examples:
        Delete a session:
        >>> output = await session_delete("session-123")
        >>> if output.deleted:
        ...     print("Session deleted successfully")
        ... else:
        ...     print(f"Failed: {output.message}")

    Raises:
        RuntimeError: If deletion fails due to system error
    """
    manager = _get_session_manager()

    # Check if session exists first
    session = await manager.get(session_id)
    if session is None:
        return SessionDeleteOutput(
            session_id=session_id,
            deleted=False,
            message=f"Session not found: {session_id}",
        )

    # Attempt to delete
    deleted = await manager.delete(session_id)

    if deleted:
        return SessionDeleteOutput(
            session_id=session_id,
            deleted=True,
            message="Session deleted successfully",
        )
    else:
        return SessionDeleteOutput(
            session_id=session_id,
            deleted=False,
            message="Failed to delete session",
        )


__all__ = [
    "session_continue",
    "session_branch",
    "session_inspect",
    "session_merge",
    "session_list",
    "session_get",
    "session_delete",
]
