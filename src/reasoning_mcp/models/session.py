"""Session models for reasoning-mcp.

This module defines session-level models for managing reasoning sessions,
including configuration, metrics, and session state management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.thought import ThoughtGraph, ThoughtNode


class SessionConfig(BaseModel):
    """Immutable configuration for a reasoning session.

    SessionConfig defines the parameters and constraints that govern how a reasoning
    session operates, including depth limits, branching behavior, and auto-pruning.
    Configs are frozen to ensure they remain constant throughout a session's lifecycle.

    Examples:
        Create a default config:
        >>> config = SessionConfig()
        >>> assert config.max_depth == 10
        >>> assert config.enable_branching is True

        Create a custom config:
        >>> config = SessionConfig(
        ...     max_depth=20,
        ...     max_thoughts=500,
        ...     timeout_seconds=600.0,
        ...     enable_branching=True,
        ...     max_branches=10,
        ...     auto_prune=True,
        ...     min_confidence_threshold=0.5
        ... )
        >>> assert config.max_depth == 20
        >>> assert config.auto_prune is True

        Add custom metadata:
        >>> config = SessionConfig(
        ...     metadata={"project": "research", "priority": "high"}
        ... )
        >>> assert config.metadata["project"] == "research"
    """

    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reasoning depth allowed (1-100)",
    )
    max_thoughts: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of thoughts allowed in the session (1-1000)",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Session timeout in seconds (must be positive)",
    )
    enable_branching: bool = Field(
        default=True,
        description="Whether to allow branching in the reasoning process",
    )
    max_branches: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of parallel branches allowed (1-20)",
    )
    auto_prune: bool = Field(
        default=False,
        description="Whether to automatically prune low confidence branches",
    )
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for keeping branches (0.0-1.0)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration metadata",
    )


class SessionMetrics(BaseModel):
    """Mutable runtime metrics for a reasoning session.

    SessionMetrics tracks various statistics about the session as it progresses,
    including thought counts, branch operations, quality metrics, and timing information.
    Unlike SessionConfig, metrics are mutable and updated throughout the session.

    Examples:
        Create new metrics:
        >>> metrics = SessionMetrics()
        >>> assert metrics.total_thoughts == 0
        >>> assert metrics.average_confidence == 0.0

        Update from a thought:
        >>> thought = ThoughtNode(
        ...     id="test",
        ...     type=ThoughtType.INITIAL,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="First thought",
        ...     confidence=0.8,
        ...     quality_score=0.9,
        ...     depth=1
        ... )
        >>> metrics.update_from_thought(thought)
        >>> assert metrics.total_thoughts == 1
        >>> assert metrics.max_depth_reached == 1
        >>> assert metrics.average_confidence == 0.8
        >>> assert metrics.methods_used[MethodIdentifier.CHAIN_OF_THOUGHT] == 1
        >>> assert metrics.thought_types[ThoughtType.INITIAL] == 1

        Track multiple thoughts:
        >>> thought2 = ThoughtNode(
        ...     id="test2",
        ...     type=ThoughtType.CONTINUATION,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="Second thought",
        ...     confidence=0.6,
        ...     quality_score=0.7,
        ...     depth=2
        ... )
        >>> metrics.update_from_thought(thought2)
        >>> assert metrics.total_thoughts == 2
        >>> assert metrics.max_depth_reached == 2
        >>> assert metrics.average_confidence == 0.7  # (0.8 + 0.6) / 2
    """

    model_config = ConfigDict(frozen=False)

    total_thoughts: int = Field(
        default=0,
        ge=0,
        description="Total number of thoughts generated in this session",
    )
    total_edges: int = Field(
        default=0,
        ge=0,
        description="Total number of edges created in the thought graph",
    )
    branches_created: int = Field(
        default=0,
        ge=0,
        description="Number of branches created during the session",
    )
    branches_merged: int = Field(
        default=0,
        ge=0,
        description="Number of branches that have been merged",
    )
    branches_pruned: int = Field(
        default=0,
        ge=0,
        description="Number of branches that have been pruned",
    )
    max_depth_reached: int = Field(
        default=0,
        ge=0,
        description="Maximum depth reached during the session",
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score across all thoughts (0.0-1.0)",
    )
    average_quality: float | None = Field(
        default=None,
        description="Average quality score across all thoughts with quality scores",
    )
    methods_used: dict[str, int] = Field(
        default_factory=dict,
        description="Count of thoughts by reasoning method",
    )
    thought_types: dict[str, int] = Field(
        default_factory=dict,
        description="Count of thoughts by thought type",
    )
    elapsed_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Time elapsed in the session (seconds)",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the last metrics update",
    )
    elicitations_made: int = Field(
        default=0,
        ge=0,
        description="Number of elicitations (user interactions) made during the session",
    )

    def update_from_thought(self, thought: ThoughtNode) -> None:
        """Update metrics based on a new thought.

        This method incrementally updates all relevant metrics when a new thought
        is added to the session. It updates counts, averages, and tracking dictionaries.

        Args:
            thought: The ThoughtNode to incorporate into the metrics

        Examples:
            >>> metrics = SessionMetrics()
            >>> thought = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            ...     content="Exploring options",
            ...     confidence=0.75,
            ...     quality_score=0.8,
            ...     depth=3
            ... )
            >>> metrics.update_from_thought(thought)
            >>> assert metrics.total_thoughts == 1
            >>> assert metrics.max_depth_reached == 3
            >>> assert metrics.average_confidence == 0.75
            >>> assert metrics.average_quality == 0.8
            >>> assert metrics.methods_used[MethodIdentifier.TREE_OF_THOUGHTS] == 1
            >>> assert metrics.thought_types[ThoughtType.INITIAL] == 1

            Update with another thought:
            >>> thought2 = ThoughtNode(
            ...     id="t2",
            ...     type=ThoughtType.BRANCH,
            ...     method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            ...     content="Branch exploration",
            ...     confidence=0.85,
            ...     depth=4,
            ...     branch_id="branch1"
            ... )
            >>> metrics.update_from_thought(thought2)
            >>> assert metrics.total_thoughts == 2
            >>> assert metrics.max_depth_reached == 4
            >>> assert metrics.average_confidence == 0.8  # (0.75 + 0.85) / 2
            >>> assert metrics.methods_used[MethodIdentifier.TREE_OF_THOUGHTS] == 2
            >>> assert metrics.thought_types[ThoughtType.BRANCH] == 1
        """
        # Update total count
        self.total_thoughts += 1

        # Update max depth
        if thought.depth > self.max_depth_reached:
            self.max_depth_reached = thought.depth

        # Update average confidence (incremental average)
        self.average_confidence = (
            self.average_confidence * (self.total_thoughts - 1) + thought.confidence
        ) / self.total_thoughts

        # Update average quality if quality_score is present
        if thought.quality_score is not None:
            if self.average_quality is None:
                self.average_quality = thought.quality_score
            else:
                # Count thoughts with quality scores
                quality_count = sum(1 for method_count in self.methods_used.values())
                self.average_quality = (
                    self.average_quality * (quality_count - 1) + thought.quality_score
                ) / quality_count

        # Update method usage
        method_key = str(thought.method_id)
        self.methods_used[method_key] = self.methods_used.get(method_key, 0) + 1

        # Update thought type counts
        type_key = str(thought.type)
        self.thought_types[type_key] = self.thought_types.get(type_key, 0) + 1

        # Track branch creation
        if thought.type == ThoughtType.BRANCH and thought.branch_id is not None:
            # Check if this is a new branch
            branch_thoughts = sum(
                1
                for t_type, count in self.thought_types.items()
                if t_type == str(ThoughtType.BRANCH)
            )
            if branch_thoughts <= self.branches_created + 1:
                self.branches_created += 1

        # Update timestamp
        self.last_updated = datetime.now()


class Session(BaseModel):
    """A mutable reasoning session containing thoughts and state.

    Session represents a complete reasoning session, managing the thought graph,
    configuration, metrics, and lifecycle state. Unlike the frozen config, sessions
    are mutable to allow state transitions and updates during execution.

    Examples:
        Create a new session with defaults:
        >>> session = Session()
        >>> assert session.status == SessionStatus.CREATED
        >>> assert session.is_active is False
        >>> assert session.thought_count == 0

        Create with custom config:
        >>> config = SessionConfig(max_depth=20, timeout_seconds=600.0)
        >>> session = Session(config=config)
        >>> assert session.config.max_depth == 20

        Session lifecycle:
        >>> session = Session()
        >>> session.start()
        >>> assert session.status == SessionStatus.ACTIVE
        >>> assert session.is_active is True
        >>> assert session.started_at is not None
        >>>
        >>> session.pause()
        >>> assert session.status == SessionStatus.PAUSED
        >>>
        >>> session.resume()
        >>> assert session.status == SessionStatus.ACTIVE
        >>>
        >>> session.complete()
        >>> assert session.status == SessionStatus.COMPLETED
        >>> assert session.is_complete is True
        >>> assert session.completed_at is not None

        Adding thoughts:
        >>> session = Session().start()
        >>> thought = ThoughtNode(
        ...     id="root",
        ...     type=ThoughtType.INITIAL,
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     content="Starting analysis",
        ...     confidence=0.8
        ... )
        >>> session.add_thought(thought)
        >>> assert session.thought_count == 1
        >>> assert session.current_depth == 0
        >>> assert session.metrics.total_thoughts == 1

        Querying thoughts:
        >>> thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)
        >>> assert len(thoughts) == 1
        >>> recent = session.get_recent_thoughts(n=5)
        >>> assert len(recent) == 1

        Error handling:
        >>> session = Session().start()
        >>> session.fail("Something went wrong")
        >>> assert session.status == SessionStatus.FAILED
        >>> assert session.error == "Something went wrong"
        >>> assert session.is_complete is True
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this session (UUID)",
    )
    config: SessionConfig = Field(
        default_factory=SessionConfig,
        description="Session configuration",
    )
    status: SessionStatus = Field(
        default=SessionStatus.CREATED,
        description="Current status of the session",
    )
    graph: ThoughtGraph = Field(
        default_factory=ThoughtGraph,
        description="The thought graph containing all thoughts and edges",
    )
    metrics: SessionMetrics = Field(
        default_factory=SessionMetrics,
        description="Runtime metrics for this session",
    )
    current_method: MethodIdentifier | None = Field(
        default=None,
        description="Currently active reasoning method",
    )
    active_branch_id: str | None = Field(
        default=None,
        description="ID of the currently active branch",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the session was created",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Timestamp when the session was started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the session was completed",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the session failed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary session metadata",
    )

    @property
    def is_active(self) -> bool:
        """Check if the session is currently active.

        Returns:
            True if status is ACTIVE, False otherwise

        Examples:
            >>> session = Session()
            >>> assert session.is_active is False
            >>> session.start()
            >>> assert session.is_active is True
            >>> session.pause()
            >>> assert session.is_active is False
        """
        return self.status == SessionStatus.ACTIVE

    @property
    def is_complete(self) -> bool:
        """Check if the session has completed (successfully, failed, or cancelled).

        Returns:
            True if status is COMPLETED, FAILED, or CANCELLED

        Examples:
            >>> session = Session()
            >>> assert session.is_complete is False
            >>> session.start()
            >>> assert session.is_complete is False
            >>> session.complete()
            >>> assert session.is_complete is True
            >>>
            >>> session2 = Session().start()
            >>> session2.fail("Error")
            >>> assert session2.is_complete is True
            >>>
            >>> session3 = Session().start()
            >>> session3.cancel()
            >>> assert session3.is_complete is True
        """
        return self.status in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
        )

    @property
    def duration(self) -> float | None:
        """Calculate elapsed time since session started.

        Returns:
            Time in seconds since started_at, or None if not started

        Examples:
            >>> session = Session()
            >>> assert session.duration is None
            >>> session.start()
            >>> assert session.duration is not None
            >>> assert session.duration >= 0.0
        """
        if self.started_at is None:
            return None
        end_time = self.completed_at if self.completed_at else datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def thought_count(self) -> int:
        """Get the total number of thoughts in the session.

        Returns:
            Number of nodes in the thought graph

        Examples:
            >>> session = Session().start()
            >>> assert session.thought_count == 0
            >>> thought = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="First"
            ... )
            >>> session.add_thought(thought)
            >>> assert session.thought_count == 1
        """
        return self.graph.node_count

    @property
    def current_depth(self) -> int:
        """Get the maximum depth reached in the thought graph.

        Returns:
            Maximum depth value across all thoughts

        Examples:
            >>> session = Session().start()
            >>> assert session.current_depth == 0
            >>> thought = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="Root",
            ...     depth=0
            ... )
            >>> session.add_thought(thought)
            >>> child = ThoughtNode(
            ...     id="t2",
            ...     type=ThoughtType.CONTINUATION,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="Child",
            ...     parent_id="t1",
            ...     depth=3
            ... )
            >>> session.add_thought(child)
            >>> assert session.current_depth == 3
        """
        return self.graph.max_depth

    def start(self) -> Session:
        """Start the session and transition to ACTIVE status.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If session is already completed or active

        Examples:
            >>> session = Session()
            >>> session.start()
            >>> assert session.status == SessionStatus.ACTIVE
            >>> assert session.started_at is not None
        """
        if self.is_complete:
            raise ValueError("Cannot start a completed session")
        if self.status == SessionStatus.ACTIVE:
            raise ValueError("Session is already active")

        self.status = SessionStatus.ACTIVE
        self.started_at = datetime.now()
        return self

    def pause(self) -> Session:
        """Pause the session and transition to PAUSED status.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If session is not active

        Examples:
            >>> session = Session().start()
            >>> session.pause()
            >>> assert session.status == SessionStatus.PAUSED
        """
        if self.status != SessionStatus.ACTIVE:
            raise ValueError("Can only pause an active session")

        self.status = SessionStatus.PAUSED
        return self

    def resume(self) -> Session:
        """Resume a paused session and transition to ACTIVE status.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If session is not paused

        Examples:
            >>> session = Session().start()
            >>> session.pause()
            >>> session.resume()
            >>> assert session.status == SessionStatus.ACTIVE
        """
        if self.status != SessionStatus.PAUSED:
            raise ValueError("Can only resume a paused session")

        self.status = SessionStatus.ACTIVE
        return self

    def complete(self) -> Session:
        """Complete the session successfully and transition to COMPLETED status.

        Returns:
            Self for method chaining

        Examples:
            >>> session = Session().start()
            >>> session.complete()
            >>> assert session.status == SessionStatus.COMPLETED
            >>> assert session.completed_at is not None
        """
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
        return self

    def fail(self, error: str) -> Session:
        """Fail the session with an error message and transition to FAILED status.

        Args:
            error: Error message describing the failure

        Returns:
            Self for method chaining

        Examples:
            >>> session = Session().start()
            >>> session.fail("Timeout exceeded")
            >>> assert session.status == SessionStatus.FAILED
            >>> assert session.error == "Timeout exceeded"
            >>> assert session.completed_at is not None
        """
        self.status = SessionStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        return self

    def cancel(self) -> Session:
        """Cancel the session and transition to CANCELLED status.

        Returns:
            Self for method chaining

        Examples:
            >>> session = Session().start()
            >>> session.cancel()
            >>> assert session.status == SessionStatus.CANCELLED
            >>> assert session.completed_at is not None
        """
        self.status = SessionStatus.CANCELLED
        self.completed_at = datetime.now()
        return self

    def add_thought(self, thought: ThoughtNode) -> Session:
        """Add a thought to the session graph and update metrics.

        This method adds the thought to the graph and updates session metrics
        accordingly. It also updates the total_edges count in metrics.

        Args:
            thought: The ThoughtNode to add

        Returns:
            Self for method chaining

        Examples:
            >>> session = Session().start()
            >>> thought = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="First thought",
            ...     confidence=0.8
            ... )
            >>> session.add_thought(thought)
            >>> assert session.thought_count == 1
            >>> assert session.metrics.total_thoughts == 1
            >>> assert session.metrics.average_confidence == 0.8

            Add a child thought:
            >>> child = ThoughtNode(
            ...     id="t2",
            ...     type=ThoughtType.CONTINUATION,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="Second thought",
            ...     parent_id="t1",
            ...     confidence=0.9,
            ...     depth=1
            ... )
            >>> session.add_thought(child)
            >>> assert session.thought_count == 2
            >>> assert session.metrics.total_edges == 1  # Edge from parent to child
        """
        # Track edge count before adding

        # Add to graph
        self.graph.add_thought(thought)

        # Update metrics
        self.metrics.update_from_thought(thought)

        # Update edge count in metrics
        edge_count_after = self.graph.edge_count
        self.metrics.total_edges = edge_count_after

        return self

    def get_thoughts_by_method(self, method: MethodIdentifier) -> list[ThoughtNode]:
        """Get all thoughts created by a specific reasoning method.

        Args:
            method: The reasoning method to filter by

        Returns:
            List of ThoughtNodes that were created by the specified method

        Examples:
            >>> session = Session().start()
            >>> cot_thought = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="CoT thought"
            ... )
            >>> tot_thought = ThoughtNode(
            ...     id="t2",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            ...     content="ToT thought"
            ... )
            >>> session.add_thought(cot_thought)
            >>> session.add_thought(tot_thought)
            >>> cot_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)
            >>> assert len(cot_thoughts) == 1
            >>> assert cot_thoughts[0].id == "t1"
        """
        return [node for node in self.graph.nodes.values() if node.method_id == method]

    def get_thoughts_by_type(self, thought_type: ThoughtType) -> list[ThoughtNode]:
        """Get all thoughts of a specific type.

        Args:
            thought_type: The thought type to filter by

        Returns:
            List of ThoughtNodes that match the specified type

        Examples:
            >>> session = Session().start()
            >>> initial = ThoughtNode(
            ...     id="t1",
            ...     type=ThoughtType.INITIAL,
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     content="Initial"
            ... )
            >>> branch = ThoughtNode(
            ...     id="t2",
            ...     type=ThoughtType.BRANCH,
            ...     method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            ...     content="Branch"
            ... )
            >>> session.add_thought(initial)
            >>> session.add_thought(branch)
            >>> initials = session.get_thoughts_by_type(ThoughtType.INITIAL)
            >>> assert len(initials) == 1
            >>> assert initials[0].id == "t1"
            >>> branches = session.get_thoughts_by_type(ThoughtType.BRANCH)
            >>> assert len(branches) == 1
            >>> assert branches[0].id == "t2"
        """
        return [node for node in self.graph.nodes.values() if node.type == thought_type]

    def get_recent_thoughts(self, n: int = 5) -> list[ThoughtNode]:
        """Get the n most recent thoughts by creation time.

        Args:
            n: Number of recent thoughts to retrieve (default: 5)

        Returns:
            List of up to n ThoughtNodes, sorted by created_at descending

        Examples:
            >>> session = Session().start()
            >>> for i in range(10):
            ...     thought = ThoughtNode(
            ...         id=f"t{i}",
            ...         type=ThoughtType.CONTINUATION,
            ...         method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...         content=f"Thought {i}"
            ...     )
            ...     session.add_thought(thought)
            >>> recent = session.get_recent_thoughts(n=3)
            >>> assert len(recent) == 3
            >>> # Most recent thoughts come first
            >>> assert recent[0].id == "t9"
            >>> assert recent[1].id == "t8"
            >>> assert recent[2].id == "t7"
        """
        # Sort by created_at descending and take first n
        sorted_thoughts = sorted(
            self.graph.nodes.values(),
            key=lambda node: node.created_at,
            reverse=True,
        )
        return sorted_thoughts[:n]
