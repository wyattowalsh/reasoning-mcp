"""
Comprehensive tests for Session models in reasoning_mcp.models.session.

This module provides complete test coverage for:
- SessionConfig: Frozen configuration model for session parameters
- SessionMetrics: Mutable metrics tracking model
- Session: Mutable session management model

Each model is tested for:
1. Creation with required and optional fields
2. Default values
3. Mutability/Immutability (frozen vs mutable)
4. Field validation (types, ranges, constraints)
5. Methods and state transitions
6. Computed properties
7. JSON serialization/deserialization
"""

from datetime import datetime

import pytest
from pydantic import BaseModel, ValidationError

from reasoning_mcp.models.core import MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.session import Session, SessionConfig, SessionMetrics
from reasoning_mcp.models.thought import ThoughtNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config() -> SessionConfig:
    """Provide a sample SessionConfig for testing.

    Returns:
        SessionConfig with custom values for testing.
    """
    return SessionConfig(max_depth=5, max_thoughts=50)


@pytest.fixture
def sample_session() -> Session:
    """Provide a sample Session for testing.

    Returns:
        Session with default configuration.
    """
    return Session()


@pytest.fixture
def active_session() -> Session:
    """Provide an active Session for testing.

    Returns:
        Session that has been started (status=ACTIVE).
    """
    session = Session()
    session.start()
    return session


@pytest.fixture
def sample_thought() -> ThoughtNode:
    """Provide a sample ThoughtNode for testing.

    Returns:
        ThoughtNode with minimal required fields.
    """
    return ThoughtNode(
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.SEQUENTIAL_THINKING,
        content="Test thought",
    )


# ============================================================================
# SessionConfig Tests - Creation
# ============================================================================


class TestSessionConfigCreation:
    """Test suite for SessionConfig creation and basic properties."""

    def test_create_with_defaults(self):
        """Test creating a SessionConfig with all default values."""
        config = SessionConfig()

        assert config.max_depth == 10
        assert config.max_thoughts == 100
        assert config.timeout_seconds == 300.0
        assert config.enable_branching is True
        assert config.max_branches == 5
        assert config.auto_prune is False
        assert config.min_confidence_threshold == 0.3
        assert config.metadata == {}

    def test_create_with_custom_values(self):
        """Test creating a SessionConfig with custom values."""
        config = SessionConfig(
            max_depth=15,
            max_thoughts=200,
            timeout_seconds=600.0,
            enable_branching=False,
            max_branches=10,
            auto_prune=True,
            min_confidence_threshold=0.5,
            metadata={"key": "value"},
        )

        assert config.max_depth == 15
        assert config.max_thoughts == 200
        assert config.timeout_seconds == 600.0
        assert config.enable_branching is False
        assert config.max_branches == 10
        assert config.auto_prune is True
        assert config.min_confidence_threshold == 0.5
        assert config.metadata == {"key": "value"}

    def test_config_is_frozen(self):
        """Test that SessionConfig is frozen (immutable)."""
        config = SessionConfig()

        with pytest.raises((ValidationError, AttributeError)):
            config.max_depth = 20

    def test_config_with_metadata(self):
        """Test creating a SessionConfig with custom metadata."""
        metadata = {
            "experiment": "test_run_1",
            "version": "1.0",
            "tags": ["reasoning", "mcp"],
        }
        config = SessionConfig(metadata=metadata)

        assert config.metadata == metadata

    def test_is_pydantic_basemodel(self):
        """Test that SessionConfig is a Pydantic BaseModel."""
        assert issubclass(SessionConfig, BaseModel)


# ============================================================================
# SessionConfig Tests - Validation
# ============================================================================


class TestSessionConfigValidation:
    """Test suite for SessionConfig field validation."""

    def test_max_depth_valid_range(self):
        """Test that max_depth accepts valid range [1, 100]."""
        # Test boundary values
        config_min = SessionConfig(max_depth=1)
        assert config_min.max_depth == 1

        config_max = SessionConfig(max_depth=100)
        assert config_max.max_depth == 100

        # Test mid-range value
        config_mid = SessionConfig(max_depth=50)
        assert config_mid.max_depth == 50

    def test_max_depth_below_minimum(self):
        """Test that max_depth below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_depth=0)

        assert "max_depth" in str(exc_info.value).lower()

    def test_max_depth_above_maximum(self):
        """Test that max_depth above 100 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_depth=101)

        assert "max_depth" in str(exc_info.value).lower()

    def test_max_thoughts_valid_range(self):
        """Test that max_thoughts accepts valid range [1, 1000]."""
        # Test boundary values
        config_min = SessionConfig(max_thoughts=1)
        assert config_min.max_thoughts == 1

        config_max = SessionConfig(max_thoughts=1000)
        assert config_max.max_thoughts == 1000

        # Test mid-range value
        config_mid = SessionConfig(max_thoughts=500)
        assert config_mid.max_thoughts == 500

    def test_max_thoughts_below_minimum(self):
        """Test that max_thoughts below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_thoughts=0)

        assert "max_thoughts" in str(exc_info.value).lower()

    def test_max_thoughts_above_maximum(self):
        """Test that max_thoughts above 1000 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_thoughts=1001)

        assert "max_thoughts" in str(exc_info.value).lower()

    def test_timeout_must_be_positive(self):
        """Test that timeout_seconds must be positive (greater than 0)."""
        # Valid positive values
        config_pos = SessionConfig(timeout_seconds=1.0)
        assert config_pos.timeout_seconds == 1.0

        # Zero should fail
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(timeout_seconds=0.0)
        assert "timeout_seconds" in str(exc_info.value).lower()

        # Negative should fail
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(timeout_seconds=-1.0)
        assert "timeout_seconds" in str(exc_info.value).lower()

    def test_max_branches_valid_range(self):
        """Test that max_branches accepts valid range [1, 20]."""
        # Test boundary values
        config_min = SessionConfig(max_branches=1)
        assert config_min.max_branches == 1

        config_max = SessionConfig(max_branches=20)
        assert config_max.max_branches == 20

        # Test mid-range value
        config_mid = SessionConfig(max_branches=10)
        assert config_mid.max_branches == 10

    def test_max_branches_below_minimum(self):
        """Test that max_branches below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_branches=0)

        assert "max_branches" in str(exc_info.value).lower()

    def test_max_branches_above_maximum(self):
        """Test that max_branches above 20 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_branches=21)

        assert "max_branches" in str(exc_info.value).lower()

    def test_min_confidence_threshold_valid_range(self):
        """Test that min_confidence_threshold accepts valid range [0.0, 1.0]."""
        # Test boundary values
        config_min = SessionConfig(min_confidence_threshold=0.0)
        assert config_min.min_confidence_threshold == 0.0

        config_max = SessionConfig(min_confidence_threshold=1.0)
        assert config_max.min_confidence_threshold == 1.0

        # Test mid-range value
        config_mid = SessionConfig(min_confidence_threshold=0.5)
        assert config_mid.min_confidence_threshold == 0.5

    def test_min_confidence_threshold_below_minimum(self):
        """Test that min_confidence_threshold below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(min_confidence_threshold=-0.1)

        assert "min_confidence_threshold" in str(exc_info.value).lower()

    def test_min_confidence_threshold_above_maximum(self):
        """Test that min_confidence_threshold above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(min_confidence_threshold=1.1)

        assert "min_confidence_threshold" in str(exc_info.value).lower()


# ============================================================================
# SessionMetrics Tests - Creation
# ============================================================================


class TestSessionMetricsCreation:
    """Test suite for SessionMetrics creation and basic properties."""

    def test_create_with_defaults(self):
        """Test creating a SessionMetrics with all default values."""
        metrics = SessionMetrics()

        assert metrics.total_thoughts == 0
        assert metrics.total_edges == 0
        assert metrics.branches_created == 0
        assert metrics.max_depth_reached == 0
        assert metrics.average_confidence == 0.0
        assert metrics.methods_used == {}
        assert metrics.thought_types == {}
        assert isinstance(metrics.last_updated, datetime)  # default_factory=datetime.now

    def test_create_with_custom_values(self):
        """Test creating a SessionMetrics with custom values."""
        now = datetime.now()
        metrics = SessionMetrics(
            total_thoughts=10,
            total_edges=9,
            branches_created=2,
            max_depth_reached=5,
            average_confidence=0.85,
            methods_used={
                MethodIdentifier.CHAIN_OF_THOUGHT: 5,
                MethodIdentifier.TREE_OF_THOUGHTS: 5,
            },
            thought_types={ThoughtType.INITIAL: 1, ThoughtType.CONTINUATION: 9},
            last_updated=now,
        )

        assert metrics.total_thoughts == 10
        assert metrics.total_edges == 9
        assert metrics.branches_created == 2
        assert metrics.max_depth_reached == 5
        assert metrics.average_confidence == 0.85
        assert len(metrics.methods_used) == 2
        assert len(metrics.thought_types) == 2
        assert metrics.last_updated == now

    def test_metrics_is_mutable(self):
        """Test that SessionMetrics is mutable (not frozen)."""
        metrics = SessionMetrics()

        # Should be able to modify fields
        metrics.total_thoughts = 5
        assert metrics.total_thoughts == 5

        metrics.average_confidence = 0.75
        assert metrics.average_confidence == 0.75

    def test_metrics_with_methods_used(self):
        """Test creating SessionMetrics with methods_used tracking."""
        methods_used = {
            MethodIdentifier.SEQUENTIAL_THINKING: 10,
            MethodIdentifier.CHAIN_OF_THOUGHT: 5,
            MethodIdentifier.TREE_OF_THOUGHTS: 3,
        }
        metrics = SessionMetrics(methods_used=methods_used)

        assert metrics.methods_used == methods_used
        assert metrics.methods_used[MethodIdentifier.SEQUENTIAL_THINKING] == 10

    def test_is_pydantic_basemodel(self):
        """Test that SessionMetrics is a Pydantic BaseModel."""
        assert issubclass(SessionMetrics, BaseModel)


# ============================================================================
# SessionMetrics Tests - Update From Thought
# ============================================================================


class TestSessionMetricsUpdateFromThought:
    """Test suite for SessionMetrics update_from_thought method."""

    def test_update_increments_total_thoughts(self, sample_thought: ThoughtNode):
        """Test that update_from_thought increments total_thoughts counter."""
        metrics = SessionMetrics()
        assert metrics.total_thoughts == 0

        metrics.update_from_thought(sample_thought)
        assert metrics.total_thoughts == 1

        metrics.update_from_thought(sample_thought)
        assert metrics.total_thoughts == 2

    def test_update_tracks_method_used(self, sample_thought: ThoughtNode):
        """Test that update_from_thought tracks method usage counts."""
        metrics = SessionMetrics()

        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test 1",
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test 2",
        )
        thought_3 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="Test 3",
        )

        metrics.update_from_thought(thought_1)
        assert metrics.methods_used[MethodIdentifier.CHAIN_OF_THOUGHT] == 1

        metrics.update_from_thought(thought_2)
        assert metrics.methods_used[MethodIdentifier.CHAIN_OF_THOUGHT] == 2

        metrics.update_from_thought(thought_3)
        assert metrics.methods_used[MethodIdentifier.TREE_OF_THOUGHTS] == 1

    def test_update_tracks_thought_type(self, sample_thought: ThoughtNode):
        """Test that update_from_thought tracks thought type counts."""
        metrics = SessionMetrics()

        thought_initial = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Initial",
        )
        thought_continuation = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Continuation",
        )

        metrics.update_from_thought(thought_initial)
        assert metrics.thought_types[ThoughtType.INITIAL] == 1

        metrics.update_from_thought(thought_initial)
        assert metrics.thought_types[ThoughtType.INITIAL] == 2

        metrics.update_from_thought(thought_continuation)
        assert metrics.thought_types[ThoughtType.CONTINUATION] == 1

    def test_update_calculates_average_confidence(self):
        """Test that update_from_thought calculates running average confidence."""
        metrics = SessionMetrics()

        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 1",
            confidence=0.8,
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 2",
            confidence=0.6,
        )
        thought_3 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 3",
            confidence=0.9,
        )

        metrics.update_from_thought(thought_1)
        assert metrics.average_confidence == 0.8

        metrics.update_from_thought(thought_2)
        # Average of 0.8 and 0.6 = 0.7
        assert metrics.average_confidence == pytest.approx(0.7, abs=0.01)

        metrics.update_from_thought(thought_3)
        # Average of 0.8, 0.6, 0.9 = 0.7666...
        assert metrics.average_confidence == pytest.approx(0.7667, abs=0.01)

    def test_update_tracks_max_depth(self):
        """Test that update_from_thought tracks maximum depth reached."""
        metrics = SessionMetrics()

        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 1",
            depth=0,
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 2",
            depth=3,
        )
        thought_3 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 3",
            depth=2,
        )

        metrics.update_from_thought(thought_1)
        assert metrics.max_depth_reached == 0

        metrics.update_from_thought(thought_2)
        assert metrics.max_depth_reached == 3

        metrics.update_from_thought(thought_3)
        assert metrics.max_depth_reached == 3  # Should not decrease

    def test_update_sets_last_updated(self, sample_thought: ThoughtNode):
        """Test that update_from_thought updates the last_updated timestamp."""
        metrics = SessionMetrics()
        initial_last_updated = metrics.last_updated

        before = datetime.now()
        metrics.update_from_thought(sample_thought)
        after = datetime.now()

        # last_updated should be updated to a new timestamp
        assert metrics.last_updated is not None
        assert metrics.last_updated >= initial_last_updated
        assert before <= metrics.last_updated <= after


# ============================================================================
# Session Tests - Creation
# ============================================================================


class TestSessionCreation:
    """Test suite for Session creation and basic properties."""

    def test_create_with_defaults(self):
        """Test creating a Session with all default values."""
        session = Session()

        assert session.id is not None
        assert len(session.id) > 0
        assert isinstance(session.config, SessionConfig)
        assert session.status == SessionStatus.CREATED
        assert session.graph is not None
        assert isinstance(session.metrics, SessionMetrics)
        assert session.current_method is None
        assert session.started_at is None
        assert session.completed_at is None
        assert session.error is None
        assert session.metadata == {}

    def test_create_with_custom_config(self):
        """Test creating a Session with custom configuration."""
        config = SessionConfig(max_depth=15, max_thoughts=200)
        session = Session(config=config)

        assert session.config.max_depth == 15
        assert session.config.max_thoughts == 200

    def test_create_with_custom_id(self):
        """Test creating a Session with custom ID."""
        custom_id = "test-session-123"
        session = Session(id=custom_id)

        assert session.id == custom_id

    def test_session_is_mutable(self):
        """Test that Session is mutable (not frozen)."""
        session = Session()

        # Should be able to modify fields
        session.status = SessionStatus.ACTIVE
        assert session.status == SessionStatus.ACTIVE

        session.current_method = MethodIdentifier.CHAIN_OF_THOUGHT
        assert session.current_method == MethodIdentifier.CHAIN_OF_THOUGHT

    def test_default_status_is_created(self):
        """Test that default status is CREATED."""
        session = Session()

        assert session.status == SessionStatus.CREATED

    def test_is_pydantic_basemodel(self):
        """Test that Session is a Pydantic BaseModel."""
        assert issubclass(Session, BaseModel)


# ============================================================================
# Session Tests - Computed Properties
# ============================================================================


class TestSessionComputedProperties:
    """Test suite for Session computed properties."""

    def test_is_active_when_active(self):
        """Test is_active property returns True for ACTIVE status."""
        session = Session()
        session.start()

        assert session.is_active is True

    def test_is_active_when_not_active(self):
        """Test is_active property returns False for non-ACTIVE statuses."""
        # CREATED status
        session_created = Session()
        assert session_created.is_active is False

        # PAUSED status
        session_paused = Session()
        session_paused.start()
        session_paused.pause()
        assert session_paused.is_active is False

        # COMPLETED status
        session_completed = Session()
        session_completed.start()
        session_completed.complete()
        assert session_completed.is_active is False

    def test_is_complete_when_completed(self):
        """Test is_complete property returns True for COMPLETED status."""
        session = Session()
        session.start()
        session.complete()

        assert session.is_complete is True

    def test_is_complete_when_failed(self):
        """Test is_complete property returns True for FAILED status."""
        session = Session()
        session.start()
        session.fail("Test error")

        assert session.is_complete is True

    def test_is_complete_when_not_complete(self):
        """Test is_complete property returns False for non-terminal statuses."""
        # CREATED status
        session_created = Session()
        assert session_created.is_complete is False

        # ACTIVE status
        session_active = Session()
        session_active.start()
        assert session_active.is_complete is False

        # PAUSED status
        session_paused = Session()
        session_paused.start()
        session_paused.pause()
        assert session_paused.is_complete is False

    def test_duration_when_started(self):
        """Test duration property when session has been started."""
        session = Session()
        session.start()

        duration = session.duration
        assert duration is not None
        assert duration >= 0.0

    def test_duration_when_not_started(self):
        """Test duration property when session has not been started."""
        session = Session()

        assert session.duration is None

    def test_thought_count_delegates_to_graph(self, sample_thought: ThoughtNode):
        """Test thought_count property delegates to graph.node_count."""
        session = Session()

        # Initially empty
        assert session.thought_count == 0

        # Add thoughts to graph
        session.graph.add_thought(sample_thought)
        assert session.thought_count == 1

        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 2",
        )
        session.graph.add_thought(thought_2)
        assert session.thought_count == 2

    def test_current_depth_delegates_to_graph(self):
        """Test current_depth property delegates to graph.max_depth."""
        session = Session()

        # Initially 0
        assert session.current_depth == 0

        # Add thoughts with different depths
        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 1",
            depth=0,
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test 2",
            depth=3,
        )

        session.graph.add_thought(thought_1)
        assert session.current_depth == 0

        session.graph.add_thought(thought_2)
        assert session.current_depth == 3


# ============================================================================
# Session Tests - State Transitions
# ============================================================================


class TestSessionStateTransitions:
    """Test suite for Session state transition methods."""

    def test_start_from_created(self):
        """Test starting a session from CREATED status."""
        session = Session()
        assert session.status == SessionStatus.CREATED

        session.start()
        assert session.status == SessionStatus.ACTIVE

    def test_start_sets_started_at(self):
        """Test that start() sets started_at timestamp."""
        session = Session()
        assert session.started_at is None

        before = datetime.now()
        session.start()
        after = datetime.now()

        assert session.started_at is not None
        assert before <= session.started_at <= after

    def test_pause_from_active(self):
        """Test pausing a session from ACTIVE status."""
        session = Session()
        session.start()
        assert session.status == SessionStatus.ACTIVE

        session.pause()
        assert session.status == SessionStatus.PAUSED

    def test_resume_from_paused(self):
        """Test resuming a session from PAUSED status."""
        session = Session()
        session.start()
        session.pause()
        assert session.status == SessionStatus.PAUSED

        session.resume()
        assert session.status == SessionStatus.ACTIVE

    def test_complete_from_active(self):
        """Test completing a session from ACTIVE status."""
        session = Session()
        session.start()
        assert session.status == SessionStatus.ACTIVE

        session.complete()
        assert session.status == SessionStatus.COMPLETED

    def test_complete_sets_completed_at(self):
        """Test that complete() sets completed_at timestamp."""
        session = Session()
        session.start()
        assert session.completed_at is None

        before = datetime.now()
        session.complete()
        after = datetime.now()

        assert session.completed_at is not None
        assert before <= session.completed_at <= after

    def test_fail_sets_error_message(self):
        """Test that fail() sets error field."""
        session = Session()
        session.start()

        error_msg = "Test error occurred"
        session.fail(error_msg)

        assert session.error == error_msg

    def test_fail_from_active(self):
        """Test failing a session from ACTIVE status."""
        session = Session()
        session.start()
        assert session.status == SessionStatus.ACTIVE

        session.fail("Test error")
        assert session.status == SessionStatus.FAILED

    def test_cancel_from_any_state(self):
        """Test canceling a session from various states."""
        # Cancel from CREATED
        session_created = Session()
        session_created.cancel()
        assert session_created.status == SessionStatus.CANCELLED

        # Cancel from ACTIVE
        session_active = Session()
        session_active.start()
        session_active.cancel()
        assert session_active.status == SessionStatus.CANCELLED

        # Cancel from PAUSED
        session_paused = Session()
        session_paused.start()
        session_paused.pause()
        session_paused.cancel()
        assert session_paused.status == SessionStatus.CANCELLED

    def test_start_returns_self_for_chaining(self):
        """Test that start() returns self for method chaining."""
        session = Session()
        result = session.start()

        assert result is session
        assert isinstance(result, Session)

    def test_all_transitions_return_self(self):
        """Test that all state transition methods return self."""
        session = Session()

        # start returns self
        result = session.start()
        assert result is session

        # pause returns self
        result = session.pause()
        assert result is session

        # resume returns self
        result = session.resume()
        assert result is session

        # complete returns self
        session_2 = Session()
        session_2.start()
        result = session_2.complete()
        assert result is session_2

        # fail returns self
        session_3 = Session()
        session_3.start()
        result = session_3.fail("error")
        assert result is session_3

        # cancel returns self
        session_4 = Session()
        result = session_4.cancel()
        assert result is session_4


# ============================================================================
# Session Tests - Add Thought
# ============================================================================


class TestSessionAddThought:
    """Test suite for Session add_thought method."""

    def test_add_thought_adds_to_graph(self, sample_thought: ThoughtNode):
        """Test that add_thought adds the thought to the graph."""
        session = Session()
        assert session.thought_count == 0

        session.add_thought(sample_thought)

        assert session.thought_count == 1
        assert sample_thought.id in session.graph.nodes

    def test_add_thought_updates_metrics(self, sample_thought: ThoughtNode):
        """Test that add_thought updates session metrics."""
        session = Session()
        assert session.metrics.total_thoughts == 0

        session.add_thought(sample_thought)

        assert session.metrics.total_thoughts == 1
        assert session.metrics.methods_used[MethodIdentifier.SEQUENTIAL_THINKING] == 1
        assert session.metrics.thought_types[ThoughtType.INITIAL] == 1

    def test_add_thought_returns_self(self, sample_thought: ThoughtNode):
        """Test that add_thought returns self for method chaining."""
        session = Session()

        result = session.add_thought(sample_thought)

        assert result is session
        assert isinstance(result, Session)

    def test_add_multiple_thoughts(self):
        """Test adding multiple thoughts in sequence."""
        session = Session()

        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="First thought",
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Second thought",
        )
        thought_3 = ThoughtNode(
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Final thought",
        )

        session.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        assert session.thought_count == 3
        assert session.metrics.total_thoughts == 3

    def test_add_thought_with_parent(self):
        """Test adding a thought with parent_id relationship."""
        session = Session()

        parent = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Parent thought",
        )
        child = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Child thought",
            parent_id=parent.id,
        )

        session.add_thought(parent).add_thought(child)

        assert session.thought_count == 2
        # Parent should have child in its children_ids
        parent_in_graph = session.graph.get_node(parent.id)
        assert parent_in_graph is not None
        assert child.id in parent_in_graph.children_ids


# ============================================================================
# Session Tests - Query Methods
# ============================================================================


class TestSessionQueryMethods:
    """Test suite for Session query methods."""

    def test_get_thoughts_by_method_empty(self):
        """Test get_thoughts_by_method on empty session."""
        session = Session()

        thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert thoughts == []

    def test_get_thoughts_by_method_with_matches(self):
        """Test get_thoughts_by_method returns matching thoughts."""
        session = Session()

        thought_cot_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="COT 1",
        )
        thought_cot_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="COT 2",
        )
        thought_tot = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            content="TOT",
        )

        session.add_thought(thought_cot_1).add_thought(thought_cot_2).add_thought(thought_tot)

        cot_thoughts = session.get_thoughts_by_method(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert len(cot_thoughts) == 2
        assert thought_cot_1 in cot_thoughts
        assert thought_cot_2 in cot_thoughts
        assert thought_tot not in cot_thoughts

    def test_get_thoughts_by_method_no_matches(self):
        """Test get_thoughts_by_method when no thoughts match."""
        session = Session()

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test",
        )
        session.add_thought(thought)

        tot_thoughts = session.get_thoughts_by_method(MethodIdentifier.TREE_OF_THOUGHTS)

        assert tot_thoughts == []

    def test_get_thoughts_by_type_empty(self):
        """Test get_thoughts_by_type on empty session."""
        session = Session()

        thoughts = session.get_thoughts_by_type(ThoughtType.INITIAL)

        assert thoughts == []

    def test_get_thoughts_by_type_with_matches(self):
        """Test get_thoughts_by_type returns matching thoughts."""
        session = Session()

        thought_initial_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Initial 1",
        )
        thought_initial_2 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Initial 2",
        )
        thought_continuation = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Continuation",
        )

        session.add_thought(thought_initial_1).add_thought(thought_initial_2).add_thought(
            thought_continuation
        )

        initial_thoughts = session.get_thoughts_by_type(ThoughtType.INITIAL)

        assert len(initial_thoughts) == 2
        assert thought_initial_1 in initial_thoughts
        assert thought_initial_2 in initial_thoughts
        assert thought_continuation not in initial_thoughts

    def test_get_recent_thoughts_default(self):
        """Test get_recent_thoughts with default count (5)."""
        session = Session()

        # Add 15 thoughts
        for i in range(15):
            thought = ThoughtNode(
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.SEQUENTIAL_THINKING,
                content=f"Thought {i}",
            )
            session.add_thought(thought)

        recent = session.get_recent_thoughts()

        # Should return only 5 most recent (default is n=5)
        assert len(recent) == 5

    def test_get_recent_thoughts_custom_count(self):
        """Test get_recent_thoughts with custom count."""
        session = Session()

        # Add 10 thoughts
        for i in range(10):
            thought = ThoughtNode(
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.SEQUENTIAL_THINKING,
                content=f"Thought {i}",
            )
            session.add_thought(thought)

        # Get only 5 most recent
        recent = session.get_recent_thoughts(n=5)

        assert len(recent) == 5

    def test_get_recent_thoughts_empty_session(self):
        """Test get_recent_thoughts on empty session."""
        session = Session()

        recent = session.get_recent_thoughts()

        assert recent == []


# ============================================================================
# Session Tests - Serialization
# ============================================================================


class TestSessionSerialization:
    """Test suite for Session JSON serialization/deserialization."""

    def test_json_serialization(self):
        """Test JSON serialization of Session."""
        session = Session()

        json_str = session.model_dump_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_deserialization(self):
        """Test JSON deserialization of Session."""
        session = Session()
        json_str = session.model_dump_json()

        deserialized = Session.model_validate_json(json_str)

        assert deserialized.id == session.id
        assert deserialized.status == session.status
        assert deserialized.config.max_depth == session.config.max_depth

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization preserves all data."""
        original = Session()
        original.start()

        # Add some thoughts
        thought_1 = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test thought 1",
        )
        thought_2 = ThoughtNode(
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="Test thought 2",
            parent_id=thought_1.id,
        )
        original.add_thought(thought_1).add_thought(thought_2)

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        deserialized = Session.model_validate_json(json_str)

        # Verify key properties preserved
        assert deserialized.id == original.id
        assert deserialized.status == original.status
        assert deserialized.thought_count == original.thought_count
        assert deserialized.metrics.total_thoughts == original.metrics.total_thoughts

    def test_serialization_with_thoughts(self, sample_thought: ThoughtNode):
        """Test serialization of Session containing thoughts."""
        session = Session()
        session.start()
        session.add_thought(sample_thought)

        json_str = session.model_dump_json()

        # Verify thought is included in serialization
        assert sample_thought.id in json_str
        assert sample_thought.content in json_str

        # Deserialize and verify
        deserialized = Session.model_validate_json(json_str)
        assert deserialized.thought_count == 1
        assert sample_thought.id in deserialized.graph.nodes
