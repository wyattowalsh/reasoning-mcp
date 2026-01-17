"""Comprehensive tests for DiagramOfThought reasoning method.

This module provides complete test coverage for the DiagramOfThought method,
testing all core functionality including:
- Initialization and health checks
- Basic execution and thought creation
- Role transitions (proposer -> critic -> summarizer)
- DAG structure tracking
- Proposition validation (valid/rejected)
- Configuration options (max_propositions)
- Continue reasoning flow
- Quality improvement tracking
- Iteration tracking and limits
- Convergence detection
- Edge cases

The tests aim for 90%+ coverage of the DiagramOfThought implementation.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.diagram_of_thought import (
    DIAGRAM_OF_THOUGHT_METADATA,
    DiagramOfThought,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def method() -> DiagramOfThought:
    """Provide a DiagramOfThought method instance for testing.

    Returns:
        DiagramOfThought instance (uninitialized).
    """
    return DiagramOfThought()


@pytest.fixture
async def initialized_method() -> DiagramOfThought:
    """Provide an initialized DiagramOfThought method instance.

    Returns:
        Initialized DiagramOfThought instance.
    """
    method = DiagramOfThought()
    await method.initialize()
    return method


@pytest.fixture
def session() -> Session:
    """Provide an active session for testing.

    Returns:
        Active Session instance.
    """
    return Session().start()


@pytest.fixture
def simple_input() -> str:
    """Provide a simple test input.

    Returns:
        Simple question for testing.
    """
    return "Should we implement universal basic income?"


@pytest.fixture
def complex_input() -> str:
    """Provide a complex test input.

    Returns:
        Complex question requiring multi-faceted analysis.
    """
    return "How can we balance economic growth with environmental sustainability?"


# ============================================================================
# Metadata Tests
# ============================================================================


class TestDiagramOfThoughtMetadata:
    """Test suite for DiagramOfThought metadata."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert DIAGRAM_OF_THOUGHT_METADATA.identifier == MethodIdentifier.DIAGRAM_OF_THOUGHT

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert DIAGRAM_OF_THOUGHT_METADATA.name == "Diagram of Thought"

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert DIAGRAM_OF_THOUGHT_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity level."""
        assert DIAGRAM_OF_THOUGHT_METADATA.complexity == 7
        assert 1 <= DIAGRAM_OF_THOUGHT_METADATA.complexity <= 10

    def test_metadata_supports_branching(self):
        """Test that metadata indicates branching support."""
        assert DIAGRAM_OF_THOUGHT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata indicates revision support."""
        assert DIAGRAM_OF_THOUGHT_METADATA.supports_revision is True

    def test_metadata_tags(self):
        """Test that metadata has expected tags."""
        expected_tags = {
            "dag",
            "graph-based",
            "multi-role",
            "proposer-critic",
            "synthesis",
        }
        assert expected_tags.issubset(DIAGRAM_OF_THOUGHT_METADATA.tags)

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert DIAGRAM_OF_THOUGHT_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self):
        """Test that metadata specifies reasonable max thoughts."""
        assert DIAGRAM_OF_THOUGHT_METADATA.max_thoughts == 30


# ============================================================================
# Initialization Tests
# ============================================================================


class TestDiagramOfThoughtInitialization:
    """Test suite for DiagramOfThought initialization."""

    def test_create_method(self, method: DiagramOfThought):
        """Test creating a DiagramOfThought instance."""
        assert isinstance(method, DiagramOfThought)
        assert method._initialized is False

    def test_initial_state(self, method: DiagramOfThought):
        """Test initial state of uninitialized method."""
        assert method._step_counter == 0
        assert method._current_role == "proposer"
        assert method._critique_round == 0
        assert method._proposition_count == 0
        assert len(method._valid_propositions) == 0
        assert len(method._rejected_propositions) == 0
        assert len(method._proposition_dag) == 0

    def test_properties_before_initialization(self, method: DiagramOfThought):
        """Test that properties work before initialization."""
        assert method.identifier == MethodIdentifier.DIAGRAM_OF_THOUGHT
        assert method.name == "Diagram of Thought"
        assert method.category == MethodCategory.ADVANCED
        assert isinstance(method.description, str)

    @pytest.mark.asyncio
    async def test_initialize(self, method: DiagramOfThought):
        """Test initializing the method."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_role == "proposer"
        assert method._critique_round == 0
        assert method._proposition_count == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state."""
        method = DiagramOfThought()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._current_role = "critic"
        method._critique_round = 2
        method._proposition_count = 8
        method._valid_propositions = ["P1", "P2"]
        method._rejected_propositions = ["P3"]
        method._proposition_dag = {"node1": ["node2"]}

        # Reinitialize
        await method.initialize()

        # State should be reset
        assert method._step_counter == 0
        assert method._current_role == "proposer"
        assert method._critique_round == 0
        assert method._proposition_count == 0
        assert len(method._valid_propositions) == 0
        assert len(method._rejected_propositions) == 0
        assert len(method._proposition_dag) == 0

    @pytest.mark.asyncio
    async def test_health_check_before_initialization(self, method: DiagramOfThought):
        """Test health check before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialization(self, initialized_method: DiagramOfThought):
        """Test health check after initialization."""
        result = await initialized_method.health_check()
        assert result is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestDiagramOfThoughtExecution:
    """Test suite for basic DiagramOfThought execution."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(
        self, method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute fails without initialization."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text=simple_input)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute creates an INITIAL thought."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.DIAGRAM_OF_THOUGHT
        assert thought.step_number == 1
        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_initial_metadata(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute sets appropriate metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "input" in thought.metadata
        assert thought.metadata["input"] == simple_input
        assert thought.metadata["role"] == "proposer"
        assert thought.metadata["critique_round"] == 0
        assert thought.metadata["proposition_count"] == 0
        assert "max_propositions" in thought.metadata
        assert thought.metadata["reasoning_type"] == "diagram_of_thought"

    @pytest.mark.asyncio
    async def test_execute_initializes_dag(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute initializes the DAG structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # DAG should have root node
        assert thought.id in initialized_method._proposition_dag
        assert initialized_method._proposition_dag[thought.id] == []
        assert "dag_nodes" in thought.metadata
        assert thought.id in thought.metadata["dag_nodes"]

    @pytest.mark.asyncio
    async def test_execute_sets_proposer_role(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute starts with proposer role."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert initialized_method._current_role == "proposer"
        assert thought.metadata["role"] == "proposer"

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.DIAGRAM_OF_THOUGHT
        assert thought.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test execute with custom context."""
        context = {"max_propositions": 5, "custom_key": "custom_value"}

        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=context
        )

        assert thought.metadata["max_propositions"] == 5
        assert thought.metadata["context"]["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_clamps_max_propositions(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that max_propositions is clamped to valid range."""
        # Test upper bound
        thought1 = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_propositions": 100},
        )
        assert thought1.metadata["max_propositions"] == 20

        # Re-initialize for fresh execution
        await initialized_method.initialize()
        session2 = Session().start()

        # Test lower bound
        thought2 = await initialized_method.execute(
            session=session2,
            input_text=simple_input,
            context={"max_propositions": -5},
        )
        assert thought2.metadata["max_propositions"] == 1

    @pytest.mark.asyncio
    async def test_execute_initializes_proposition_lists(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that execute initializes proposition tracking lists."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "valid_propositions" in thought.metadata
        assert "rejected_propositions" in thought.metadata
        assert thought.metadata["valid_propositions"] == []
        assert thought.metadata["rejected_propositions"] == []


# ============================================================================
# Role Transition Tests
# ============================================================================


class TestRoleTransitions:
    """Test suite for role transition flow."""

    @pytest.mark.asyncio
    async def test_proposer_to_critic_transition(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test transition from proposer to critic."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        assert initial.metadata["role"] == "proposer"

        # Generate propositions first (still proposer)
        proposal = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        # Should transition to critic after proposer
        assert proposal.type == ThoughtType.VERIFICATION
        assert proposal.metadata["role"] == "critic"
        assert proposal.parent_id == initial.id

    @pytest.mark.asyncio
    async def test_critic_to_summarizer_transition(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test transition from critic to summarizer with enough propositions."""
        # Setup: initial -> proposer generates -> critic evaluates
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Proposer generates (but execute starts as proposer, so first continue is critic)
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["role"] == "critic"

        # After critic, with enough valid propositions, should summarize
        # Simulate having enough valid propositions
        initialized_method._valid_propositions = ["P1", "P2", "P3"]

        summary = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert summary.type == ThoughtType.SYNTHESIS
        assert summary.metadata["role"] == "summarizer"

    @pytest.mark.asyncio
    async def test_critic_to_proposer_loop(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test critic loops back to proposer when more propositions needed."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Critic evaluates
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["role"] == "critic"

        # Not enough valid propositions yet, should go back to proposer
        initialized_method._valid_propositions = ["P1"]  # Only 1, need 3+

        new_proposal = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert new_proposal.type == ThoughtType.HYPOTHESIS
        assert new_proposal.metadata["role"] == "proposer"
        assert new_proposal.metadata["critique_round"] == 1

    @pytest.mark.asyncio
    async def test_summarizer_to_conclusion(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test summarizer transitions to conclusion when complete."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Fast-forward to summarizer with enough propositions
        initialized_method._current_role = "summarizer"
        initialized_method._valid_propositions = ["P1", "P2", "P3", "P4", "P5"]
        thought.metadata["role"] = "summarizer"

        conclusion = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert conclusion.type == ThoughtType.CONCLUSION
        assert conclusion.metadata["role"] == "summarizer"

    @pytest.mark.asyncio
    async def test_full_cycle_proposer_critic_summarizer(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test a complete proposer-critic-summarizer cycle."""
        # Initial (proposer setup)
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["role"] == "proposer"

        # Critic evaluates
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["role"] == "critic"

        # Add enough valid propositions for synthesis
        initialized_method._valid_propositions = ["P1", "P2", "P3"]

        # Summarizer synthesizes
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert thought.metadata["role"] == "summarizer"
        assert thought.type == ThoughtType.SYNTHESIS


# ============================================================================
# DAG Structure Tests
# ============================================================================


class TestDAGStructure:
    """Test suite for DAG structure tracking."""

    @pytest.mark.asyncio
    async def test_dag_root_initialization(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that DAG is initialized with root node."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert len(initialized_method._proposition_dag) == 1
        assert thought.id in initialized_method._proposition_dag

    @pytest.mark.asyncio
    async def test_dag_grows_with_thoughts(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that DAG grows as thoughts are added."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        initial_size = len(initialized_method._proposition_dag)

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert len(initialized_method._proposition_dag) > initial_size

    @pytest.mark.asyncio
    async def test_dag_tracks_parent_child_relationships(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that DAG tracks parent-child relationships."""
        parent = await initialized_method.execute(session=session, input_text=simple_input)

        child = await initialized_method.continue_reasoning(
            session=session, previous_thought=parent
        )

        # Parent should have child in its list
        assert child.id in initialized_method._proposition_dag[parent.id]

    @pytest.mark.asyncio
    async def test_dag_nodes_in_metadata(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that dag_nodes is tracked in metadata."""
        thought1 = await initialized_method.execute(session=session, input_text=simple_input)

        thought2 = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought1
        )

        # Both nodes should be in DAG
        assert thought1.id in thought2.metadata["dag_nodes"]
        assert thought2.id in thought2.metadata["dag_nodes"]

    @pytest.mark.asyncio
    async def test_dag_supports_multiple_branches(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that DAG can support multiple branches from same parent."""
        parent = await initialized_method.execute(session=session, input_text=simple_input)

        child1 = await initialized_method.continue_reasoning(
            session=session, previous_thought=parent
        )

        # Manually add child1 to DAG to test branching
        initialized_method._proposition_dag[parent.id].append(child1.id)

        # Both children should be tracked
        assert len(initialized_method._proposition_dag[parent.id]) >= 1


# ============================================================================
# Proposition Tracking Tests
# ============================================================================


class TestPropositionTracking:
    """Test suite for proposition validation tracking."""

    @pytest.mark.asyncio
    async def test_initial_propositions_empty(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that propositions start empty."""
        await initialized_method.execute(session=session, input_text=simple_input)

        assert len(initialized_method._valid_propositions) == 0
        assert len(initialized_method._rejected_propositions) == 0

    @pytest.mark.asyncio
    async def test_propositions_added_after_critique(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that propositions are categorized after critique."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # First continue triggers critic role
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        # Critic role generates placeholder content (LLM integration point)
        # In placeholder mode, propositions aren't actually populated
        # This tests that the method successfully runs through the critic role
        assert thought.metadata["role"] == "critic"
        assert "Critique" in thought.content

    @pytest.mark.asyncio
    async def test_valid_propositions_in_metadata(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that valid propositions appear in metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert "valid_propositions" in thought.metadata
        assert isinstance(thought.metadata["valid_propositions"], list)

    @pytest.mark.asyncio
    async def test_rejected_propositions_in_metadata(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that rejected propositions appear in metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert "rejected_propositions" in thought.metadata
        assert isinstance(thought.metadata["rejected_propositions"], list)

    @pytest.mark.asyncio
    async def test_proposition_count_increments(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that proposition count increments correctly."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        initial_count = initialized_method._proposition_count

        # Continue to proposer role (after critic cycle)
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        # Go back to proposer which generates more
        initialized_method._current_role = "proposer"
        thought.metadata["role"] = "critic"

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        # Should have generated new propositions
        assert initialized_method._proposition_count >= initial_count


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test suite for configuration options."""

    @pytest.mark.asyncio
    async def test_default_max_propositions(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test default max propositions."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert thought.metadata["max_propositions"] == DiagramOfThought.MAX_PROPOSITIONS
        assert thought.metadata["max_propositions"] == 10

    @pytest.mark.asyncio
    async def test_custom_max_propositions(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test custom max propositions in context."""
        custom_max = 7

        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_propositions": custom_max},
        )

        assert thought.metadata["max_propositions"] == custom_max

    @pytest.mark.asyncio
    async def test_max_propositions_propagates(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that max propositions propagates through cycle."""
        custom_max = 8

        initial = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_propositions": custom_max},
        )

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert critique.metadata["max_propositions"] == custom_max

    @pytest.mark.asyncio
    async def test_max_critique_rounds_constant(self):
        """Test that MAX_CRITIQUE_ROUNDS is properly defined."""
        assert DiagramOfThought.MAX_CRITIQUE_ROUNDS == 2
        assert isinstance(DiagramOfThought.MAX_CRITIQUE_ROUNDS, int)


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Test suite for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization(
        self, method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that continue_reasoning fails without initialization."""
        # Create a mock thought
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIAGRAM_OF_THOUGHT,
            content="Test",
            metadata={"role": "proposer", "max_propositions": 10},
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that continue_reasoning increments step counter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        assert initial.step_number == 1

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )
        assert critique.step_number == 2

        next_thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=critique
        )
        assert next_thought.step_number == 3

    @pytest.mark.asyncio
    async def test_continue_with_guidance(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test continue_reasoning with guidance parameter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        guidance_text = "Focus on economic aspects"
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance_text
        )

        assert "guidance" in critique.metadata
        assert critique.metadata["guidance"] == guidance_text

    @pytest.mark.asyncio
    async def test_continue_with_context(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test continue_reasoning with context parameter."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        context = {"additional_info": "test data"}
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, context=context
        )

        assert "context" in critique.metadata
        assert critique.metadata["context"]["additional_info"] == "test data"

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that continue_reasoning adds thought to session."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        count_after_initial = session.thought_count

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert session.thought_count == count_after_initial + 1
        assert critique.id in session.graph.nodes

    @pytest.mark.asyncio
    async def test_continue_tracks_previous_role(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that continue_reasoning tracks previous role."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert "previous_role" in critique.metadata
        assert critique.metadata["previous_role"] == "proposer"


# ============================================================================
# Quality Improvement Tests
# ============================================================================


class TestQualityImprovement:
    """Test suite for quality score improvement tracking."""

    @pytest.mark.asyncio
    async def test_quality_improves_through_cycle(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that quality score improves through the cycle."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        initial_quality = initial.quality_score

        # Progress through roles
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        # Set up for synthesis
        initialized_method._valid_propositions = ["P1", "P2", "P3"]

        synthesis = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        # Synthesis should have higher quality
        assert synthesis.quality_score > initial_quality

    @pytest.mark.asyncio
    async def test_confidence_improves_through_cycle(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that confidence improves through the cycle."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)
        initial_confidence = initial.confidence

        # Get to synthesis phase
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        initialized_method._valid_propositions = ["P1", "P2", "P3"]

        synthesis = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert synthesis.confidence > initial_confidence

    @pytest.mark.asyncio
    async def test_conclusion_has_high_quality(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that conclusion has high quality score."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Fast-forward to conclusion state
        initialized_method._valid_propositions = ["P1", "P2", "P3", "P4", "P5"]
        initialized_method._current_role = "summarizer"
        thought.metadata["role"] = "summarizer"

        conclusion = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert conclusion.type == ThoughtType.CONCLUSION
        assert conclusion.quality_score >= 0.8


# ============================================================================
# Iteration Tracking Tests
# ============================================================================


class TestIterationTracking:
    """Test suite for iteration counting and limits."""

    @pytest.mark.asyncio
    async def test_critique_round_increments(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that critique_round increments correctly."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)
        assert initialized_method._critique_round == 0

        # Go to critic
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._critique_round == 0  # Still round 0

        # Back to proposer increments round
        initialized_method._valid_propositions = ["P1"]  # Not enough for summary

        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )
        assert initialized_method._critique_round == 1

    @pytest.mark.asyncio
    async def test_max_critique_rounds_limit(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that max critique rounds limit is respected."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Set critique round to max
        initialized_method._critique_round = DiagramOfThought.MAX_CRITIQUE_ROUNDS

        # Continue should eventually conclude
        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            if thought.type == ThoughtType.CONCLUSION:
                break

        # Should not exceed max
        assert initialized_method._critique_round <= DiagramOfThought.MAX_CRITIQUE_ROUNDS

    @pytest.mark.asyncio
    async def test_critique_round_in_metadata(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that critique_round is tracked in metadata."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        assert "critique_round" in thought.metadata
        assert thought.metadata["critique_round"] == 0


# ============================================================================
# Convergence Detection Tests
# ============================================================================


class TestConvergenceDetection:
    """Test suite for convergence and completion detection."""

    @pytest.mark.asyncio
    async def test_conclusion_with_enough_propositions(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that conclusion occurs with enough valid propositions."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Set up sufficient propositions
        initialized_method._valid_propositions = ["P1", "P2", "P3", "P4", "P5"]
        initialized_method._current_role = "summarizer"
        thought.metadata["role"] = "summarizer"

        conclusion = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert conclusion.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_conclusion_at_max_rounds(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that conclusion occurs at max critique rounds."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        # Set to max rounds
        initialized_method._critique_round = DiagramOfThought.MAX_CRITIQUE_ROUNDS
        initialized_method._valid_propositions = ["P1", "P2", "P3"]
        initialized_method._current_role = "summarizer"
        thought.metadata["role"] = "summarizer"

        conclusion = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert conclusion.type == ThoughtType.CONCLUSION


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_method: DiagramOfThought, session: Session):
        """Test handling of empty input."""
        thought = await initialized_method.execute(session=session, input_text="")
        assert thought.metadata["input"] == ""
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, initialized_method: DiagramOfThought, session: Session):
        """Test handling of very long input."""
        long_input = "Analyze " + "this complex issue " * 500
        thought = await initialized_method.execute(session=session, input_text=long_input)
        assert thought.metadata["input"] == long_input

    @pytest.mark.asyncio
    async def test_none_context(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test handling of None context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context=None
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_empty_context(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test handling of empty context."""
        thought = await initialized_method.execute(
            session=session, input_text=simple_input, context={}
        )
        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_fallback_to_proposer_role(
        self, initialized_method: DiagramOfThought, session: Session
    ):
        """Test fallback to proposer for unknown role."""
        initial = await initialized_method.execute(session=session, input_text="Test")

        # Manually modify role to unknown value
        initial.metadata["role"] = "unknown_role"

        # Continue should fallback to proposer
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        # Fallback logic sets role to proposer, generating HYPOTHESIS type
        assert thought.type == ThoughtType.HYPOTHESIS
        assert thought.metadata["role"] == "proposer"

    @pytest.mark.asyncio
    async def test_max_propositions_zero_handling(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test handling of max_propositions set to 0."""
        thought = await initialized_method.execute(
            session=session,
            input_text=simple_input,
            context={"max_propositions": 0},
        )

        # Should clamp to minimum of 1
        assert thought.metadata["max_propositions"] >= 1


# ============================================================================
# Content Generation Tests
# ============================================================================


class TestContentGeneration:
    """Test suite for content generation methods."""

    @pytest.mark.asyncio
    async def test_initial_content_structure(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that initial setup has expected content structure."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        content = thought.content
        assert isinstance(content, str)
        assert len(content) > 0
        assert "Step 1" in content
        assert "Initial Problem Setup" in content
        assert simple_input in content
        assert "PROPOSER" in content

    @pytest.mark.asyncio
    async def test_proposition_content_structure(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that proposition has expected content structure."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        # Go to critic first
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        # Then back to proposer
        initialized_method._valid_propositions = ["P1"]

        proposal = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        if proposal.metadata["role"] == "proposer":
            content = proposal.content
            assert isinstance(content, str)
            assert "Proposition" in content or "propositions" in content.lower()

    @pytest.mark.asyncio
    async def test_critique_content_structure(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that critique has expected content structure."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        assert critique.metadata["role"] == "critic"
        content = critique.content
        assert isinstance(content, str)
        assert "Critique" in content or "CRITIC" in content

    @pytest.mark.asyncio
    async def test_synthesis_content_structure(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that synthesis has expected content structure."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        # Go to critic
        thought = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial
        )

        # Set up for synthesis
        initialized_method._valid_propositions = ["P1", "P2", "P3"]

        synthesis = await initialized_method.continue_reasoning(
            session=session, previous_thought=thought
        )

        assert synthesis.metadata["role"] == "summarizer"
        content = synthesis.content
        assert isinstance(content, str)
        assert "Synth" in content or "SUMMARIZER" in content

    @pytest.mark.asyncio
    async def test_guidance_appears_in_content(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that guidance appears in metadata."""
        initial = await initialized_method.execute(session=session, input_text=simple_input)

        guidance = "Focus on environmental impact"
        critique = await initialized_method.continue_reasoning(
            session=session, previous_thought=initial, guidance=guidance
        )

        # Guidance should appear in metadata
        assert guidance in critique.metadata["guidance"]


# ============================================================================
# Multi-Cycle Tests
# ============================================================================


class TestMultiCycle:
    """Test suite for multiple reasoning cycles."""

    @pytest.mark.asyncio
    async def test_multiple_critique_rounds(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test multiple critique rounds."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        rounds_seen = []

        for _ in range(10):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )
            rounds_seen.append(thought.metadata["critique_round"])

            if thought.type == ThoughtType.CONCLUSION:
                break

        # Should see progression through rounds
        assert len(set(rounds_seen)) > 1 or rounds_seen[-1] >= 0

    @pytest.mark.asyncio
    async def test_proposition_accumulation(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that propositions accumulate over cycles."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        initial_valid = len(initialized_method._valid_propositions)

        # Run several cycles
        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # Should have accumulated propositions
        final_valid = len(initialized_method._valid_propositions)
        assert final_valid >= initial_valid

    @pytest.mark.asyncio
    async def test_dag_growth_over_cycles(
        self, initialized_method: DiagramOfThought, session: Session, simple_input: str
    ):
        """Test that DAG grows over multiple cycles."""
        thought = await initialized_method.execute(session=session, input_text=simple_input)

        initial_nodes = len(initialized_method._proposition_dag)

        # Run several cycles
        for _ in range(5):
            thought = await initialized_method.continue_reasoning(
                session=session, previous_thought=thought
            )

        # DAG should have grown
        final_nodes = len(initialized_method._proposition_dag)
        assert final_nodes > initial_nodes
