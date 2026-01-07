"""Unit tests for LotusWisdom reasoning method.

This module provides comprehensive tests for the LotusWisdom method implementation,
covering initialization, execution, five wisdom domains, domain balance, configuration,
continuation reasoning, center thought, petal generation, synthesis, and edge cases.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from reasoning_mcp.methods.native.lotus_wisdom import (
    LOTUS_WISDOM_METADATA,
    LotusWisdomMethod,
)
from reasoning_mcp.models import Session, ThoughtGraph, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def lotus_method() -> LotusWisdomMethod:
    """Create a LotusWisdomMethod instance for testing.

    Returns:
        A fresh LotusWisdomMethod instance
    """
    return LotusWisdomMethod()


@pytest.fixture
def initialized_method() -> LotusWisdomMethod:
    """Create an initialized LotusWisdomMethod instance.

    Returns:
        A LotusWisdomMethod instance (tests will need to await initialize())
    """
    return LotusWisdomMethod()


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A new Session instance in ACTIVE status
    """
    return Session().start()


@pytest.fixture
def sample_problem() -> str:
    """Provide a sample problem for testing.

    Returns:
        A sample decision problem string
    """
    return "Should our company adopt a 4-day work week?"


@pytest.fixture
def technical_problem() -> str:
    """Provide a technical-focused problem for testing.

    Returns:
        A technical problem string
    """
    return "How can we optimize our database queries to reduce latency by 50%?"


@pytest.fixture
def emotional_problem() -> str:
    """Provide an emotional-focused problem for testing.

    Returns:
        An emotional problem string
    """
    return "How can I help my team feel more valued and motivated after recent layoffs?"


@pytest.fixture
def balanced_problem() -> str:
    """Provide a balanced multi-domain problem for testing.

    Returns:
        A complex balanced problem string
    """
    return (
        "Should we expand our business into international markets? Consider the "
        "technical infrastructure requirements, emotional impact on current employees, "
        "ethical implications of entering new markets, practical resource constraints, "
        "and whether it intuitively feels like the right strategic move."
    )


class TestLotusWisdomInitialization:
    """Tests for LotusWisdom initialization and setup."""

    def test_create_method(self, lotus_method: LotusWisdomMethod):
        """Test that LotusWisdomMethod can be instantiated."""
        assert lotus_method is not None
        assert isinstance(lotus_method, LotusWisdomMethod)

    def test_initial_state(self, lotus_method: LotusWisdomMethod):
        """Test that a new method starts in the correct initial state."""
        assert lotus_method._step_counter == 0
        assert lotus_method._initialized is False
        assert len(lotus_method._domain_balance) == 5
        assert all(count == 0 for count in lotus_method._domain_balance.values())

    async def test_initialize(self, lotus_method: LotusWisdomMethod):
        """Test that initialize() sets up the method correctly."""
        await lotus_method.initialize()
        assert lotus_method._initialized is True
        assert lotus_method._step_counter == 0
        assert all(count == 0 for count in lotus_method._domain_balance.values())

    async def test_initialize_resets_state(self):
        """Test that initialize() resets state even if called multiple times."""
        method = LotusWisdomMethod()
        await method.initialize()
        method._step_counter = 10
        method._domain_balance["TECHNICAL"] = 5

        # Re-initialize
        await method.initialize()
        assert method._step_counter == 0
        assert method._domain_balance["TECHNICAL"] == 0
        assert method._initialized is True

    async def test_health_check_not_initialized(self, lotus_method: LotusWisdomMethod):
        """Test that health_check returns False before initialization."""
        result = await lotus_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, lotus_method: LotusWisdomMethod):
        """Test that health_check returns True after initialization."""
        await lotus_method.initialize()
        result = await lotus_method.health_check()
        assert result is True


class TestLotusWisdomProperties:
    """Tests for LotusWisdom property accessors."""

    def test_identifier_property(self, lotus_method: LotusWisdomMethod):
        """Test that identifier returns the correct method identifier."""
        assert lotus_method.identifier == str(MethodIdentifier.LOTUS_WISDOM)

    def test_name_property(self, lotus_method: LotusWisdomMethod):
        """Test that name returns the correct human-readable name."""
        assert lotus_method.name == "Lotus Wisdom"

    def test_description_property(self, lotus_method: LotusWisdomMethod):
        """Test that description returns the correct method description."""
        assert "5-domain" in lotus_method.description or "holistic" in lotus_method.description.lower()
        # Description focuses on the 5 domains: technical, emotional, ethical, practical, intuitive
        assert any(domain in lotus_method.description.lower() for domain in ["technical", "emotional", "ethical", "practical", "intuitive"])

    def test_category_property(self, lotus_method: LotusWisdomMethod):
        """Test that category returns the correct method category."""
        assert lotus_method.category == str(MethodCategory.HOLISTIC)

    def test_domains_constant(self, lotus_method: LotusWisdomMethod):
        """Test that DOMAINS constant contains all 5 required domains."""
        assert len(lotus_method.DOMAINS) == 5
        assert "TECHNICAL" in lotus_method.DOMAINS
        assert "EMOTIONAL" in lotus_method.DOMAINS
        assert "ETHICAL" in lotus_method.DOMAINS
        assert "PRACTICAL" in lotus_method.DOMAINS
        assert "INTUITIVE" in lotus_method.DOMAINS


class TestLotusWisdomMetadata:
    """Tests for LotusWisdom metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert LOTUS_WISDOM_METADATA.identifier == MethodIdentifier.LOTUS_WISDOM

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert LOTUS_WISDOM_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"holistic", "wisdom", "multi-domain"}
        assert expected_tags.issubset(LOTUS_WISDOM_METADATA.tags)

    def test_metadata_supports_branching(self):
        """Test that metadata correctly indicates branching support."""
        assert LOTUS_WISDOM_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata correctly indicates revision support."""
        assert LOTUS_WISDOM_METADATA.supports_revision is True

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity rating for holistic method."""
        assert LOTUS_WISDOM_METADATA.complexity >= 5  # Should be moderately complex
        assert LOTUS_WISDOM_METADATA.complexity <= 10

    def test_metadata_min_thoughts(self):
        """Test that metadata defines minimum thoughts (center + 5 domains + synthesis)."""
        assert LOTUS_WISDOM_METADATA.min_thoughts == 7


class TestLotusWisdomBasicExecution:
    """Tests for basic LotusWisdom execute() method."""

    async def test_execute_requires_initialization(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute raises error if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await lotus_method.execute(session, sample_problem)

    async def test_execute_creates_lotus_structure(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates complete lotus structure with 7 thoughts."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        assert graph is not None
        assert isinstance(graph, ThoughtGraph)
        assert graph.node_count == 7  # Center + 5 domains + synthesis

    async def test_execute_creates_center_thought(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates a center thought as the root."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        assert root is not None
        assert root.type == ThoughtType.INITIAL
        assert root.metadata.get("phase") == "center"
        assert root.metadata.get("structure") == "lotus_core"
        assert root.depth == 0

    async def test_execute_creates_five_domain_petals(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates exactly 5 domain petal thoughts."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        # Count domain thoughts
        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]
        assert len(domain_thoughts) == 5

        # Verify each domain is represented
        domains_found = {node.metadata.get("domain") for node in domain_thoughts}
        assert domains_found == {"TECHNICAL", "EMOTIONAL", "ETHICAL", "PRACTICAL", "INTUITIVE"}

    async def test_execute_creates_synthesis(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates a synthesis thought."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert len(synthesis) == 1
        assert synthesis[0].type == ThoughtType.SYNTHESIS
        assert synthesis[0].metadata.get("structure") == "lotus_bloom"

    async def test_execute_updates_session(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute adds thoughts to the session."""
        await lotus_method.initialize()
        initial_count = session.thought_count

        await lotus_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 7
        assert session.current_method == MethodIdentifier.LOTUS_WISDOM

    async def test_execute_sets_step_numbers(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets sequential step numbers."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        step_numbers = sorted([node.step_number for node in graph.nodes.values()])
        assert step_numbers == [1, 2, 3, 4, 5, 6, 7]


class TestLotusWisdomFiveDomains:
    """Tests for the five wisdom domains."""

    async def test_technical_domain_exists(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that TECHNICAL domain is created and has correct properties."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        technical = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "TECHNICAL"
        ]
        assert len(technical) == 1
        assert technical[0].type == ThoughtType.BRANCH
        assert technical[0].depth == 1

    async def test_emotional_domain_exists(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that EMOTIONAL domain is created and has correct properties."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        emotional = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "EMOTIONAL"
        ]
        assert len(emotional) == 1
        assert emotional[0].type == ThoughtType.BRANCH
        assert emotional[0].metadata.get("structure") == "lotus_petal"

    async def test_ethical_domain_exists(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that ETHICAL domain is created and has correct properties."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        ethical = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "ETHICAL"
        ]
        assert len(ethical) == 1
        assert ethical[0].type == ThoughtType.BRANCH

    async def test_practical_domain_exists(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that PRACTICAL domain is created and has correct properties."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        practical = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "PRACTICAL"
        ]
        assert len(practical) == 1
        assert practical[0].type == ThoughtType.BRANCH

    async def test_intuitive_domain_exists(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that INTUITIVE domain is created and has correct properties."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        intuitive = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "INTUITIVE"
        ]
        assert len(intuitive) == 1
        assert intuitive[0].type == ThoughtType.BRANCH

    async def test_all_domains_have_parent(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that all domain petals have the center as parent."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]

        for domain_thought in domain_thoughts:
            assert domain_thought.parent_id == root.id


class TestLotusWisdomDomainBalance:
    """Tests for domain balance tracking."""

    async def test_domain_balance_initialized_to_zero(self, lotus_method: LotusWisdomMethod):
        """Test that domain balance starts at zero for all domains."""
        await lotus_method.initialize()
        balance = lotus_method._check_domain_balance()
        assert all(count == 0 for count in balance.values())

    async def test_domain_balance_after_execution(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that domain balance is updated after execution."""
        await lotus_method.initialize()
        await lotus_method.execute(session, sample_problem)

        balance = lotus_method._check_domain_balance()
        assert all(count == 1 for count in balance.values())

    async def test_domain_balance_in_synthesis_metadata(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that synthesis includes domain balance in metadata."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert "balance_check" in synthesis[0].metadata
        balance = synthesis[0].metadata["balance_check"]
        assert all(count == 1 for count in balance.values())

    async def test_ensuring_all_domains_explored(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that all 5 domains are explored exactly once in basic execution."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        balance = lotus_method._check_domain_balance()
        assert len(balance) == 5
        assert all(count >= 1 for count in balance.values())


class TestLotusWisdomCenterThought:
    """Tests for center thought generation."""

    async def test_center_thought_is_root(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that center thought is the root of the graph."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        assert root.metadata.get("phase") == "center"

    async def test_center_thought_contains_problem(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that center thought metadata contains the problem."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        assert root.metadata.get("problem") == sample_problem

    async def test_center_thought_lists_pending_domains(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that center thought lists domains to be analyzed."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        pending_domains = root.metadata.get("domains_pending")
        assert len(pending_domains) == 5
        assert set(pending_domains) == {"TECHNICAL", "EMOTIONAL", "ETHICAL", "PRACTICAL", "INTUITIVE"}

    async def test_center_thought_has_content(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that center thought has generated content."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        root = graph.get_node(graph.root_id)
        assert len(root.content) > 0
        assert sample_problem in root.content


class TestLotusWisdomPetalGeneration:
    """Tests for petal (domain) generation."""

    async def test_petal_has_domain_metadata(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that each petal has domain metadata."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]

        for domain_thought in domain_thoughts:
            assert "domain" in domain_thought.metadata
            assert "domain_name" in domain_thought.metadata
            assert domain_thought.metadata["domain"] in lotus_method.DOMAINS

    async def test_petal_has_branch_id(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that each petal has a branch_id."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]

        for domain_thought in domain_thoughts:
            assert domain_thought.branch_id is not None
            assert domain_thought.branch_id.startswith("domain_")

    async def test_petal_content_mentions_domain(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that petal content mentions its domain."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]

        for domain_thought in domain_thoughts:
            domain_name = domain_thought.metadata["domain_name"]
            assert domain_name.lower() in domain_thought.content.lower()

    async def test_petal_confidence_scores(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that petals have reasonable confidence scores."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]

        for domain_thought in domain_thoughts:
            assert 0.0 <= domain_thought.confidence <= 1.0
            assert domain_thought.confidence > 0.0


class TestLotusWisdomSynthesis:
    """Tests for synthesis generation."""

    async def test_synthesis_combines_all_domains(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that synthesis metadata shows all domains were synthesized."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert "domains_synthesized" in synthesis[0].metadata
        synthesized = synthesis[0].metadata["domains_synthesized"]
        assert len(synthesized) == 5

    async def test_synthesis_has_parent(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that synthesis is a child of the center."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert synthesis[0].parent_id == graph.root_id

    async def test_synthesis_depth(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that synthesis is at depth 2."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert synthesis[0].depth == 2

    async def test_synthesis_quality_score(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that synthesis has a quality score."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        synthesis = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "synthesis"
        ]
        assert synthesis[0].quality_score is not None
        assert synthesis[0].quality_score > 0.0


class TestLotusWisdomContinueReasoning:
    """Tests for continue_reasoning functionality."""

    async def test_continue_requires_initialization(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning raises error if not initialized."""
        graph = ThoughtGraph()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await lotus_method.continue_reasoning(session, graph)

    async def test_continue_deepens_analysis(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning adds new thoughts."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)
        initial_count = graph.node_count

        updated_graph = await lotus_method.continue_reasoning(
            session, graph, feedback="Explore technical aspects more deeply"
        )

        assert updated_graph.node_count > initial_count

    async def test_continue_with_technical_feedback(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning focuses on TECHNICAL when asked."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        updated_graph = await lotus_method.continue_reasoning(
            session, graph, feedback="Explore technical implementation more deeply"
        )

        # Should have created a deeper technical thought
        deeper_thoughts = [
            node for node in updated_graph.nodes.values()
            if node.metadata.get("phase") == "deepening"
        ]
        assert len(deeper_thoughts) > 0
        assert any(node.metadata.get("domain") == "TECHNICAL" for node in deeper_thoughts)

    async def test_continue_with_emotional_feedback(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning focuses on EMOTIONAL when asked."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        updated_graph = await lotus_method.continue_reasoning(
            session, graph, feedback="Consider the emotional impact on people"
        )

        deeper_thoughts = [
            node for node in updated_graph.nodes.values()
            if node.metadata.get("phase") == "deepening"
        ]
        assert any(node.metadata.get("domain") == "EMOTIONAL" for node in deeper_thoughts)

    async def test_continue_creates_refined_synthesis(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning creates a refined synthesis."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        updated_graph = await lotus_method.continue_reasoning(
            session, graph, feedback="Explore practical constraints"
        )

        refined_synthesis = [
            node for node in updated_graph.nodes.values()
            if node.metadata.get("phase") == "refined_synthesis"
        ]
        assert len(refined_synthesis) > 0
        assert refined_synthesis[0].type == ThoughtType.SYNTHESIS

    async def test_continue_without_feedback(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning works without specific feedback."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        # Capture initial count before modification (graph is modified in place)
        initial_node_count = graph.node_count

        # Should still work and pick least-analyzed domain
        updated_graph = await lotus_method.continue_reasoning(session, graph)

        assert updated_graph.node_count > initial_node_count

    async def test_continue_updates_domain_balance(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning updates domain balance."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)
        initial_balance = lotus_method._check_domain_balance()

        await lotus_method.continue_reasoning(
            session, graph, feedback="Deepen ethical analysis"
        )

        updated_balance = lotus_method._check_domain_balance()
        # ETHICAL should have increased
        assert updated_balance["ETHICAL"] > initial_balance["ETHICAL"]


class TestLotusWisdomEdgeCases:
    """Tests for edge cases and specific problem types."""

    async def test_technical_only_problem(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        technical_problem: str,
    ):
        """Test handling of a purely technical problem."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, technical_problem)

        # Should still create all 5 domains even for technical problem
        assert graph.node_count == 7
        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]
        assert len(domain_thoughts) == 5

    async def test_emotional_problem(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        emotional_problem: str,
    ):
        """Test handling of an emotional/people-focused problem."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, emotional_problem)

        # Should still analyze from all domains
        assert graph.node_count == 7
        # Emotional domain should exist
        emotional = [
            node for node in graph.nodes.values()
            if node.metadata.get("domain") == "EMOTIONAL"
        ]
        assert len(emotional) == 1

    async def test_balanced_multidomain_problem(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        balanced_problem: str,
    ):
        """Test handling of a balanced problem requiring all domains."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, balanced_problem)

        # Should handle complex problem with full structure
        assert graph.node_count == 7
        # All domains should be present
        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]
        domains_found = {node.metadata.get("domain") for node in domain_thoughts}
        assert len(domains_found) == 5

    async def test_empty_problem_string(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
    ):
        """Test handling of empty problem string."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, "")

        # Should still create structure even with empty problem
        assert graph.node_count == 7

    async def test_very_long_problem(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
    ):
        """Test handling of very long problem description."""
        long_problem = " ".join(["This is a complex problem."] * 100)
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, long_problem)

        assert graph.node_count == 7
        root = graph.get_node(graph.root_id)
        assert root.metadata.get("problem") == long_problem

    async def test_multiple_executions_reset_state(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that multiple executions reset internal state correctly."""
        await lotus_method.initialize()

        # First execution
        graph1 = await lotus_method.execute(session, sample_problem)
        assert graph1.node_count == 7

        # Second execution should reset and create new structure
        session2 = Session().start()
        graph2 = await lotus_method.execute(session2, "Different problem")
        assert graph2.node_count == 7

        # Domain balance should be reset
        balance = lotus_method._check_domain_balance()
        assert all(count == 1 for count in balance.values())

    async def test_graph_structure_validity(
        self,
        lotus_method: LotusWisdomMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that the generated graph structure is valid."""
        await lotus_method.initialize()
        graph = await lotus_method.execute(session, sample_problem)

        # Root should exist
        assert graph.root_id is not None
        root = graph.get_node(graph.root_id)
        assert root is not None

        # All nodes should be reachable from root
        assert graph.node_count == 7

        # Check that all domain petals point to root as parent
        domain_thoughts = [
            node for node in graph.nodes.values()
            if node.metadata.get("phase") == "domain_analysis"
        ]
        for domain_thought in domain_thoughts:
            assert domain_thought.parent_id == root.id
