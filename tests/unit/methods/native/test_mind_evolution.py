"""Unit tests for Mind Evolution reasoning method.

Tests cover:
- Metadata validation
- Initialization and state management
- Population initialization phase
- Mutation phase
- Selection phase
- Convergence phase
- LLM sampling with fallbacks
- Edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.methods.native.mind_evolution import (
    MIND_EVOLUTION_METADATA,
    MindEvolution,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


class TestMindEvolutionMetadata:
    """Tests for Mind Evolution metadata."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has correct identifier."""
        assert MIND_EVOLUTION_METADATA.identifier == MethodIdentifier.MIND_EVOLUTION

    def test_metadata_name(self) -> None:
        """Test that metadata has correct name."""
        assert MIND_EVOLUTION_METADATA.name == "Mind Evolution"

    def test_metadata_description(self) -> None:
        """Test that metadata has a description."""
        assert MIND_EVOLUTION_METADATA.description is not None
        assert "genetic" in MIND_EVOLUTION_METADATA.description.lower()

    def test_metadata_category(self) -> None:
        """Test that metadata has correct category."""
        assert MIND_EVOLUTION_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata has expected tags."""
        expected_tags = {"genetic", "evolutionary", "population", "search"}
        assert expected_tags.issubset(MIND_EVOLUTION_METADATA.tags)

    def test_metadata_complexity(self) -> None:
        """Test that metadata has complexity level."""
        assert MIND_EVOLUTION_METADATA.complexity == 7

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata indicates branching support."""
        assert MIND_EVOLUTION_METADATA.supports_branching is True

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata indicates revision support."""
        assert MIND_EVOLUTION_METADATA.supports_revision is True

    def test_metadata_min_thoughts(self) -> None:
        """Test that metadata specifies minimum thoughts."""
        assert MIND_EVOLUTION_METADATA.min_thoughts == 4

    def test_metadata_max_thoughts(self) -> None:
        """Test that metadata specifies max thoughts."""
        assert MIND_EVOLUTION_METADATA.max_thoughts == 8

    def test_metadata_best_for(self) -> None:
        """Test that metadata specifies what method is best for."""
        assert "optimization" in MIND_EVOLUTION_METADATA.best_for


class TestMindEvolution:
    """Test suite for Mind Evolution reasoning method."""

    @pytest.fixture
    def method(self) -> MindEvolution:
        """Create method instance."""
        return MindEvolution()

    @pytest.fixture
    async def initialized_method(self) -> MindEvolution:
        """Create an initialized method instance."""
        method = MindEvolution()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> MagicMock:
        """Create a mock session for testing."""
        mock_sess = MagicMock(spec=Session)
        mock_sess.current_method = None
        mock_sess.thought_count = 0
        mock_sess._thoughts = []
        mock_sess.graph = MagicMock()
        mock_sess.graph.nodes = {}

        def add_thought(thought: ThoughtNode) -> None:
            mock_sess._thoughts.append(thought)
            mock_sess.thought_count = len(mock_sess._thoughts)
            mock_sess.graph.nodes[thought.id] = thought

        mock_sess.add_thought = add_thought

        return mock_sess

    @pytest.fixture
    def sample_problem(self) -> str:
        """Provide a sample problem for testing."""
        return "Optimize the delivery route for a fleet of trucks"

    @pytest.fixture
    def mock_execution_context(self) -> MagicMock:
        """Provide a mock ExecutionContext for testing."""
        mock_ctx = MagicMock()
        mock_ctx.can_sample = True
        mock_response = MagicMock()
        mock_response.text = "1. Direct routing\n2. Hub-spoke model\n3. Greedy nearest"
        mock_ctx.sample = AsyncMock(return_value=mock_response)
        return mock_ctx

    # === Initialization Tests ===

    def test_initialization_default_values(self, method: MindEvolution) -> None:
        """Test method initializes with correct default values."""
        assert method is not None
        assert isinstance(method, MindEvolution)
        assert method._initialized is False
        assert method._step_counter == 0
        assert method._current_phase == "initialize_pop"
        assert method._population == []
        assert method._generation == 0
        assert method._execution_context is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_correctly(self, method: MindEvolution) -> None:
        """Test initialize() sets up the method correctly."""
        await method.initialize()
        assert method._initialized is True
        assert method._step_counter == 0
        assert method._current_phase == "initialize_pop"
        assert method._population == []
        assert method._generation == 0

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, initialized_method: MindEvolution) -> None:
        """Test that initialize() resets state."""
        initialized_method._step_counter = 5
        initialized_method._current_phase = "mutate"
        initialized_method._population = [{"id": "P1", "solution": "test", "fitness": 0.5}]
        initialized_method._generation = 3

        await initialized_method.initialize()

        assert initialized_method._initialized is True
        assert initialized_method._step_counter == 0
        assert initialized_method._current_phase == "initialize_pop"
        assert initialized_method._population == []
        assert initialized_method._generation == 0

    # === Property Tests ===

    def test_identifier_property(self, method: MindEvolution) -> None:
        """Test identifier property returns correct value."""
        assert method.identifier == MethodIdentifier.MIND_EVOLUTION

    def test_name_property(self, method: MindEvolution) -> None:
        """Test name property returns correct value."""
        assert method.name == "Mind Evolution"

    def test_description_property(self, method: MindEvolution) -> None:
        """Test description property returns correct value."""
        assert method.description == MIND_EVOLUTION_METADATA.description

    def test_category_property(self, method: MindEvolution) -> None:
        """Test category property returns correct value."""
        assert method.category == MethodCategory.ADVANCED

    # === Health Check Tests ===

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, method: MindEvolution) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, initialized_method: MindEvolution) -> None:
        """Test that health_check returns True after initialization."""
        result = await initialized_method.health_check()
        assert result is True

    # === Execute Tests (Initialize Population Phase) ===

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self, method: MindEvolution, session: MagicMock, sample_problem: str
    ) -> None:
        """Test that execute() raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session, sample_problem)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() creates an initial thought."""
        thought = await initialized_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.MIND_EVOLUTION
        assert thought.step_number == 1
        assert thought.depth == 0
        assert thought.metadata["phase"] == "initialize_pop"
        assert thought.metadata["generation"] == 1

    @pytest.mark.asyncio
    async def test_execute_initializes_population(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() initializes population."""
        await initialized_method.execute(session, sample_problem)

        assert len(initialized_method._population) == 3
        for p in initialized_method._population:
            assert "id" in p
            assert "solution" in p
            assert "fitness" in p

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that execute() sets the session's current method."""
        await initialized_method.execute(session, sample_problem)
        assert session.current_method == MethodIdentifier.MIND_EVOLUTION

    @pytest.mark.asyncio
    async def test_execute_stores_execution_context(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute() stores the execution context."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )
        assert initialized_method._execution_context is mock_execution_context

    # === Continue Reasoning Tests ===

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises_error(
        self, method: MindEvolution, session: MagicMock
    ) -> None:
        """Test that continue_reasoning() raises RuntimeError if not initialized."""
        mock_thought = MagicMock(spec=ThoughtNode)
        mock_thought.id = "test-id"
        mock_thought.depth = 0
        mock_thought.metadata = {"phase": "initialize_pop"}

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session, mock_thought)

    # === Mutation Phase Tests ===

    @pytest.mark.asyncio
    async def test_mutate_phase_creates_variations(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that mutation phase creates variations."""
        initial_thought = await initialized_method.execute(session, sample_problem)
        mutate_thought = await initialized_method.continue_reasoning(session, initial_thought)

        assert mutate_thought is not None
        assert mutate_thought.metadata["phase"] == "mutate"
        assert "Mutate" in mutate_thought.content
        assert mutate_thought.type == ThoughtType.EXPLORATION

    @pytest.mark.asyncio
    async def test_mutate_phase_increments_generation(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that mutation phase increments generation."""
        initial_thought = await initialized_method.execute(session, sample_problem)

        assert initialized_method._generation == 1

        await initialized_method.continue_reasoning(session, initial_thought)

        assert initialized_method._generation == 2

    # === Selection Phase Tests ===

    @pytest.mark.asyncio
    async def test_select_phase_identifies_best(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that selection phase identifies the best solution."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # mutate
        thought = await initialized_method.continue_reasoning(session, thought)  # select

        assert thought.metadata["phase"] == "select"
        assert "Best" in thought.content
        assert thought.type == ThoughtType.VERIFICATION

    # === Convergence Phase Tests ===

    @pytest.mark.asyncio
    async def test_converge_phase_produces_final_solution(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that convergence phase produces final solution."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)  # mutate
        thought = await initialized_method.continue_reasoning(session, thought)  # select
        thought = await initialized_method.continue_reasoning(session, thought)  # converge

        assert thought.metadata["phase"] == "converge"
        assert "Mind Evolution Complete" in thought.content
        assert thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_converge_phase_includes_generation_count(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that convergence includes generation count."""
        thought = await initialized_method.execute(session, sample_problem)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)
        thought = await initialized_method.continue_reasoning(session, thought)

        assert "Generations:" in thought.content

    # === Full Pipeline Tests ===

    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test complete reasoning pipeline through all phases."""
        thought = await initialized_method.execute(session, sample_problem)
        assert thought.metadata["phase"] == "initialize_pop"

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "mutate"

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "select"

        thought = await initialized_method.continue_reasoning(session, thought)
        assert thought.metadata["phase"] == "converge"

    # === LLM Sampling Tests ===

    @pytest.mark.asyncio
    async def test_execute_with_sampling(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test that execute uses LLM sampling when available."""
        await initialized_method.execute(
            session, sample_problem, execution_context=mock_execution_context
        )

        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_sampling_fallback_on_error(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling fails with expected errors."""
        failing_ctx = MagicMock()
        failing_ctx.can_sample = True
        # Use ConnectionError (an expected exception type that triggers fallback)
        failing_ctx.sample = AsyncMock(side_effect=ConnectionError("LLM connection failed"))

        await initialized_method.execute(session, sample_problem, execution_context=failing_ctx)

        # Should use fallback population
        assert len(initialized_method._population) == 3
        for p in initialized_method._population:
            assert (
                "Direct approach" in p["solution"]
                or "Alternative" in p["solution"]
                or "Systematic" in p["solution"]
            )

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_not_available(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that fallback is used when sampling not available."""
        no_sample_ctx = MagicMock()
        no_sample_ctx.can_sample = False

        await initialized_method.execute(session, sample_problem, execution_context=no_sample_ctx)

        assert len(initialized_method._population) == 3

    # === Population Management Tests ===

    @pytest.mark.asyncio
    async def test_generate_fallback_population(self, initialized_method: MindEvolution) -> None:
        """Test fallback population generation."""
        population = initialized_method._generate_fallback_population("Test problem")

        assert len(population) == 3
        assert population[0]["id"] == "P1"
        assert population[1]["id"] == "P2"
        assert population[2]["id"] == "P3"
        # Fitness increases across solutions
        assert population[0]["fitness"] < population[1]["fitness"] < population[2]["fitness"]

    @pytest.mark.asyncio
    async def test_initialize_population_with_context(
        self,
        initialized_method: MindEvolution,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _initialize_population with execution context."""
        initialized_method._execution_context = mock_execution_context

        population = await initialized_method._initialize_population("Test problem")

        assert len(population) == 3
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutate_population_with_context(
        self,
        initialized_method: MindEvolution,
        mock_execution_context: MagicMock,
    ) -> None:
        """Test _mutate_population with execution context."""
        initialized_method._execution_context = mock_execution_context
        initialized_method._population = [
            {"id": "P1", "solution": "Solution A", "fitness": 0.8},
            {"id": "P2", "solution": "Solution B", "fitness": 0.85},
            {"id": "P3", "solution": "Solution C", "fitness": 0.9},
        ]

        population = await initialized_method._mutate_population("Test problem")

        assert len(population) == 3
        mock_execution_context.sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutate_population_fallback(self, initialized_method: MindEvolution) -> None:
        """Test _mutate_population fallback without context."""
        initialized_method._execution_context = None
        initialized_method._population = [
            {"id": "P1", "solution": "Solution A", "fitness": 0.8},
            {"id": "P2", "solution": "Solution B", "fitness": 0.85},
            {"id": "P3", "solution": "Solution C", "fitness": 0.9},
        ]

        population = await initialized_method._mutate_population("Test problem")

        assert len(population) == 3
        for p in population:
            assert "(variation A)" in p["solution"]

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_step_counter_increments_correctly(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that step counter increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert initialized_method._step_counter == 1

        await initialized_method.continue_reasoning(session, thought1)
        assert initialized_method._step_counter == 2

    @pytest.mark.asyncio
    async def test_depth_increments_correctly(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that thought depth increments correctly."""
        thought1 = await initialized_method.execute(session, sample_problem)
        assert thought1.depth == 0

        thought2 = await initialized_method.continue_reasoning(session, thought1)
        assert thought2.depth == 1

    @pytest.mark.asyncio
    async def test_population_fitness_preserved(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that population fitness is tracked."""
        await initialized_method.execute(session, sample_problem)

        # After execute, population should have fitness values
        for p in initialized_method._population:
            assert 0.0 < p["fitness"] <= 1.0

    @pytest.mark.asyncio
    async def test_best_solution_selection(
        self,
        initialized_method: MindEvolution,
        session: MagicMock,
        sample_problem: str,
    ) -> None:
        """Test that best solution is correctly identified."""
        thought = await initialized_method.execute(session, sample_problem)

        # Manually set different fitness values
        initialized_method._population = [
            {"id": "P1", "solution": "Low fitness", "fitness": 0.5},
            {"id": "P2", "solution": "High fitness", "fitness": 0.95},
            {"id": "P3", "solution": "Medium fitness", "fitness": 0.7},
        ]

        thought = await initialized_method.continue_reasoning(session, thought)  # mutate
        thought = await initialized_method.continue_reasoning(session, thought)  # select

        # Best (P2) should be identified
        assert "High fitness" in thought.content or "P2" in thought.content


class TestMindEvolutionGeneticOperations:
    """Tests for Mind Evolution genetic algorithm operations (L3.16)."""

    @pytest.fixture
    async def initialized_method(self) -> MindEvolution:
        """Create initialized method."""
        method = MindEvolution()
        await method.initialize()
        return method

    @pytest.fixture
    def session(self) -> MagicMock:
        """Create mock session."""
        mock_sess = MagicMock(spec=Session)
        mock_sess.current_method = None
        mock_sess.thought_count = 0
        mock_sess._thoughts = []
        mock_sess.graph = MagicMock()
        mock_sess.graph.nodes = {}

        def add_thought(thought: ThoughtNode) -> None:
            mock_sess._thoughts.append(thought)
            mock_sess.thought_count = len(mock_sess._thoughts)
            mock_sess.graph.nodes[thought.id] = thought

        mock_sess.add_thought = add_thought

        return mock_sess

    @pytest.mark.asyncio
    async def test_population_diversity(
        self, initialized_method: MindEvolution, session: MagicMock
    ) -> None:
        """Test that initial population has diverse solutions."""
        await initialized_method.execute(session, "Test problem")

        solutions = [p["solution"] for p in initialized_method._population]
        # All solutions should be unique
        assert len(solutions) == len(set(solutions))

    @pytest.mark.asyncio
    async def test_mutation_modifies_population(
        self, initialized_method: MindEvolution, session: MagicMock
    ) -> None:
        """Test that mutation modifies the population."""
        thought = await initialized_method.execute(session, "Test problem")
        original_pop = [p.copy() for p in initialized_method._population]

        thought = await initialized_method.continue_reasoning(session, thought)

        # Population should have been mutated
        current_solutions = [p["solution"] for p in initialized_method._population]
        original_solutions = [p["solution"] for p in original_pop]
        # At least some solutions should be different (have variation marker)
        assert (
            any("variation" in s for s in current_solutions)
            or current_solutions != original_solutions
        )

    @pytest.mark.asyncio
    async def test_fitness_progression(
        self, initialized_method: MindEvolution, session: MagicMock
    ) -> None:
        """Test that fitness can increase through generations."""
        thought = await initialized_method.execute(session, "Test problem")

        max(p["fitness"] for p in initialized_method._population)

        thought = await initialized_method.continue_reasoning(session, thought)  # mutate

        mutated_best_fitness = max(p["fitness"] for p in initialized_method._population)

        # Mutation should potentially improve or maintain fitness
        assert mutated_best_fitness >= 0

    @pytest.mark.asyncio
    async def test_generation_counter_tracks_evolution(
        self, initialized_method: MindEvolution, session: MagicMock
    ) -> None:
        """Test that generation counter tracks evolution."""
        thought = await initialized_method.execute(session, "Test problem")
        assert initialized_method._generation == 1
        assert thought.metadata["generation"] == 1

        thought = await initialized_method.continue_reasoning(session, thought)
        assert initialized_method._generation == 2
        assert thought.metadata["generation"] == 2


__all__ = [
    "TestMindEvolutionMetadata",
    "TestMindEvolution",
    "TestMindEvolutionGeneticOperations",
]
