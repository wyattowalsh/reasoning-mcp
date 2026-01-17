"""Unit tests for BestOfN reasoning method."""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.best_of_n import BEST_OF_N_METADATA, BestOfN
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def best_of_n_method() -> BestOfN:
    """Create a BestOfN method instance for testing.

    Returns:
        A fresh BestOfN instance
    """
    return BestOfN()


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
        A sample problem string
    """
    return "What is the capital of France?"


class TestBestOfNInitialization:
    """Tests for BestOfN initialization and setup."""

    def test_create_method(self, best_of_n_method: BestOfN):
        """Test that BestOfN can be instantiated."""
        assert best_of_n_method is not None
        assert isinstance(best_of_n_method, BestOfN)

    def test_initial_state(self, best_of_n_method: BestOfN):
        """Test that a new method starts in the correct initial state."""
        assert best_of_n_method._initialized is False
        assert best_of_n_method._step_counter == 0
        assert best_of_n_method._current_phase == "sample"

    async def test_initialize(self, best_of_n_method: BestOfN):
        """Test that initialize() sets up the method correctly."""
        await best_of_n_method.initialize()
        assert best_of_n_method._initialized is True
        assert best_of_n_method._step_counter == 0

    async def test_health_check_not_initialized(self, best_of_n_method: BestOfN):
        """Test that health_check returns False before initialization."""
        result = await best_of_n_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, best_of_n_method: BestOfN):
        """Test that health_check returns True after initialization."""
        await best_of_n_method.initialize()
        result = await best_of_n_method.health_check()
        assert result is True


class TestBestOfNProperties:
    """Tests for BestOfN property accessors."""

    def test_identifier_property(self, best_of_n_method: BestOfN):
        """Test that identifier returns the correct method identifier."""
        assert best_of_n_method.identifier == MethodIdentifier.BEST_OF_N

    def test_name_property(self, best_of_n_method: BestOfN):
        """Test that name returns the correct human-readable name."""
        assert best_of_n_method.name == "Best-of-N"

    def test_description_property(self, best_of_n_method: BestOfN):
        """Test that description returns the correct method description."""
        assert "sample" in best_of_n_method.description.lower()
        assert "best" in best_of_n_method.description.lower()

    def test_category_property(self, best_of_n_method: BestOfN):
        """Test that category returns the correct method category."""
        assert best_of_n_method.category == MethodCategory.CORE


class TestBestOfNMetadata:
    """Tests for BestOfN metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert BEST_OF_N_METADATA.identifier == MethodIdentifier.BEST_OF_N

    def test_metadata_category(self):
        """Test that metadata has the correct category."""
        assert BEST_OF_N_METADATA.category == MethodCategory.CORE

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"sampling", "selection", "reward-model"}
        assert expected_tags.issubset(BEST_OF_N_METADATA.tags)


class TestBestOfNExecution:
    """Tests for BestOfN execute() method."""

    async def test_execute_basic(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test basic execution creates a thought."""
        await best_of_n_method.initialize()
        thought = await best_of_n_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.BEST_OF_N

    async def test_execute_without_initialization_raises(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute raises error if not initialized."""
        with pytest.raises(RuntimeError):
            await best_of_n_method.execute(session, sample_problem)

    async def test_execute_creates_initial_thought(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates an INITIAL thought type."""
        await best_of_n_method.initialize()
        thought = await best_of_n_method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_generates_samples(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute generates N samples."""
        await best_of_n_method.initialize()
        thought = await best_of_n_method.execute(session, sample_problem)

        assert len(best_of_n_method._samples) == best_of_n_method._n
        assert "samples" in thought.metadata
        assert thought.metadata["samples"] == best_of_n_method._n


class TestBestOfNContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_scores_samples(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test that first continuation scores the samples."""
        await best_of_n_method.initialize()
        initial = await best_of_n_method.execute(session, sample_problem)

        continuation = await best_of_n_method.continue_reasoning(session, initial)

        assert continuation is not None
        assert len(best_of_n_method._scores) == len(best_of_n_method._samples)
        assert continuation.metadata["phase"] == "score"

    async def test_continue_selects_best(
        self,
        best_of_n_method: BestOfN,
        session: Session,
        sample_problem: str,
    ):
        """Test that second continuation selects the best sample."""
        await best_of_n_method.initialize()
        initial = await best_of_n_method.execute(session, sample_problem)
        score = await best_of_n_method.continue_reasoning(session, initial)

        select = await best_of_n_method.continue_reasoning(session, score)

        assert select is not None
        assert select.metadata["phase"] == "select"
        assert best_of_n_method._best_idx >= 0
        assert best_of_n_method._best_idx < len(best_of_n_method._samples)


class TestBestOfNHelperMethods:
    """Tests for helper methods."""

    async def test_generate_candidates_fallback(self, best_of_n_method: BestOfN):
        """Test that _generate_candidates creates fallback candidates."""
        candidates = best_of_n_method._generate_candidates("Test problem", 3)

        assert len(candidates) == 3
        assert all("id" in c for c in candidates)
        assert all("answer" in c for c in candidates)
        assert all("reasoning" in c for c in candidates)

    async def test_generate_scores_fallback(self, best_of_n_method: BestOfN):
        """Test that _generate_scores creates fallback scores."""
        samples = [
            {"id": 1, "answer": "A1", "reasoning": "R1"},
            {"id": 2, "answer": "A2", "reasoning": "R2"},
        ]
        scores = best_of_n_method._generate_scores(samples)

        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
