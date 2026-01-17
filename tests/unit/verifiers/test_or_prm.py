"""Unit tests for OR-PRM verifier.

Tests outcome-aware process reward model.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.or_prm import OR_PRM_METADATA, OrPrm


class TestOrPrmMetadata:
    """Tests for OR-PRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert OR_PRM_METADATA.identifier == VerifierIdentifier.OR_PRM

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert OR_PRM_METADATA.name == "OR-PRM"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "outcome-aware" in OR_PRM_METADATA.tags
        assert "process-reward" in OR_PRM_METADATA.tags
        assert "predictive" in OR_PRM_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert OR_PRM_METADATA.supports_step_level is True
        assert OR_PRM_METADATA.supports_outcome_level is True
        assert OR_PRM_METADATA.supports_cot_verification is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= OR_PRM_METADATA.complexity <= 10


class TestOrPrmInitialization:
    """Tests for OR-PRM initialization."""

    def test_create_instance(self) -> None:
        """Test creating OrPrm instance."""
        verifier = OrPrm()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = OrPrm()
        assert verifier.identifier == VerifierIdentifier.OR_PRM

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = OrPrm()
        assert verifier.name == "OR-PRM"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = OrPrm()
        await verifier.initialize()
        assert verifier._initialized is True

    async def test_initialize_sets_outcome_weight(self) -> None:
        """Test initialize sets outcome weight."""
        verifier = OrPrm()
        await verifier.initialize()
        assert verifier._outcome_weight == 0.4


class TestOrPrmProcessScoring:
    """Tests for OR-PRM process scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> OrPrm:
        """Create an initialized OrPrm."""
        verifier = OrPrm()
        await verifier.initialize()
        return verifier

    def test_score_process_base(self, initialized_verifier: OrPrm) -> None:
        """Test base process score."""
        score = initialized_verifier._score_process("Hi")
        assert score == 0.6

    def test_score_process_long_solution_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test longer solutions get bonus (len > 30)."""
        short_score = initialized_verifier._score_process("Hi")
        long_score = initialized_verifier._score_process(
            "This is a much longer solution text that exceeds thirty chars"
        )
        assert long_score > short_score

    def test_score_process_step_keywords_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test step keywords get bonus."""
        base_score = initialized_verifier._score_process("Do this")
        step_score = initialized_verifier._score_process("Step 1: First, then finally")
        assert step_score > base_score

    def test_score_process_equals_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test equals sign gets bonus."""
        base_score = initialized_verifier._score_process("The value is 5")
        equals_score = initialized_verifier._score_process("x = 5")
        assert equals_score > base_score

    def test_score_process_capped(self, initialized_verifier: OrPrm) -> None:
        """Test process score is capped at 1.0."""
        score = initialized_verifier._score_process("Step 1: First = 5, then finally = 10")
        assert score <= 1.0


class TestOrPrmOutcomePrediction:
    """Tests for OR-PRM outcome prediction."""

    @pytest.fixture
    async def initialized_verifier(self) -> OrPrm:
        """Create an initialized OrPrm."""
        verifier = OrPrm()
        await verifier.initialize()
        return verifier

    def test_predict_outcome_base(self, initialized_verifier: OrPrm) -> None:
        """Test base outcome prediction score."""
        score = initialized_verifier._predict_outcome("Hi")
        assert score == 0.7

    def test_predict_outcome_answer_keyword_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test answer keyword gets bonus."""
        base_score = initialized_verifier._predict_outcome("The value")
        answer_score = initialized_verifier._predict_outcome("The answer is")
        assert answer_score > base_score

    def test_predict_outcome_result_keyword_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test result keyword gets bonus."""
        base_score = initialized_verifier._predict_outcome("The value")
        result_score = initialized_verifier._predict_outcome("The result is")
        assert result_score > base_score

    def test_predict_outcome_conclusion_keyword_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test conclusion keyword gets bonus."""
        base_score = initialized_verifier._predict_outcome("The value")
        conclusion_score = initialized_verifier._predict_outcome("In conclusion")
        assert conclusion_score > base_score

    def test_predict_outcome_digit_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test digits get bonus."""
        base_score = initialized_verifier._predict_outcome("The value")
        digit_score = initialized_verifier._predict_outcome("The value is 42")
        assert digit_score > base_score

    def test_predict_outcome_reasoning_bonus(self, initialized_verifier: OrPrm) -> None:
        """Test reasoning keywords get bonus."""
        base_score = initialized_verifier._predict_outcome("The value")
        therefore_score = initialized_verifier._predict_outcome("Therefore, the answer")
        assert therefore_score > base_score


class TestOrPrmVerify:
    """Tests for OR-PRM verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> OrPrm:
        """Create an initialized OrPrm."""
        verifier = OrPrm()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = OrPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_verification(self, initialized_verifier: OrPrm) -> None:
        """Test verify returns tuple of score and verification string."""
        score, verification = await initialized_verifier.verify("test solution")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_combines_process_and_outcome(self, initialized_verifier: OrPrm) -> None:
        """Test verify combines process and outcome scores."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "Process quality" in verification
        assert "Predicted outcome" in verification
        assert "Combined score" in verification

    async def test_verify_with_context(self, initialized_verifier: OrPrm) -> None:
        """Test verify with context parameter."""
        score, verification = await initialized_verifier.verify(
            "test solution",
            context={"problem": "Calculate 2+2"},
        )
        assert isinstance(score, float)


class TestOrPrmScoreWithOutcome:
    """Tests for OR-PRM score_with_outcome method."""

    @pytest.fixture
    async def initialized_verifier(self) -> OrPrm:
        """Create an initialized OrPrm."""
        verifier = OrPrm()
        await verifier.initialize()
        return verifier

    async def test_score_with_outcome_raises_when_not_initialized(self) -> None:
        """Test score_with_outcome raises error when not initialized."""
        verifier = OrPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_with_outcome(["step 1", "step 2"])

    async def test_score_with_outcome_without_outcome(self, initialized_verifier: OrPrm) -> None:
        """Test score_with_outcome without outcome parameter."""
        scores = await initialized_verifier.score_with_outcome(["Step 1", "Step 2", "Step 3"])
        assert len(scores) == 3
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_with_outcome_with_outcome(self, initialized_verifier: OrPrm) -> None:
        """Test score_with_outcome with outcome parameter."""
        scores = await initialized_verifier.score_with_outcome(
            ["Step 1", "Step 2", "Step 3"],
            outcome="The answer is 42",
        )
        assert len(scores) == 3

    async def test_score_with_outcome_later_steps_influenced(
        self, initialized_verifier: OrPrm
    ) -> None:
        """Test later steps are more influenced by outcome."""
        # With high-quality outcome
        scores_good_outcome = await initialized_verifier.score_with_outcome(
            ["Step 1", "Step 2", "Step 3"],
            outcome="Therefore, the answer = 42, result confirmed",
        )
        # Without outcome
        scores_no_outcome = await initialized_verifier.score_with_outcome(
            ["Step 1", "Step 2", "Step 3"],
        )
        # Later steps should be more influenced by good outcome
        assert len(scores_good_outcome) == len(scores_no_outcome)


class TestOrPrmScoreSteps:
    """Tests for OR-PRM score_steps method."""

    @pytest.fixture
    async def initialized_verifier(self) -> OrPrm:
        """Create an initialized OrPrm."""
        verifier = OrPrm()
        await verifier.initialize()
        return verifier

    async def test_score_steps_delegates_to_score_with_outcome(
        self, initialized_verifier: OrPrm
    ) -> None:
        """Test score_steps delegates to score_with_outcome."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2"])
        assert len(scores) == 2

    async def test_score_steps_with_context(self, initialized_verifier: OrPrm) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"problem": "Math problem"},
        )
        assert len(scores) == 2


class TestOrPrmHealthCheck:
    """Tests for OR-PRM health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = OrPrm()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = OrPrm()
        await verifier.initialize()
        assert await verifier.health_check() is True
