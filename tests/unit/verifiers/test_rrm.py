"""Unit tests for RRM verifier.

Tests reward reasoning models with deliberative rationales.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.rrm import RRM_METADATA, Rrm


class TestRrmMetadata:
    """Tests for RRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert RRM_METADATA.identifier == VerifierIdentifier.RRM

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert RRM_METADATA.name == "RRM"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "reward" in RRM_METADATA.tags
        assert "reasoning" in RRM_METADATA.tags
        assert "deliberative" in RRM_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert RRM_METADATA.supports_step_level is True
        assert RRM_METADATA.supports_outcome_level is True
        assert RRM_METADATA.supports_cot_verification is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= RRM_METADATA.complexity <= 10


class TestRrmInitialization:
    """Tests for RRM initialization."""

    def test_create_instance(self) -> None:
        """Test creating Rrm instance."""
        verifier = Rrm()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = Rrm()
        assert verifier.identifier == VerifierIdentifier.RRM

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = Rrm()
        assert verifier.name == "RRM"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = Rrm()
        await verifier.initialize()
        assert verifier._initialized is True

    async def test_initialize_sets_deliberation_depth(self) -> None:
        """Test initialize sets deliberation depth."""
        verifier = Rrm()
        await verifier.initialize()
        assert verifier._deliberation_depth == 3


class TestRrmDeliberation:
    """Tests for RRM deliberation logic."""

    @pytest.fixture
    async def initialized_verifier(self) -> Rrm:
        """Create an initialized Rrm."""
        verifier = Rrm()
        await verifier.initialize()
        return verifier

    def test_deliberate_returns_string(self, initialized_verifier: Rrm) -> None:
        """Test deliberation returns string."""
        result = initialized_verifier._deliberate("test solution")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deliberate_includes_rounds(self, initialized_verifier: Rrm) -> None:
        """Test deliberation includes multiple rounds."""
        result = initialized_verifier._deliberate("test solution")
        # Should have rounds based on deliberation depth
        assert "Round 1" in result
        assert "Round 2" in result
        assert "Round 3" in result

    def test_deliberate_includes_aspects(self, initialized_verifier: Rrm) -> None:
        """Test deliberation includes aspect considerations."""
        result = initialized_verifier._deliberate("test solution")
        assert "aspect" in result.lower()

    def test_deliberate_includes_rewards(self, initialized_verifier: Rrm) -> None:
        """Test deliberation includes intermediate rewards."""
        result = initialized_verifier._deliberate("test solution")
        assert "reward" in result.lower()


class TestRrmVerify:
    """Tests for RRM verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> Rrm:
        """Create an initialized Rrm."""
        verifier = Rrm()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = Rrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_verification(self, initialized_verifier: Rrm) -> None:
        """Test verify returns tuple of score and verification string."""
        score, verification = await initialized_verifier.verify("test solution")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_includes_deliberation(self, initialized_verifier: Rrm) -> None:
        """Test verification includes deliberation process."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "Reward Reasoning Verification" in verification
        assert "Deliberation depth" in verification

    async def test_verify_returns_expected_score(self, initialized_verifier: Rrm) -> None:
        """Test verify returns expected score."""
        score, _ = await initialized_verifier.verify("test solution")
        assert score == 0.89

    async def test_verify_with_context(self, initialized_verifier: Rrm) -> None:
        """Test verify with context parameter."""
        score, verification = await initialized_verifier.verify(
            "test solution",
            context={"problem": "Calculate 2+2"},
        )
        assert isinstance(score, float)


class TestRrmDeliberateReward:
    """Tests for RRM deliberate_reward method."""

    @pytest.fixture
    async def initialized_verifier(self) -> Rrm:
        """Create an initialized Rrm."""
        verifier = Rrm()
        await verifier.initialize()
        return verifier

    async def test_deliberate_reward_raises_when_not_initialized(self) -> None:
        """Test deliberate_reward raises error when not initialized."""
        verifier = Rrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.deliberate_reward("test output")

    async def test_deliberate_reward_returns_score_and_rationale(
        self, initialized_verifier: Rrm
    ) -> None:
        """Test deliberate_reward returns tuple of score and rationale."""
        score, rationale = await initialized_verifier.deliberate_reward("test output")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(rationale, str)

    async def test_deliberate_reward_includes_phases(self, initialized_verifier: Rrm) -> None:
        """Test deliberate_reward includes multiple phases."""
        _, rationale = await initialized_verifier.deliberate_reward("test output")
        assert "PHASE 1" in rationale
        assert "PHASE 2" in rationale
        assert "PHASE 3" in rationale

    async def test_deliberate_reward_includes_assessments(self, initialized_verifier: Rrm) -> None:
        """Test deliberate_reward includes various assessments."""
        _, rationale = await initialized_verifier.deliberate_reward("test output")
        assert "coherence" in rationale.lower() or "Initial Assessment" in rationale
        assert "Deep Analysis" in rationale
        assert "Final Deliberation" in rationale

    async def test_deliberate_reward_returns_expected_score(
        self, initialized_verifier: Rrm
    ) -> None:
        """Test deliberate_reward returns expected score."""
        score, _ = await initialized_verifier.deliberate_reward("test output")
        assert score == 0.87

    async def test_deliberate_reward_with_context(self, initialized_verifier: Rrm) -> None:
        """Test deliberate_reward with context parameter."""
        score, rationale = await initialized_verifier.deliberate_reward(
            "test output",
            context={"problem": "Math problem"},
        )
        assert isinstance(score, float)


class TestRrmScoreSteps:
    """Tests for RRM step scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> Rrm:
        """Create an initialized Rrm."""
        verifier = Rrm()
        await verifier.initialize()
        return verifier

    async def test_score_steps_raises_when_not_initialized(self) -> None:
        """Test score_steps raises error when not initialized."""
        verifier = Rrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_steps(["step 1", "step 2"])

    async def test_score_steps_returns_list(self, initialized_verifier: Rrm) -> None:
        """Test score_steps returns list of scores."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        assert isinstance(scores, list)
        assert len(scores) == 3

    async def test_score_steps_valid_range(self, initialized_verifier: Rrm) -> None:
        """Test all step scores are in valid range."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_steps_length_bonus(self, initialized_verifier: Rrm) -> None:
        """Test longer steps get quality bonus."""
        scores_short = await initialized_verifier.score_steps(["Hi"])
        scores_long = await initialized_verifier.score_steps(
            ["This is a much longer step with more content"]
        )
        assert scores_long[0] >= scores_short[0]

    async def test_score_steps_reasoning_keywords_bonus(self, initialized_verifier: Rrm) -> None:
        """Test steps with reasoning keywords get bonus."""
        scores_basic = await initialized_verifier.score_steps(["Do something"])
        scores_because = await initialized_verifier.score_steps(["Because of this, we conclude"])
        scores_therefore = await initialized_verifier.score_steps(["Therefore, the result follows"])
        scores_thus = await initialized_verifier.score_steps(["Thus, we can deduce that"])

        # Steps with reasoning keywords should have higher scores
        assert scores_because[0] >= scores_basic[0]
        assert scores_therefore[0] >= scores_basic[0]
        assert scores_thus[0] >= scores_basic[0]

    async def test_score_steps_cumulative_quality(self, initialized_verifier: Rrm) -> None:
        """Test steps build cumulative quality."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3", "Step 4"])
        # Later steps should potentially benefit from cumulative quality
        assert len(scores) == 4

    async def test_score_steps_with_context(self, initialized_verifier: Rrm) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"problem": "Math problem"},
        )
        assert len(scores) == 2


class TestRrmHealthCheck:
    """Tests for RRM health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = Rrm()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = Rrm()
        await verifier.initialize()
        assert await verifier.health_check() is True
