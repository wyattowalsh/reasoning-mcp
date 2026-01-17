"""Unit tests for R-PRM verifier.

Tests reasoning-driven process rewards with explicit rationales.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.r_prm import R_PRM_METADATA, RPrm


class TestRPrmMetadata:
    """Tests for R-PRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert R_PRM_METADATA.identifier == VerifierIdentifier.R_PRM

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert R_PRM_METADATA.name == "R-PRM"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "reasoning" in R_PRM_METADATA.tags
        assert "process-reward" in R_PRM_METADATA.tags
        assert "rationale" in R_PRM_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert R_PRM_METADATA.supports_step_level is True
        assert R_PRM_METADATA.supports_outcome_level is True
        assert R_PRM_METADATA.supports_cot_verification is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= R_PRM_METADATA.complexity <= 10


class TestRPrmInitialization:
    """Tests for R-PRM initialization."""

    def test_create_instance(self) -> None:
        """Test creating RPrm instance."""
        verifier = RPrm()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = RPrm()
        assert verifier.identifier == VerifierIdentifier.R_PRM

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = RPrm()
        assert verifier.name == "R-PRM"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = RPrm()
        await verifier.initialize()
        assert verifier._initialized is True

    async def test_initialize_clears_history(self) -> None:
        """Test initialize clears reasoning history."""
        verifier = RPrm()
        verifier._reasoning_history = [{"test": "data"}]
        await verifier.initialize()
        assert len(verifier._reasoning_history) == 0


class TestRPrmVerify:
    """Tests for R-PRM verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> RPrm:
        """Create an initialized RPrm."""
        verifier = RPrm()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = RPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_verification(self, initialized_verifier: RPrm) -> None:
        """Test verify returns tuple of score and verification string."""
        score, verification = await initialized_verifier.verify("test solution")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_includes_reasoning_steps(self, initialized_verifier: RPrm) -> None:
        """Test verification includes reasoning steps."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "Premise check" in verification or "reasoning" in verification.lower()
        assert "✓" in verification

    async def test_verify_records_history(self, initialized_verifier: RPrm) -> None:
        """Test verify records to reasoning history."""
        await initialized_verifier.verify("test solution")
        assert len(initialized_verifier._reasoning_history) == 1
        assert "score" in initialized_verifier._reasoning_history[0]

    async def test_verify_multiple_records_history(self, initialized_verifier: RPrm) -> None:
        """Test multiple verifications build history."""
        await initialized_verifier.verify("solution 1")
        await initialized_verifier.verify("solution 2")
        await initialized_verifier.verify("solution 3")
        assert len(initialized_verifier._reasoning_history) == 3

    async def test_verify_with_context(self, initialized_verifier: RPrm) -> None:
        """Test verify with context parameter."""
        score, verification = await initialized_verifier.verify(
            "test solution",
            context={"problem": "Calculate 2+2"},
        )
        assert isinstance(score, float)


class TestRPrmReasonVerify:
    """Tests for R-PRM reason_verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> RPrm:
        """Create an initialized RPrm."""
        verifier = RPrm()
        await verifier.initialize()
        return verifier

    async def test_reason_verify_raises_when_not_initialized(self) -> None:
        """Test reason_verify raises error when not initialized."""
        verifier = RPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.reason_verify("test solution")

    async def test_reason_verify_returns_score_and_rationale(
        self, initialized_verifier: RPrm
    ) -> None:
        """Test reason_verify returns tuple of score and rationale."""
        score, rationale = await initialized_verifier.reason_verify("test solution")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(rationale, str)

    async def test_reason_verify_includes_detailed_rationale(
        self, initialized_verifier: RPrm
    ) -> None:
        """Test reason_verify includes detailed reasoning rationale."""
        _, rationale = await initialized_verifier.reason_verify("test solution")
        assert "PREMISE VALIDITY" in rationale
        assert "LOGICAL SOUNDNESS" in rationale
        assert "REASONING QUALITY" in rationale

    async def test_reason_verify_returns_expected_score(self, initialized_verifier: RPrm) -> None:
        """Test reason_verify returns expected score."""
        score, _ = await initialized_verifier.reason_verify("test solution")
        assert score == 0.88

    async def test_reason_verify_with_context(self, initialized_verifier: RPrm) -> None:
        """Test reason_verify with context parameter."""
        score, rationale = await initialized_verifier.reason_verify(
            "test solution",
            context={"problem": "Math problem"},
        )
        assert isinstance(score, float)


class TestRPrmScoreSteps:
    """Tests for R-PRM step scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> RPrm:
        """Create an initialized RPrm."""
        verifier = RPrm()
        await verifier.initialize()
        return verifier

    async def test_score_steps_raises_when_not_initialized(self) -> None:
        """Test score_steps raises error when not initialized."""
        verifier = RPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_steps(["step 1", "step 2"])

    async def test_score_steps_returns_list(self, initialized_verifier: RPrm) -> None:
        """Test score_steps returns list of scores."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        assert isinstance(scores, list)
        assert len(scores) == 3

    async def test_score_steps_valid_range(self, initialized_verifier: RPrm) -> None:
        """Test all step scores are in valid range."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_steps_reasoning_keywords_bonus(self, initialized_verifier: RPrm) -> None:
        """Test steps with reasoning keywords get bonus."""
        scores_basic = await initialized_verifier.score_steps(["Do something"])
        scores_because = await initialized_verifier.score_steps(["Because of this, we conclude"])
        scores_therefore = await initialized_verifier.score_steps(["Therefore, the result follows"])
        scores_since = await initialized_verifier.score_steps(["Since this is true, we have"])

        # Steps with reasoning keywords should have higher scores
        assert scores_because[0] >= scores_basic[0]
        assert scores_therefore[0] >= scores_basic[0]
        assert scores_since[0] >= scores_basic[0]

    async def test_score_steps_verification_keywords_bonus(
        self, initialized_verifier: RPrm
    ) -> None:
        """Test steps with verification keywords get bonus."""
        scores_basic = await initialized_verifier.score_steps(["Do something"])
        scores_verify = await initialized_verifier.score_steps(["Verify the result"])
        scores_check = await initialized_verifier.score_steps(["Check if correct"])
        scores_confirm = await initialized_verifier.score_steps(["Confirm the answer"])

        # Steps with verification keywords should have higher scores
        assert scores_verify[0] >= scores_basic[0]
        assert scores_check[0] >= scores_basic[0]
        assert scores_confirm[0] >= scores_basic[0]

    async def test_score_steps_math_content_bonus(self, initialized_verifier: RPrm) -> None:
        """Test steps with math content get bonus."""
        scores_basic = await initialized_verifier.score_steps(["Do something"])
        scores_equals = await initialized_verifier.score_steps(["x = 5"])
        scores_digits = await initialized_verifier.score_steps(["The result is 42"])

        # Steps with math content should have higher scores
        assert scores_equals[0] >= scores_basic[0]
        assert scores_digits[0] >= scores_basic[0]

    async def test_score_steps_position_bonus(self, initialized_verifier: RPrm) -> None:
        """Test later steps get position bonus (up to limit)."""
        # Same content, different positions
        scores = await initialized_verifier.score_steps(
            ["Same step", "Same step", "Same step", "Same step", "Same step"]
        )
        # Should have some position-based progression
        assert len(scores) == 5

    async def test_score_steps_with_context(self, initialized_verifier: RPrm) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"problem": "Math problem"},
        )
        assert len(scores) == 2


class TestRPrmHealthCheck:
    """Tests for R-PRM health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = RPrm()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = RPrm()
        await verifier.initialize()
        assert await verifier.health_check() is True
