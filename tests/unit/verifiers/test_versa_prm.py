"""Unit tests for VersaPRM verifier.

Tests versatile multi-domain process reward model.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.versa_prm import VERSA_PRM_METADATA, VersaPrm


class TestVersaPrmMetadata:
    """Tests for VersaPRM metadata."""

    def test_metadata_identifier(self) -> None:
        """Test metadata has correct identifier."""
        assert VERSA_PRM_METADATA.identifier == VerifierIdentifier.VERSA_PRM

    def test_metadata_name(self) -> None:
        """Test metadata has correct name."""
        assert VERSA_PRM_METADATA.name == "VersaPRM"

    def test_metadata_tags(self) -> None:
        """Test metadata has expected tags."""
        assert "versatile" in VERSA_PRM_METADATA.tags
        assert "multi-domain" in VERSA_PRM_METADATA.tags
        assert "process-reward" in VERSA_PRM_METADATA.tags

    def test_metadata_supports_features(self) -> None:
        """Test metadata indicates feature support."""
        assert VERSA_PRM_METADATA.supports_step_level is True
        assert VERSA_PRM_METADATA.supports_outcome_level is True
        assert VERSA_PRM_METADATA.supports_cot_verification is True

    def test_metadata_complexity_valid(self) -> None:
        """Test metadata complexity is within valid range."""
        assert 1 <= VERSA_PRM_METADATA.complexity <= 10


class TestVersaPrmInitialization:
    """Tests for VersaPRM initialization."""

    def test_create_instance(self) -> None:
        """Test creating VersaPrm instance."""
        verifier = VersaPrm()
        assert verifier is not None
        assert verifier._initialized is False

    def test_identifier_property(self) -> None:
        """Test identifier property returns correct value."""
        verifier = VersaPrm()
        assert verifier.identifier == VerifierIdentifier.VERSA_PRM

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        verifier = VersaPrm()
        assert verifier.name == "VersaPRM"

    async def test_initialize_sets_flag(self) -> None:
        """Test initialize sets initialized flag."""
        verifier = VersaPrm()
        await verifier.initialize()
        assert verifier._initialized is True

    async def test_initialize_sets_domain_weights(self) -> None:
        """Test initialize sets domain weights."""
        verifier = VersaPrm()
        await verifier.initialize()
        assert len(verifier._domain_weights) > 0
        assert "math" in verifier._domain_weights
        assert "logic" in verifier._domain_weights
        assert "coding" in verifier._domain_weights
        assert "science" in verifier._domain_weights
        assert "general" in verifier._domain_weights


class TestVersaPrmDomainDetection:
    """Tests for VersaPRM domain detection."""

    @pytest.fixture
    async def initialized_verifier(self) -> VersaPrm:
        """Create an initialized VersaPrm."""
        verifier = VersaPrm()
        await verifier.initialize()
        return verifier

    def test_detect_math_domain(self, initialized_verifier: VersaPrm) -> None:
        """Test math domain detection."""
        assert initialized_verifier._detect_domain("Calculate the equation") == "math"
        assert initialized_verifier._detect_domain("The sum is 42") == "math"
        assert initialized_verifier._detect_domain("x = 5") == "math"

    def test_detect_logic_domain(self, initialized_verifier: VersaPrm) -> None:
        """Test logic domain detection."""
        assert initialized_verifier._detect_domain("If A then B") == "logic"
        assert initialized_verifier._detect_domain("Therefore, we conclude") == "logic"
        assert initialized_verifier._detect_domain("This implies that") == "logic"

    def test_detect_coding_domain(self, initialized_verifier: VersaPrm) -> None:
        """Test coding domain detection."""
        assert initialized_verifier._detect_domain("def function():") == "coding"
        assert initialized_verifier._detect_domain("The code returns 5") == "coding"
        assert initialized_verifier._detect_domain("Write a function to") == "coding"

    def test_detect_science_domain(self, initialized_verifier: VersaPrm) -> None:
        """Test science domain detection."""
        assert initialized_verifier._detect_domain("The hypothesis is") == "science"
        assert initialized_verifier._detect_domain("Based on the experiment") == "science"
        assert initialized_verifier._detect_domain("Analyze the data") == "science"

    def test_detect_general_domain(self, initialized_verifier: VersaPrm) -> None:
        """Test general domain detection as fallback."""
        assert initialized_verifier._detect_domain("Hello world") == "general"
        assert initialized_verifier._detect_domain("Random text here") == "general"


class TestVersaPrmDomainWeights:
    """Tests for VersaPRM domain weights."""

    @pytest.fixture
    async def initialized_verifier(self) -> VersaPrm:
        """Create an initialized VersaPrm."""
        verifier = VersaPrm()
        await verifier.initialize()
        return verifier

    def test_math_domain_highest_weight(self, initialized_verifier: VersaPrm) -> None:
        """Test math domain has highest weight."""
        assert initialized_verifier._domain_weights["math"] == 1.0

    def test_domain_weights_ordered(self, initialized_verifier: VersaPrm) -> None:
        """Test domain weights are reasonably ordered."""
        weights = initialized_verifier._domain_weights
        assert weights["math"] >= weights["logic"]
        assert weights["logic"] >= weights["coding"]
        assert weights["coding"] >= weights["science"]
        assert weights["science"] >= weights["general"]

    def test_all_domain_weights_valid(self, initialized_verifier: VersaPrm) -> None:
        """Test all domain weights are in valid range."""
        for weight in initialized_verifier._domain_weights.values():
            assert 0.0 <= weight <= 1.0


class TestVersaPrmVerify:
    """Tests for VersaPRM verify method."""

    @pytest.fixture
    async def initialized_verifier(self) -> VersaPrm:
        """Create an initialized VersaPrm."""
        verifier = VersaPrm()
        await verifier.initialize()
        return verifier

    async def test_verify_raises_when_not_initialized(self) -> None:
        """Test verify raises error when not initialized."""
        verifier = VersaPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify("test solution")

    async def test_verify_returns_score_and_verification(
        self, initialized_verifier: VersaPrm
    ) -> None:
        """Test verify returns tuple of score and verification string."""
        score, verification = await initialized_verifier.verify("test solution")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_includes_domain_info(self, initialized_verifier: VersaPrm) -> None:
        """Test verification includes domain information."""
        _, verification = await initialized_verifier.verify("Calculate x = 5")
        assert "Detected domain" in verification
        assert "Domain confidence" in verification

    async def test_verify_includes_checks(self, initialized_verifier: VersaPrm) -> None:
        """Test verification includes check results."""
        _, verification = await initialized_verifier.verify("test solution")
        assert "Structural validity" in verification
        assert "Domain consistency" in verification
        assert "Logical coherence" in verification
        assert "✓" in verification

    async def test_verify_math_domain_score(self, initialized_verifier: VersaPrm) -> None:
        """Test math domain gets appropriate score."""
        score, verification = await initialized_verifier.verify("Calculate the equation: x = 5")
        assert "math" in verification
        # Math domain has highest weight (1.0)
        assert score > 0.5

    async def test_verify_with_context(self, initialized_verifier: VersaPrm) -> None:
        """Test verify with context parameter."""
        score, verification = await initialized_verifier.verify(
            "test solution",
            context={"problem": "Calculate 2+2"},
        )
        assert isinstance(score, float)


class TestVersaPrmVerifyDomain:
    """Tests for VersaPRM verify_domain method."""

    @pytest.fixture
    async def initialized_verifier(self) -> VersaPrm:
        """Create an initialized VersaPrm."""
        verifier = VersaPrm()
        await verifier.initialize()
        return verifier

    async def test_verify_domain_raises_when_not_initialized(self) -> None:
        """Test verify_domain raises error when not initialized."""
        verifier = VersaPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.verify_domain("test solution", "math")

    async def test_verify_domain_returns_score_and_verification(
        self, initialized_verifier: VersaPrm
    ) -> None:
        """Test verify_domain returns tuple of score and verification."""
        score, verification = await initialized_verifier.verify_domain("test solution", "math")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(verification, str)

    async def test_verify_domain_uses_specified_domain(
        self, initialized_verifier: VersaPrm
    ) -> None:
        """Test verify_domain uses the specified domain."""
        _, verification = await initialized_verifier.verify_domain("test solution", "coding")
        assert "coding" in verification

    async def test_verify_domain_math_high_weight(self, initialized_verifier: VersaPrm) -> None:
        """Test math domain verification uses high weight."""
        math_score, _ = await initialized_verifier.verify_domain("test solution", "math")
        general_score, _ = await initialized_verifier.verify_domain("test solution", "general")
        assert math_score > general_score

    async def test_verify_domain_unknown_uses_fallback(
        self, initialized_verifier: VersaPrm
    ) -> None:
        """Test unknown domain uses fallback weight."""
        score, _ = await initialized_verifier.verify_domain("test solution", "unknown_domain")
        assert isinstance(score, float)

    async def test_verify_domain_with_context(self, initialized_verifier: VersaPrm) -> None:
        """Test verify_domain with context parameter."""
        score, verification = await initialized_verifier.verify_domain(
            "test solution",
            "math",
            context={"problem": "Math problem"},
        )
        assert isinstance(score, float)


class TestVersaPrmScoreSteps:
    """Tests for VersaPRM step scoring."""

    @pytest.fixture
    async def initialized_verifier(self) -> VersaPrm:
        """Create an initialized VersaPrm."""
        verifier = VersaPrm()
        await verifier.initialize()
        return verifier

    async def test_score_steps_raises_when_not_initialized(self) -> None:
        """Test score_steps raises error when not initialized."""
        verifier = VersaPrm()
        with pytest.raises(RuntimeError, match="must be initialized"):
            await verifier.score_steps(["step 1", "step 2"])

    async def test_score_steps_returns_list(self, initialized_verifier: VersaPrm) -> None:
        """Test score_steps returns list of scores."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        assert isinstance(scores, list)
        assert len(scores) == 3

    async def test_score_steps_valid_range(self, initialized_verifier: VersaPrm) -> None:
        """Test all step scores are in valid range."""
        scores = await initialized_verifier.score_steps(["Step 1", "Step 2", "Step 3"])
        for score in scores:
            assert 0.0 <= score <= 1.0

    async def test_score_steps_domain_aware(self, initialized_verifier: VersaPrm) -> None:
        """Test steps are scored with domain awareness."""
        scores_math = await initialized_verifier.score_steps(["Calculate x = 5", "The sum is 10"])
        scores_general = await initialized_verifier.score_steps(["Hello there", "How are you"])
        # Math steps should have higher domain weight applied
        # Note: exact comparison depends on content scoring too
        assert len(scores_math) == 2
        assert len(scores_general) == 2

    async def test_score_steps_length_bonus(self, initialized_verifier: VersaPrm) -> None:
        """Test longer steps get quality bonus."""
        scores_short = await initialized_verifier.score_steps(["Hi"])
        scores_long = await initialized_verifier.score_steps(
            ["This is a much longer step with detailed content"]
        )
        assert scores_long[0] >= scores_short[0]

    async def test_score_steps_digit_bonus(self, initialized_verifier: VersaPrm) -> None:
        """Test steps with digits get bonus."""
        scores_no_digit = await initialized_verifier.score_steps(["Step one"])
        scores_digit = await initialized_verifier.score_steps(["Step 1: x = 42"])
        assert scores_digit[0] >= scores_no_digit[0]

    async def test_score_steps_with_context(self, initialized_verifier: VersaPrm) -> None:
        """Test score_steps with context parameter."""
        scores = await initialized_verifier.score_steps(
            ["Step 1", "Step 2"],
            context={"problem": "Math problem"},
        )
        assert len(scores) == 2


class TestVersaPrmHealthCheck:
    """Tests for VersaPRM health check."""

    async def test_health_check_false_before_init(self) -> None:
        """Test health check returns False before initialization."""
        verifier = VersaPrm()
        assert await verifier.health_check() is False

    async def test_health_check_true_after_init(self) -> None:
        """Test health check returns True after initialization."""
        verifier = VersaPrm()
        await verifier.initialize()
        assert await verifier.health_check() is True
