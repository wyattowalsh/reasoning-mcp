"""Unit tests for Ensemble Reasoning method.

This module provides comprehensive tests for the EnsembleReasoning method implementation,
covering initialization, execution, configuration, voting strategies, and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.methods.native.ensemble_reasoning import (
    ENSEMBLE_REASONING_METADATA,
    EnsembleReasoning,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    VotingStrategy,
)


@pytest.fixture
def method() -> EnsembleReasoning:
    """Create an EnsembleReasoning method instance for testing.

    Returns:
        A fresh EnsembleReasoning instance
    """
    return EnsembleReasoning()


@pytest.fixture
def custom_config() -> EnsembleConfig:
    """Create a custom ensemble configuration.

    Returns:
        An EnsembleConfig with custom members
    """
    return EnsembleConfig(
        members=[
            EnsembleMember(method_name="chain_of_thought", weight=1.0),
            EnsembleMember(method_name="self_reflection", weight=0.8),
        ],
        strategy=VotingStrategy.WEIGHTED,
        min_agreement=0.6,
        timeout_ms=15000,
    )


@pytest.fixture
def method_with_config(custom_config: EnsembleConfig) -> EnsembleReasoning:
    """Create an EnsembleReasoning method with custom config.

    Returns:
        An EnsembleReasoning instance with custom configuration
    """
    return EnsembleReasoning(config=custom_config)


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
    return "What is the best approach to solve complex multi-step problems?"


@pytest.fixture
def mock_execution_context() -> MagicMock:
    """Create a mock execution context with sampling capability.

    Returns:
        A mock execution context
    """
    ctx = MagicMock()
    ctx.can_sample = True
    ctx.sample = AsyncMock(return_value="Ensemble reasoning result")
    return ctx


@pytest.fixture
def mock_ensemble_result() -> MagicMock:
    """Create a mock ensemble result.

    Returns:
        A mock EnsembleResult object
    """
    result = MagicMock()
    result.final_answer = "The best approach is to decompose and analyze step by step."
    result.confidence = 0.85
    result.agreement_score = 0.8
    result.member_results = [
        MagicMock(method_name="chain_of_thought", answer="Step by step approach"),
        MagicMock(method_name="tree_of_thoughts", answer="Decomposition approach"),
    ]
    result.voting_details = {"strategy": "majority", "votes": {"approach_a": 2}}
    return result


class TestEnsembleReasoningInitialization:
    """Tests for EnsembleReasoning initialization and setup."""

    def test_create_method(self, method: EnsembleReasoning) -> None:
        """Test that EnsembleReasoning can be instantiated."""
        assert method is not None
        assert isinstance(method, EnsembleReasoning)

    def test_initial_state(self, method: EnsembleReasoning) -> None:
        """Test that a new method starts in the correct initial state."""
        assert method._is_initialized is False
        assert method.config is not None

    def test_default_config_applied(self, method: EnsembleReasoning) -> None:
        """Test that default config is applied when none provided."""
        assert len(method.config.members) == 3  # Default has 3 members
        assert method.config.strategy == VotingStrategy.MAJORITY
        assert method.config.min_agreement == 0.5
        assert method.config.timeout_ms == 30000

    def test_custom_config_applied(
        self, method_with_config: EnsembleReasoning, custom_config: EnsembleConfig
    ) -> None:
        """Test that custom config is applied when provided."""
        assert len(method_with_config.config.members) == 2
        assert method_with_config.config.strategy == VotingStrategy.WEIGHTED
        assert method_with_config.config.min_agreement == 0.6

    async def test_initialize(self, method: EnsembleReasoning) -> None:
        """Test that initialize() sets up the method correctly."""
        await method.initialize()
        assert method._is_initialized is True

    async def test_initialize_validates_config(self) -> None:
        """Test that initialize validates configuration has members."""
        empty_config = EnsembleConfig(
            members=[],
            strategy=VotingStrategy.MAJORITY,
            min_agreement=0.5,
            timeout_ms=30000,
        )
        method = EnsembleReasoning(config=empty_config)

        with pytest.raises(ValueError, match="at least one member"):
            await method.initialize()

    async def test_health_check_not_initialized(self, method: EnsembleReasoning) -> None:
        """Test that health_check returns False before initialization."""
        result = await method.health_check()
        assert result is False

    async def test_health_check_initialized(self, method: EnsembleReasoning) -> None:
        """Test that health_check returns True after initialization."""
        await method.initialize()
        result = await method.health_check()
        assert result is True


class TestEnsembleReasoningProperties:
    """Tests for EnsembleReasoning property accessors."""

    def test_identifier_property(self, method: EnsembleReasoning) -> None:
        """Test that identifier returns the correct method identifier."""
        assert method.identifier == str(MethodIdentifier.ENSEMBLE_REASONING)

    def test_name_property(self, method: EnsembleReasoning) -> None:
        """Test that name returns the correct human-readable name."""
        assert method.name == "Ensemble Reasoning"

    def test_description_property(self, method: EnsembleReasoning) -> None:
        """Test that description returns the correct method description."""
        assert "multiple" in method.description.lower()
        assert "voting" in method.description.lower()

    def test_category_property(self, method: EnsembleReasoning) -> None:
        """Test that category returns the correct method category."""
        assert method.category == str(MethodCategory.ADVANCED)


class TestEnsembleReasoningMetadata:
    """Tests for ENSEMBLE_REASONING metadata constant."""

    def test_metadata_identifier(self) -> None:
        """Test that metadata has the correct identifier."""
        assert ENSEMBLE_REASONING_METADATA.identifier == MethodIdentifier.ENSEMBLE_REASONING

    def test_metadata_category(self) -> None:
        """Test that metadata has the correct category."""
        assert ENSEMBLE_REASONING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self) -> None:
        """Test that metadata contains expected tags."""
        expected_tags = {"ensemble", "multi-method", "voting", "parallel"}
        assert expected_tags.issubset(ENSEMBLE_REASONING_METADATA.tags)

    def test_metadata_supports_branching(self) -> None:
        """Test that metadata correctly indicates branching support."""
        assert ENSEMBLE_REASONING_METADATA.supports_branching is False

    def test_metadata_supports_revision(self) -> None:
        """Test that metadata correctly indicates revision support."""
        assert ENSEMBLE_REASONING_METADATA.supports_revision is False

    def test_metadata_complexity(self) -> None:
        """Test that metadata has appropriate complexity rating."""
        assert ENSEMBLE_REASONING_METADATA.complexity == 7


class TestDefaultConfig:
    """Tests for default configuration."""

    def test_get_default_config(self) -> None:
        """Test that get_default_config returns valid configuration."""
        config = EnsembleReasoning.get_default_config()

        assert len(config.members) == 3
        assert config.strategy == VotingStrategy.MAJORITY
        assert config.min_agreement == 0.5
        assert config.timeout_ms == 30000

    def test_default_config_member_names(self) -> None:
        """Test that default config has expected member names."""
        config = EnsembleReasoning.get_default_config()

        member_names = {m.method_name for m in config.members}
        expected_names = {"chain_of_thought", "tree_of_thoughts", "self_reflection"}
        assert member_names == expected_names

    def test_default_config_member_weights(self) -> None:
        """Test that default config members have equal weights."""
        config = EnsembleReasoning.get_default_config()

        for member in config.members:
            assert member.weight == 1.0


class TestEnsembleReasoningExecution:
    """Tests for EnsembleReasoning execute() method."""

    async def test_execute_basic(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test basic execution creates a thought."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert thought is not None
            assert isinstance(thought, ThoughtNode)
            assert thought.content != ""
            assert thought.method_id == MethodIdentifier.ENSEMBLE_REASONING

    async def test_execute_auto_initializes(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that execute auto-initializes if not initialized."""
        assert method._is_initialized is False

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert method._is_initialized is True
            assert thought is not None

    async def test_execute_creates_initial_thought(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that execute creates an INITIAL thought type for empty session."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert thought.type == ThoughtType.INITIAL
            assert thought.parent_id is None
            assert thought.depth == 0

    async def test_execute_includes_metadata(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that execute includes proper metadata."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert "agreement_score" in thought.metadata
            assert "member_count" in thought.metadata
            assert "strategy" in thought.metadata
            assert "voting_details" in thought.metadata
            assert thought.metadata["method"] == "ensemble_reasoning"

    async def test_execute_uses_orchestrator(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that execute uses EnsembleOrchestrator."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            await method.execute(session, sample_problem)

            mock_orchestrator.assert_called_once()
            mock_orchestrator.return_value.execute.assert_called_once_with(sample_problem)


class TestEnsembleReasoningContinuation:
    """Tests for continue_reasoning() method."""

    async def test_continue_reasoning(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that continue_reasoning executes ensemble again."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            initial = await method.execute(session, sample_problem)
            continuation = await method.continue_reasoning(
                session, initial, guidance="Consider alternative approaches"
            )

            assert continuation is not None
            assert continuation.parent_id == initial.id

    async def test_continue_reasoning_with_guidance(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that continue_reasoning incorporates guidance."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            initial = await method.execute(session, sample_problem)

            guidance = "Focus on edge cases"
            await method.continue_reasoning(session, initial, guidance=guidance)

            # Verify orchestrator was called with continuation input
            call_args = mock_orchestrator.return_value.execute.call_args
            input_text = call_args[0][0]
            assert guidance in input_text


class TestVotingStrategies:
    """Tests for different voting strategies."""

    async def test_majority_voting_strategy(
        self,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test majority voting strategy configuration."""
        config = EnsembleConfig(
            members=[EnsembleMember(method_name="chain_of_thought", weight=1.0)],
            strategy=VotingStrategy.MAJORITY,
            min_agreement=0.5,
            timeout_ms=30000,
        )
        method = EnsembleReasoning(config=config)
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert thought.metadata["strategy"] == "majority"

    async def test_weighted_voting_strategy(
        self,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test weighted voting strategy configuration."""
        mock_ensemble_result.voting_details = {"strategy": "weighted"}

        config = EnsembleConfig(
            members=[
                EnsembleMember(method_name="chain_of_thought", weight=2.0),
                EnsembleMember(method_name="self_reflection", weight=1.0),
            ],
            strategy=VotingStrategy.WEIGHTED,
            min_agreement=0.5,
            timeout_ms=30000,
        )
        method = EnsembleReasoning(config=config)
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, sample_problem)

            assert thought.metadata["strategy"] == "weighted"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        method: EnsembleReasoning,
        session: Session,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test execution with empty problem string."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, "")

            assert thought is not None

    async def test_very_long_problem(
        self,
        method: EnsembleReasoning,
        session: Session,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test execution with very long problem."""
        await method.initialize()
        long_problem = "Analyze the following complex scenario: " + "context " * 500

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            thought = await method.execute(session, long_problem)

            assert thought is not None

    async def test_context_override_config(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
        custom_config: EnsembleConfig,
    ) -> None:
        """Test that context can override ensemble config."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            await method.execute(
                session, sample_problem, context={"ensemble_config": custom_config}
            )

            # Verify orchestrator was created with the context config
            call_kwargs = mock_orchestrator.call_args[1]
            assert call_kwargs["config"] == custom_config


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_thought_added_to_session(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that thought is added to session."""
        await method.initialize()
        initial_count = session.thought_count

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            await method.execute(session, sample_problem)

            assert session.thought_count == initial_count + 1

    async def test_continuation_sets_parent(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
        mock_ensemble_result: MagicMock,
    ) -> None:
        """Test that continuation thought sets parent correctly."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(return_value=mock_ensemble_result)

            initial = await method.execute(session, sample_problem)
            continuation = await method.continue_reasoning(session, initial)

            assert continuation.parent_id == initial.id
            assert continuation.depth == initial.depth + 1


class TestExceptionHandling:
    """Tests for exception handling in ensemble reasoning."""

    async def test_timeout_error_propagates(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that TimeoutError is properly re-raised."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(
                side_effect=TimeoutError("Ensemble execution timed out")
            )

            with pytest.raises(TimeoutError):
                await method.execute(session, sample_problem)

    async def test_connection_error_raises_runtime_error(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that ConnectionError is wrapped in RuntimeError."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(
                side_effect=ConnectionError("Network error")
            )

            with pytest.raises(RuntimeError, match="connection error"):
                await method.execute(session, sample_problem)

    async def test_os_error_raises_runtime_error(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that OSError is wrapped in RuntimeError."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(
                side_effect=OSError("I/O error")
            )

            with pytest.raises(RuntimeError, match="connection error"):
                await method.execute(session, sample_problem)

    async def test_value_error_propagates(
        self,
        method: EnsembleReasoning,
        session: Session,
        sample_problem: str,
    ) -> None:
        """Test that ValueError (no results) is properly re-raised."""
        await method.initialize()

        with patch("reasoning_mcp.ensemble.orchestrator.EnsembleOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.execute = AsyncMock(
                side_effect=ValueError("No ensemble members completed")
            )

            with pytest.raises(ValueError, match="No ensemble members"):
                await method.execute(session, sample_problem)
