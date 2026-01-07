"""Unit tests for Abductive reasoning method.

This test file provides comprehensive test coverage for the AbductiveMethod class,
testing initialization, execution, hypothesis generation, evidence evaluation,
and all edge cases for abductive reasoning.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.abductive import ABDUCTIVE_METADATA, Abductive
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def abductive_method():
    """Create a fresh Abductive method instance for each test."""
    return Abductive()


@pytest.fixture
async def initialized_method():
    """Create an initialized Abductive method instance."""
    method = Abductive()
    await method.initialize()
    return method


@pytest.fixture
def session():
    """Create a started session for testing."""
    return Session().start()


@pytest.fixture
async def session_with_observation(session, initialized_method):
    """Create a session with an initial observation thought."""
    thought = await initialized_method.execute(
        session=session,
        input_text="Patient has fever, headache, and stiff neck",
    )
    return session, thought


class TestAbductiveMetadata:
    """Tests for ABDUCTIVE_METADATA configuration."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert ABDUCTIVE_METADATA.identifier == MethodIdentifier.ABDUCTIVE

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert ABDUCTIVE_METADATA.name == "Abductive Reasoning"

    def test_metadata_description(self):
        """Test that metadata has a description."""
        assert "best explanation" in ABDUCTIVE_METADATA.description.lower()
        assert "observations" in ABDUCTIVE_METADATA.description.lower()

    def test_metadata_category(self):
        """Test that metadata is in SPECIALIZED category."""
        assert ABDUCTIVE_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has reasonable complexity."""
        assert ABDUCTIVE_METADATA.complexity == 6
        assert 1 <= ABDUCTIVE_METADATA.complexity <= 10

    def test_metadata_supports_branching(self):
        """Test that abductive reasoning supports branching for multiple hypotheses."""
        assert ABDUCTIVE_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that abductive reasoning supports revision."""
        assert ABDUCTIVE_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test that abductive reasoning doesn't require context."""
        assert ABDUCTIVE_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test that metadata specifies minimum thoughts."""
        assert ABDUCTIVE_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata has no max thought limit."""
        assert ABDUCTIVE_METADATA.max_thoughts == 0  # unlimited

    def test_metadata_tags(self):
        """Test that metadata has appropriate tags."""
        assert "abductive" in ABDUCTIVE_METADATA.tags
        assert "hypothesis" in ABDUCTIVE_METADATA.tags
        assert "inference" in ABDUCTIVE_METADATA.tags
        assert "diagnosis" in ABDUCTIVE_METADATA.tags
        assert "investigation" in ABDUCTIVE_METADATA.tags

    def test_metadata_best_for(self):
        """Test that metadata specifies appropriate use cases."""
        best_for_text = " ".join(ABDUCTIVE_METADATA.best_for).lower()
        assert "diagnostic" in best_for_text or "diagnosis" in best_for_text
        assert "investigation" in best_for_text or "mystery" in best_for_text

    def test_metadata_not_recommended_for(self):
        """Test that metadata specifies inappropriate use cases."""
        not_recommended_text = " ".join(ABDUCTIVE_METADATA.not_recommended_for).lower()
        assert "deductive" in not_recommended_text or "mathematical" in not_recommended_text


class TestAbductiveInitialization:
    """Tests for Abductive method initialization."""

    def test_create_instance(self, abductive_method):
        """Test that we can create an Abductive instance."""
        assert isinstance(abductive_method, Abductive)

    def test_initial_state(self, abductive_method):
        """Test that initial state is correct before initialization."""
        assert abductive_method._initialized is False
        assert abductive_method._step_counter == 0
        assert abductive_method._stage == "observations"
        assert abductive_method._hypotheses == []
        assert abductive_method._observations == []

    @pytest.mark.asyncio
    async def test_initialize(self, abductive_method):
        """Test that initialize sets up the method correctly."""
        await abductive_method.initialize()
        assert abductive_method._initialized is True
        assert abductive_method._step_counter == 0
        assert abductive_method._stage == "observations"
        assert abductive_method._hypotheses == []
        assert abductive_method._observations == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize resets state from previous executions."""
        method = Abductive()
        await method.initialize()

        # Modify state
        method._step_counter = 5
        method._stage = "evaluation"
        method._hypotheses = [{"id": 1, "test": "data"}]
        method._observations = ["obs1", "obs2"]

        # Reinitialize
        await method.initialize()

        # Should be reset
        assert method._step_counter == 0
        assert method._stage == "observations"
        assert method._hypotheses == []
        assert method._observations == []

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, abductive_method):
        """Test health_check returns False before initialization."""
        health = await abductive_method.health_check()
        assert health is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, initialized_method):
        """Test health_check returns True after initialization."""
        health = await initialized_method.health_check()
        assert health is True


class TestAbductiveProperties:
    """Tests for Abductive method properties."""

    def test_identifier_property(self, abductive_method):
        """Test that identifier property returns correct value."""
        assert abductive_method.identifier == MethodIdentifier.ABDUCTIVE

    def test_name_property(self, abductive_method):
        """Test that name property returns correct value."""
        assert abductive_method.name == "Abductive Reasoning"

    def test_description_property(self, abductive_method):
        """Test that description property returns correct value."""
        assert abductive_method.description == ABDUCTIVE_METADATA.description

    def test_category_property(self, abductive_method):
        """Test that category property returns correct value."""
        assert abductive_method.category == MethodCategory.SPECIALIZED


class TestAbductiveExecution:
    """Tests for basic execution of Abductive reasoning."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_fails(self, abductive_method, session):
        """Test that execute fails if method not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized"):
            await abductive_method.execute(
                session=session,
                input_text="Test observation",
            )

    @pytest.mark.asyncio
    async def test_execute_basic(self, initialized_method, session):
        """Test basic execution creates observation thought."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Server crashes at midnight",
        )

        assert isinstance(thought, ThoughtNode)
        assert thought.type == ThoughtType.OBSERVATION
        assert thought.method_id == MethodIdentifier.ABDUCTIVE
        assert thought.step_number == 1
        assert thought.depth == 0
        assert "Server crashes at midnight" in initialized_method._observations

    @pytest.mark.asyncio
    async def test_execute_sets_stage(self, initialized_method, session):
        """Test that execute sets stage to observations."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
        )

        assert initialized_method._stage == "observations"
        assert thought.metadata["stage"] == "observations"

    @pytest.mark.asyncio
    async def test_execute_parses_observations(self, initialized_method, session):
        """Test that execute parses observations from input."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Main observation",
        )

        assert "Main observation" in initialized_method._observations
        assert len(initialized_method._observations) >= 1

    @pytest.mark.asyncio
    async def test_execute_with_context_observations(self, initialized_method, session):
        """Test that execute incorporates context observations."""
        context = {
            "observations": ["Additional obs 1", "Additional obs 2"],
            "evidence": "Some evidence",
        }

        thought = await initialized_method.execute(
            session=session,
            input_text="Main observation",
            context=context,
        )

        assert "Main observation" in initialized_method._observations
        assert "Additional obs 1" in initialized_method._observations
        assert "Additional obs 2" in initialized_method._observations
        assert "Some evidence" in initialized_method._observations

    @pytest.mark.asyncio
    async def test_execute_with_list_evidence(self, initialized_method, session):
        """Test that execute handles list evidence in context."""
        context = {
            "evidence": ["Evidence 1", "Evidence 2"],
        }

        thought = await initialized_method.execute(
            session=session,
            input_text="Main observation",
            context=context,
        )

        assert "Evidence 1" in initialized_method._observations
        assert "Evidence 2" in initialized_method._observations

    @pytest.mark.asyncio
    async def test_execute_adds_to_session(self, initialized_method, session):
        """Test that execute adds thought to session."""
        initial_count = session.thought_count

        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
        )

        assert session.thought_count == initial_count + 1
        assert session.current_method == MethodIdentifier.ABDUCTIVE

    @pytest.mark.asyncio
    async def test_execute_content_format(self, initialized_method, session):
        """Test that execute generates properly formatted content."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
        )

        assert "Step 1" in thought.content
        assert "Observation Collection" in thought.content
        assert "best explanation" in thought.content

    @pytest.mark.asyncio
    async def test_execute_confidence_level(self, initialized_method, session):
        """Test that execute sets appropriate confidence for observations."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
        )

        assert thought.confidence == 0.8  # High confidence in observations

    @pytest.mark.asyncio
    async def test_execute_metadata(self, initialized_method, session):
        """Test that execute sets appropriate metadata."""
        input_text = "Test observation"
        context = {"key": "value"}

        thought = await initialized_method.execute(
            session=session,
            input_text=input_text,
            context=context,
        )

        assert thought.metadata["input"] == input_text
        assert thought.metadata["context"] == context
        assert thought.metadata["stage"] == "observations"
        assert thought.metadata["reasoning_type"] == "abductive"
        assert "observations" in thought.metadata


class TestHypothesisGeneration:
    """Tests for hypothesis generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_hypotheses(self, session_with_observation):
        """Test that continue_reasoning generates hypotheses."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()

        # Need to restore state
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Generate hypotheses",
        )

        assert hyp_thought.type == ThoughtType.HYPOTHESIS
        assert method._stage == "hypothesis_generation"
        assert len(method._hypotheses) > 0

    @pytest.mark.asyncio
    async def test_multiple_hypotheses_generated(self, session_with_observation):
        """Test that multiple candidate hypotheses are generated."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Generate hypotheses",
        )

        # Should generate at least 2-3 hypotheses
        assert len(method._hypotheses) >= 2
        assert len(method._hypotheses) <= 5

    @pytest.mark.asyncio
    async def test_hypothesis_structure(self, session_with_observation):
        """Test that hypotheses have proper structure."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Generate hypotheses",
        )

        for hypothesis in method._hypotheses:
            assert "id" in hypothesis
            assert "explanation" in hypothesis
            assert "supports" in hypothesis
            assert "likelihood" in hypothesis

    @pytest.mark.asyncio
    async def test_hypothesis_content_includes_all(self, session_with_observation):
        """Test that hypothesis content includes all generated hypotheses."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Generate hypotheses",
        )

        # Content should reference hypothesis generation
        assert "Hypothesis Generation" in hyp_thought.content
        assert "candidate explanations" in hyp_thought.content.lower()

    @pytest.mark.asyncio
    async def test_hypothesis_metadata(self, session_with_observation):
        """Test that hypothesis thought has proper metadata."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Generate hypotheses",
        )

        assert hyp_thought.metadata["stage"] == "hypothesis_generation"
        assert hyp_thought.metadata["previous_stage"] == "observations"
        assert hyp_thought.metadata["hypotheses_count"] > 0


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_without_initialization_fails(self, abductive_method, session):
        """Test that continue_reasoning fails if not initialized."""
        thought = ThoughtNode(
            type=ThoughtType.OBSERVATION,
            method_id=MethodIdentifier.ABDUCTIVE,
            content="Test",
            confidence=0.8,
        )

        with pytest.raises(RuntimeError, match="must be initialized"):
            await abductive_method.continue_reasoning(
                session=session,
                previous_thought=thought,
            )

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(self, session_with_observation):
        """Test that continue_reasoning increments step counter."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        assert hyp_thought.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_sets_parent_id(self, session_with_observation):
        """Test that continue_reasoning sets parent_id correctly."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        assert hyp_thought.parent_id == obs_thought.id

    @pytest.mark.asyncio
    async def test_continue_increases_depth(self, session_with_observation):
        """Test that continue_reasoning increases depth."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        assert hyp_thought.depth == obs_thought.depth + 1

    @pytest.mark.asyncio
    async def test_continue_stage_progression_hypothesis(self, session_with_observation):
        """Test stage progression from observations to hypothesis generation."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        assert method._stage == "hypothesis_generation"
        assert hyp_thought.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_continue_stage_progression_evaluation(self, session_with_observation):
        """Test stage progression from hypothesis to evaluation."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # First generate hypotheses
        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Then evaluate
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )

        assert method._stage == "evaluation"
        assert eval_thought.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_continue_stage_progression_conclusion(self, session_with_observation):
        """Test stage progression to conclusion."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Generate hypotheses
        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Evaluate
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )

        # Conclude
        conclusion_thought = await method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )

        assert method._stage == "conclusion"
        assert conclusion_thought.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_continue_with_guidance_override(self, session_with_observation):
        """Test that guidance can override default stage progression."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Force evaluation stage with guidance
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="Evaluate the evidence",
        )

        assert method._stage == "evaluation"
        assert eval_thought.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_continue_adds_to_session(self, session_with_observation):
        """Test that continue_reasoning adds thought to session."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        initial_count = session.thought_count

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        assert session.thought_count == initial_count + 1


class TestEvidenceEvaluation:
    """Tests for evidence evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluation_content(self, session_with_observation):
        """Test that evaluation generates appropriate content."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Generate hypotheses first
        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Then evaluate
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )

        assert "Hypothesis Evaluation" in eval_thought.content
        assert "Evaluation Criteria" in eval_thought.content

    @pytest.mark.asyncio
    async def test_evaluation_criteria_present(self, session_with_observation):
        """Test that evaluation mentions key criteria."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )

        content_lower = eval_thought.content.lower()
        assert "explanatory power" in content_lower
        assert "simplicity" in content_lower or "occam" in content_lower
        assert "consistency" in content_lower

    @pytest.mark.asyncio
    async def test_evaluation_references_hypothesis_count(self, session_with_observation):
        """Test that evaluation references the number of hypotheses."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )

        hypothesis_count = len(method._hypotheses)
        assert str(hypothesis_count) in eval_thought.content


class TestBestExplanationSelection:
    """Tests for selecting the best explanation."""

    @pytest.mark.asyncio
    async def test_conclusion_content(self, session_with_observation):
        """Test that conclusion generates appropriate content."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Progress through stages
        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )
        conclusion_thought = await method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )

        assert "Best Explanation" in conclusion_thought.content
        assert "most plausible" in conclusion_thought.content.lower()

    @pytest.mark.asyncio
    async def test_conclusion_provides_reasoning(self, session_with_observation):
        """Test that conclusion provides reasoning for selection."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )
        conclusion_thought = await method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )

        content_lower = conclusion_thought.content.lower()
        assert "reasoning" in content_lower or "accounts for" in content_lower
        assert "confidence" in content_lower

    @pytest.mark.asyncio
    async def test_conclusion_acknowledges_uncertainty(self, session_with_observation):
        """Test that conclusion acknowledges abductive reasoning uncertainty."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )
        conclusion_thought = await method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )

        content_lower = conclusion_thought.content.lower()
        # Should acknowledge that abductive reasoning isn't guaranteed truth
        assert "not guaranteed" in content_lower or "best explanation" in content_lower


class TestHypothesisRanking:
    """Tests for hypothesis ranking by explanatory power."""

    @pytest.mark.asyncio
    async def test_hypotheses_have_likelihood(self, session_with_observation):
        """Test that generated hypotheses include likelihood rankings."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Check hypotheses have likelihood
        for hypothesis in method._hypotheses:
            assert "likelihood" in hypothesis
            assert hypothesis["likelihood"] in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_hypotheses_ranked_by_likelihood(self, session_with_observation):
        """Test that hypotheses include different likelihood levels."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Should have varied likelihoods
        likelihoods = [h["likelihood"] for h in method._hypotheses]
        assert len(set(likelihoods)) > 1  # At least some variation


class TestConfigurationOptions:
    """Tests for configuration options like hypothesis_count and evidence_threshold."""

    @pytest.mark.asyncio
    async def test_context_modifies_behavior(self, initialized_method, session):
        """Test that context can influence hypothesis generation."""
        # This test demonstrates that context is passed through
        context = {"hypothesis_count": 5, "evidence_threshold": 0.7}

        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
            context=context,
        )

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_additional_observations_stage(self, session_with_observation):
        """Test that new evidence can be added via additional observations."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "conclusion"  # Set to later stage

        new_obs_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
            guidance="New evidence has emerged",
            context={"new_evidence": ["Evidence A", "Evidence B"]},
        )

        assert new_obs_thought.type == ThoughtType.OBSERVATION
        assert method._stage == "observations"
        assert "Additional Observations" in new_obs_thought.content


class TestEdgeCases:
    """Tests for edge cases in abductive reasoning."""

    @pytest.mark.asyncio
    async def test_single_observation(self, initialized_method, session):
        """Test handling of single obvious observation."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Single clear observation",
        )

        assert len(initialized_method._observations) == 1
        assert thought.type == ThoughtType.OBSERVATION

    @pytest.mark.asyncio
    async def test_competing_explanations(self, session_with_observation):
        """Test handling of competing explanations with similar likelihood."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Should generate multiple competing hypotheses
        assert len(method._hypotheses) >= 2

    @pytest.mark.asyncio
    async def test_empty_context(self, initialized_method, session):
        """Test execution with empty context."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
            context={},
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_none_context(self, initialized_method, session):
        """Test execution with None context."""
        thought = await initialized_method.execute(
            session=session,
            input_text="Test observation",
            context=None,
        )

        assert thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_refinement_stage(self, session_with_observation):
        """Test that refinement stage can be reached."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Progress through all stages
        hyp_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )
        eval_thought = await method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )
        conclusion_thought = await method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )
        refinement_thought = await method.continue_reasoning(
            session=session,
            previous_thought=conclusion_thought,
        )

        assert method._stage == "refinement"
        assert refinement_thought.type == ThoughtType.REVISION

    @pytest.mark.asyncio
    async def test_continuation_stage(self, session_with_observation):
        """Test generic continuation for unknown stages."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "unknown_stage"
        method._observations = obs_thought.metadata["observations"]

        cont_thought = await method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Unknown stage falls through to default hypothesis generation
        assert cont_thought is not None
        assert cont_thought.type == ThoughtType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_guidance_with_various_keywords(self, session_with_observation):
        """Test that guidance keywords properly trigger stage changes."""
        session, obs_thought = session_with_observation
        method = Abductive()
        await method.initialize()
        method._step_counter = 1
        method._stage = "observations"
        method._observations = obs_thought.metadata["observations"]

        # Test different guidance keywords
        test_cases = [
            ("Generate new hypotheses", ThoughtType.HYPOTHESIS),
            ("Assess the evidence", ThoughtType.VERIFICATION),
            ("Select the best explanation", ThoughtType.CONCLUSION),
            ("Observe new evidence", ThoughtType.OBSERVATION),
        ]

        for guidance, expected_type in test_cases:
            method._step_counter = 1
            thought = await method.continue_reasoning(
                session=session,
                previous_thought=obs_thought,
                guidance=guidance,
            )
            assert thought.type == expected_type

    @pytest.mark.asyncio
    async def test_multiple_execution_cycles(self, initialized_method, session):
        """Test that method can handle multiple execution cycles."""
        # First execution
        thought1 = await initialized_method.execute(
            session=session,
            input_text="First problem",
        )
        assert thought1.step_number == 1

        # Second execution (should reset)
        thought2 = await initialized_method.execute(
            session=session,
            input_text="Second problem",
        )
        assert thought2.step_number == 1
        assert initialized_method._hypotheses == []


class TestIntegration:
    """Integration tests for complete abductive reasoning flow."""

    @pytest.mark.asyncio
    async def test_complete_reasoning_flow(self, initialized_method, session):
        """Test a complete abductive reasoning flow from start to finish."""
        # 1. Initial observation
        obs_thought = await initialized_method.execute(
            session=session,
            input_text="Patient has high fever and severe headache",
            context={"evidence": ["Stiff neck", "Light sensitivity"]},
        )
        assert obs_thought.type == ThoughtType.OBSERVATION
        assert len(initialized_method._observations) >= 3

        # 2. Generate hypotheses
        hyp_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )
        assert hyp_thought.type == ThoughtType.HYPOTHESIS
        assert len(initialized_method._hypotheses) > 0

        # 3. Evaluate hypotheses
        eval_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
        )
        assert eval_thought.type == ThoughtType.VERIFICATION

        # 4. Select best explanation
        conclusion_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=eval_thought,
        )
        assert conclusion_thought.type == ThoughtType.CONCLUSION

        # Verify session state
        assert session.thought_count == 4
        assert session.current_method == MethodIdentifier.ABDUCTIVE
        assert session.metrics.total_thoughts == 4

    @pytest.mark.asyncio
    async def test_reasoning_with_new_evidence(self, initialized_method, session):
        """Test abductive reasoning with new evidence emerging mid-process."""
        # Initial observation and hypothesis
        obs_thought = await initialized_method.execute(
            session=session,
            input_text="Car won't start",
        )

        hyp_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # New evidence emerges
        new_evidence_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=hyp_thought,
            guidance="New observations",
            context={"new_evidence": ["Battery terminals are corroded"]},
        )

        assert new_evidence_thought.type == ThoughtType.OBSERVATION
        assert "Additional Observations" in new_evidence_thought.content

    @pytest.mark.asyncio
    async def test_diagnostic_scenario(self, initialized_method, session):
        """Test a diagnostic scenario - the primary use case for abductive reasoning."""
        # Medical diagnosis scenario
        obs_thought = await initialized_method.execute(
            session=session,
            input_text="Server response time degraded significantly",
            context={
                "observations": [
                    "CPU usage normal",
                    "Memory usage normal",
                    "Disk I/O very high",
                ],
            },
        )

        assert len(initialized_method._observations) >= 4
        assert obs_thought.type == ThoughtType.OBSERVATION

        # Generate diagnostic hypotheses
        hyp_thought = await initialized_method.continue_reasoning(
            session=session,
            previous_thought=obs_thought,
        )

        # Should have multiple diagnostic hypotheses
        assert len(initialized_method._hypotheses) >= 2
        assert hyp_thought.type == ThoughtType.HYPOTHESIS
