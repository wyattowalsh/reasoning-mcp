"""Unit tests for CausalReasoning method."""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.causal import (
    CAUSAL_REASONING_METADATA,
    CausalReasoning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def causal_method():
    """Create a CausalReasoning method instance."""
    return CausalReasoning()


@pytest.fixture
def session():
    """Create a fresh session for testing."""
    return Session().start()


class TestCausalReasoningMetadata:
    """Tests for CAUSAL_REASONING_METADATA."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert CAUSAL_REASONING_METADATA.identifier == MethodIdentifier.CAUSAL_REASONING

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert CAUSAL_REASONING_METADATA.name == "Causal Reasoning"

    def test_metadata_category(self):
        """Test metadata is in ADVANCED category."""
        assert CAUSAL_REASONING_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {
            "causal",
            "cause-effect",
            "root-cause",
            "diagnosis",
            "debugging",
            "prediction",
            "systems-thinking",
        }
        assert CAUSAL_REASONING_METADATA.tags == frozenset(expected_tags)

    def test_metadata_complexity(self):
        """Test metadata complexity level."""
        assert CAUSAL_REASONING_METADATA.complexity == 6

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert CAUSAL_REASONING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert CAUSAL_REASONING_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context is not required."""
        assert CAUSAL_REASONING_METADATA.requires_context is False

    def test_metadata_thought_bounds(self):
        """Test metadata thought count bounds."""
        assert CAUSAL_REASONING_METADATA.min_thoughts == 3
        assert CAUSAL_REASONING_METADATA.max_thoughts == 0  # unlimited

    def test_metadata_avg_tokens(self):
        """Test metadata average tokens per thought."""
        assert CAUSAL_REASONING_METADATA.avg_tokens_per_thought == 400

    def test_metadata_best_for(self):
        """Test metadata best use cases."""
        assert len(CAUSAL_REASONING_METADATA.best_for) > 0
        assert "debugging and troubleshooting" in CAUSAL_REASONING_METADATA.best_for
        assert "root cause analysis" in CAUSAL_REASONING_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata not recommended use cases."""
        assert len(CAUSAL_REASONING_METADATA.not_recommended_for) > 0
        assert "purely creative tasks" in CAUSAL_REASONING_METADATA.not_recommended_for


class TestCausalReasoningInitialization:
    """Tests for initialization and health check."""

    def test_initial_state(self, causal_method):
        """Test method is not initialized on creation."""
        assert causal_method._initialized is False
        assert causal_method._step_counter == 0
        assert causal_method._causal_chain == []
        assert causal_method._root_causes == []
        assert causal_method._effects == []

    def test_identifier_property(self, causal_method):
        """Test identifier property returns correct value."""
        assert causal_method.identifier == MethodIdentifier.CAUSAL_REASONING

    def test_name_property(self, causal_method):
        """Test name property returns correct value."""
        assert causal_method.name == "Causal Reasoning"

    def test_description_property(self, causal_method):
        """Test description property returns correct value."""
        assert "cause-effect relationships" in causal_method.description

    def test_category_property(self, causal_method):
        """Test category property returns correct value."""
        assert causal_method.category == MethodCategory.ADVANCED

    @pytest.mark.asyncio
    async def test_initialize(self, causal_method):
        """Test initialize method sets initialized flag and resets state."""
        await causal_method.initialize()
        assert causal_method._initialized is True
        assert causal_method._step_counter == 0
        assert causal_method._causal_chain == []
        assert causal_method._root_causes == []
        assert causal_method._effects == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, causal_method):
        """Test initialize resets state even if previously used."""
        # Simulate prior use
        causal_method._step_counter = 5
        causal_method._causal_chain = [{"test": "data"}]
        causal_method._root_causes = ["cause1"]
        causal_method._effects = ["effect1"]

        await causal_method.initialize()

        assert causal_method._step_counter == 0
        assert causal_method._causal_chain == []
        assert causal_method._root_causes == []
        assert causal_method._effects == []

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, causal_method):
        """Test health check returns False before initialization."""
        healthy = await causal_method.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, causal_method):
        """Test health check returns True after initialization."""
        await causal_method.initialize()
        healthy = await causal_method.health_check()
        assert healthy is True


class TestCausalReasoningExecution:
    """Tests for basic execution flow."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises(self, causal_method, session):
        """Test execute raises RuntimeError if not initialized."""
        assert causal_method._initialized is False
        with pytest.raises(RuntimeError, match="must be initialized"):
            await causal_method.execute(session, "Why did the server crash?")

    @pytest.mark.asyncio
    async def test_execute_returns_thought_node(self, causal_method, session):
        """Test execute returns a ThoughtNode."""
        await causal_method.initialize()
        effect = "The website is loading slowly"
        result = await causal_method.execute(session, effect)

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_execute_returns_initial_thought_type(self, causal_method, session):
        """Test execute returns ThoughtNode with INITIAL type."""
        await causal_method.initialize()
        effect = "Database queries timing out"
        result = await causal_method.execute(session, effect)

        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_execute_sets_correct_method_id(self, causal_method, session):
        """Test execute sets correct method identifier."""
        await causal_method.initialize()
        effect = "Application crashes on startup"
        result = await causal_method.execute(session, effect)

        assert result.method_id == MethodIdentifier.CAUSAL_REASONING

    @pytest.mark.asyncio
    async def test_execute_sets_step_number_to_one(self, causal_method, session):
        """Test execute creates thought with step_number 1."""
        await causal_method.initialize()
        effect = "Memory leak detected"
        result = await causal_method.execute(session, effect)

        assert result.step_number == 1

    @pytest.mark.asyncio
    async def test_execute_sets_depth_to_zero(self, causal_method, session):
        """Test execute creates thought with depth 0."""
        await causal_method.initialize()
        effect = "Network connection dropped"
        result = await causal_method.execute(session, effect)

        assert result.depth == 0

    @pytest.mark.asyncio
    async def test_execute_with_context(self, causal_method, session):
        """Test execute accepts and processes context parameter."""
        await causal_method.initialize()
        effect = "CPU usage spiked to 100%"
        context: dict[str, Any] = {
            "timestamp": "3am",
            "system_load": "high",
            "recent_changes": "deployed new code"
        }
        result = await causal_method.execute(session, effect, context=context)

        assert result is not None
        assert result.metadata.get("context") == context

    @pytest.mark.asyncio
    async def test_execute_without_context(self, causal_method, session):
        """Test execute works without context parameter."""
        await causal_method.initialize()
        effect = "User login failures increasing"
        result = await causal_method.execute(session, effect, context=None)

        assert result is not None
        assert result.metadata.get("context") == {}

    @pytest.mark.asyncio
    async def test_execute_content_includes_effect(self, causal_method, session):
        """Test execute content includes the effect being analyzed."""
        await causal_method.initialize()
        effect = "Sales dropped 30% last quarter"
        result = await causal_method.execute(session, effect)

        assert effect in result.content

    @pytest.mark.asyncio
    async def test_execute_content_describes_causal_analysis(self, causal_method, session):
        """Test execute content describes causal analysis framework."""
        await causal_method.initialize()
        effect = "Customer complaints increased"
        result = await causal_method.execute(session, effect)

        assert "Causal Analysis" in result.content or "causal" in result.content.lower()
        assert "OBSERVATION" in result.content

    @pytest.mark.asyncio
    async def test_execute_sets_observation_stage(self, causal_method, session):
        """Test execute sets stage to observation_and_hypothesis."""
        await causal_method.initialize()
        effect = "Production errors spiked"
        result = await causal_method.execute(session, effect)

        assert result.metadata.get("stage") == "observation_and_hypothesis"

    @pytest.mark.asyncio
    async def test_execute_sets_reasoning_type(self, causal_method, session):
        """Test execute sets reasoning_type metadata."""
        await causal_method.initialize()
        effect = "Test failures on CI/CD"
        result = await causal_method.execute(session, effect)

        assert result.metadata.get("reasoning_type") == "causal"

    @pytest.mark.asyncio
    async def test_execute_initializes_causal_factors(self, causal_method, session):
        """Test execute initializes empty causal_factors list."""
        await causal_method.initialize()
        effect = "Deployment rollback triggered"
        result = await causal_method.execute(session, effect)

        assert result.metadata.get("causal_factors") == []

    @pytest.mark.asyncio
    async def test_execute_sets_moderate_confidence(self, causal_method, session):
        """Test execute sets moderate initial confidence."""
        await causal_method.initialize()
        effect = "API latency increased"
        result = await causal_method.execute(session, effect)

        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self, causal_method, session):
        """Test execute adds thought to session."""
        await causal_method.initialize()
        initial_count = session.thought_count
        effect = "Cache miss rate high"
        await causal_method.execute(session, effect)

        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_sets_current_method(self, causal_method, session):
        """Test execute sets session's current_method."""
        await causal_method.initialize()
        effect = "Queue backlog growing"
        await causal_method.execute(session, effect)

        assert session.current_method == MethodIdentifier.CAUSAL_REASONING

    @pytest.mark.asyncio
    async def test_execute_resets_state_on_new_execution(self, causal_method, session):
        """Test execute resets internal state for new execution."""
        await causal_method.initialize()

        # Simulate previous execution state
        causal_method._step_counter = 5
        causal_method._causal_chain = [{"test": "old"}]

        effect = "New problem to analyze"
        result = await causal_method.execute(session, effect)

        # State should be reset
        assert result.step_number == 1
        assert causal_method._step_counter == 1


class TestCauseIdentification:
    """Tests for cause-effect relationship identification."""

    @pytest.mark.asyncio
    async def test_initial_analysis_mentions_causes(self, causal_method, session):
        """Test initial analysis discusses potential causes."""
        await causal_method.initialize()
        effect = "Server response time degraded"
        result = await causal_method.execute(session, effect)

        content_lower = result.content.lower()
        assert "cause" in content_lower or "factor" in content_lower

    @pytest.mark.asyncio
    async def test_initial_analysis_mentions_causal_hypotheses(self, causal_method, session):
        """Test initial analysis generates causal hypotheses."""
        await causal_method.initialize()
        effect = "User session timeouts"
        result = await causal_method.execute(session, effect)

        assert "HYPOTHESES" in result.content or "hypothesis" in result.content.lower()

    @pytest.mark.asyncio
    async def test_initial_analysis_discusses_necessary_conditions(self, causal_method, session):
        """Test initial analysis mentions necessary vs sufficient conditions."""
        await causal_method.initialize()
        effect = "Payment processing failures"
        result = await causal_method.execute(session, effect)

        assert "necessary" in result.content.lower() or "sufficient" in result.content.lower()

    @pytest.mark.asyncio
    async def test_initial_analysis_identifies_direct_causes(self, causal_method, session):
        """Test initial analysis identifies direct causes."""
        await causal_method.initialize()
        effect = "Data corruption detected"
        result = await causal_method.execute(session, effect)

        assert "direct" in result.content.lower() or "Direct" in result.content


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_initialization_raises(self, causal_method, session):
        """Test continue_reasoning raises RuntimeError if not initialized."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Problem")

        # Reset initialization
        causal_method._initialized = False

        with pytest.raises(RuntimeError, match="must be initialized"):
            await causal_method.continue_reasoning(session, first)

    @pytest.mark.asyncio
    async def test_continue_reasoning_returns_thought_node(self, causal_method, session):
        """Test continue_reasoning returns a ThoughtNode."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Root cause needed")
        result = await causal_method.continue_reasoning(session, first)

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_counter(self, causal_method, session):
        """Test continue_reasoning increments step number."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Analyze this")
        second = await causal_method.continue_reasoning(session, first)

        assert second.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(self, causal_method, session):
        """Test continue_reasoning sets parent_id to previous thought."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect to trace")
        second = await causal_method.continue_reasoning(session, first)

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_depth(self, causal_method, session):
        """Test continue_reasoning increases depth."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Initial effect")
        second = await causal_method.continue_reasoning(session, first)

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self, causal_method, session):
        """Test continue_reasoning with user guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "System failure")
        guidance = "Trace to root cause"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result is not None
        assert guidance in result.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(self, causal_method, session):
        """Test continue_reasoning without guidance (automatic progression)."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Performance issue")
        result = await causal_method.continue_reasoning(session, first)

        assert result is not None

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self, causal_method, session):
        """Test continue_reasoning accepts context parameter."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Error occurred")
        context: dict[str, Any] = {"new_data": "recent logs"}

        result = await causal_method.continue_reasoning(
            session, first, context=context
        )

        assert result is not None
        assert result.metadata.get("context") == context

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_to_session(self, causal_method, session):
        """Test continue_reasoning adds thought to session."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Problem")
        initial_count = session.thought_count

        await causal_method.continue_reasoning(session, first)

        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_default_progression_to_causal_tracing(self, causal_method, session):
        """Test continue_reasoning progresses to causal_tracing stage by default."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)

        assert second.metadata.get("stage") == "causal_tracing"

    @pytest.mark.asyncio
    async def test_continue_reasoning_stage_progression(self, causal_method, session):
        """Test continue_reasoning follows stage progression."""
        await causal_method.initialize()

        # Initial -> causal_tracing
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)
        assert second.metadata.get("stage") == "causal_tracing"

        # causal_tracing -> root_cause_identification
        third = await causal_method.continue_reasoning(session, second)
        assert third.metadata.get("stage") == "root_cause_identification"


class TestChainConstruction:
    """Tests for building multi-step causal chains."""

    @pytest.mark.asyncio
    async def test_causal_tracing_stage_content(self, causal_method, session):
        """Test causal tracing stage generates appropriate content."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)

        assert "Causal Chain" in second.content
        assert "Immediate Causes" in second.content

    @pytest.mark.asyncio
    async def test_causal_chain_tracks_depth(self, causal_method, session):
        """Test causal chain depth is tracked in metadata."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)

        assert "causal_chain_depth" in second.metadata

    @pytest.mark.asyncio
    async def test_confounding_factors_mentioned(self, causal_method, session):
        """Test confounding factors are considered in causal tracing."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Correlation observed")
        second = await causal_method.continue_reasoning(session, first)

        assert "Confounding" in second.content or "confound" in second.content.lower()

    @pytest.mark.asyncio
    async def test_temporal_sequence_mentioned(self, causal_method, session):
        """Test temporal sequence is mentioned in causal analysis."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Event sequence")
        second = await causal_method.continue_reasoning(session, first)

        assert "temporal" in second.content.lower() or "sequence" in second.content.lower()


class TestRootCauseIdentification:
    """Tests for root cause identification."""

    @pytest.mark.asyncio
    async def test_root_cause_stage_with_guidance(self, causal_method, session):
        """Test root cause identification stage is triggered by guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        guidance = "Identify the root cause"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result.metadata.get("stage") == "root_cause_identification"

    @pytest.mark.asyncio
    async def test_root_cause_stage_content(self, causal_method, session):
        """Test root cause stage generates appropriate content."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="Find fundamental cause"
        )

        assert "Root Cause" in second.content
        assert "Fundamental" in second.content or "fundamental" in second.content.lower()

    @pytest.mark.asyncio
    async def test_root_cause_tracks_count(self, causal_method, session):
        """Test root causes identified count is tracked."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="root cause"
        )

        assert "root_causes_identified" in second.metadata

    @pytest.mark.asyncio
    async def test_root_cause_five_whys_mentioned(self, causal_method, session):
        """Test Five Whys technique is mentioned in root cause analysis."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="root cause"
        )

        assert "Five Whys" in second.content or "Whys" in second.content


class TestEffectPrediction:
    """Tests for predicting downstream effects."""

    @pytest.mark.asyncio
    async def test_effect_prediction_stage_with_guidance(self, causal_method, session):
        """Test effect prediction stage is triggered by guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Cause identified")
        guidance = "Predict the effects"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result.metadata.get("stage") == "effect_prediction"

    @pytest.mark.asyncio
    async def test_effect_prediction_content(self, causal_method, session):
        """Test effect prediction stage generates appropriate content."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Root cause found")
        second = await causal_method.continue_reasoning(
            session, first, guidance="predict effects"
        )

        assert "Effect Prediction" in second.content
        assert "Direct Effects" in second.content

    @pytest.mark.asyncio
    async def test_cascading_effects_mentioned(self, causal_method, session):
        """Test cascading effects are considered in predictions."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Primary cause")
        second = await causal_method.continue_reasoning(
            session, first, guidance="predict outcomes"
        )

        assert "Cascading" in second.content or "cascading" in second.content.lower()

    @pytest.mark.asyncio
    async def test_side_effects_mentioned(self, causal_method, session):
        """Test side effects and unintended consequences are considered."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Intervention planned")
        second = await causal_method.continue_reasoning(
            session, first, guidance="predict effects"
        )

        assert "Side Effects" in second.content or "Unintended" in second.content


class TestMechanismExplanation:
    """Tests for explaining HOW causes lead to effects."""

    @pytest.mark.asyncio
    async def test_validation_stage_discusses_mechanism(self, causal_method, session):
        """Test validation stage discusses causal mechanism."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="validate the causal claim"
        )

        assert "Mechanism" in second.content or "mechanism" in second.content.lower()

    @pytest.mark.asyncio
    async def test_validation_stage_with_guidance(self, causal_method, session):
        """Test validation stage is triggered by guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Hypothesis formed")
        guidance = "Validate this causal relationship"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result.metadata.get("stage") == "validation"

    @pytest.mark.asyncio
    async def test_validation_checks_temporal_precedence(self, causal_method, session):
        """Test validation stage checks temporal precedence."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Causal claim")
        second = await causal_method.continue_reasoning(
            session, first, guidance="verify"
        )

        assert "Temporal Precedence" in second.content

    @pytest.mark.asyncio
    async def test_validation_checks_covariation(self, causal_method, session):
        """Test validation stage checks covariation."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Relationship observed")
        second = await causal_method.continue_reasoning(
            session, first, guidance="validate"
        )

        assert "Covariation" in second.content

    @pytest.mark.asyncio
    async def test_validation_considers_alternatives(self, causal_method, session):
        """Test validation stage considers alternative explanations."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Suspected cause")
        second = await causal_method.continue_reasoning(
            session, first, guidance="validate"
        )

        assert "Alternative" in second.content or "alternative" in second.content.lower()


class TestBranching:
    """Tests for exploring alternative causal paths (branching)."""

    @pytest.mark.asyncio
    async def test_alternative_path_stage_with_guidance(self, causal_method, session):
        """Test alternative path stage is triggered by guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Primary hypothesis")
        guidance = "Explore an alternative causal path"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result.metadata.get("stage") == "alternative_path"

    @pytest.mark.asyncio
    async def test_alternative_path_thought_type(self, causal_method, session):
        """Test alternative path creates BRANCH thought type."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Main path")
        second = await causal_method.continue_reasoning(
            session, first, guidance="branch to alternative"
        )

        assert second.type == ThoughtType.BRANCH

    @pytest.mark.asyncio
    async def test_alternative_path_content(self, causal_method, session):
        """Test alternative path generates appropriate content."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Hypothesis A")
        second = await causal_method.continue_reasoning(
            session, first, guidance="alternative"
        )

        assert "Alternative" in second.content
        assert "hypothesis" in second.content.lower()

    @pytest.mark.asyncio
    async def test_branch_guidance_creates_branch_type(self, causal_method, session):
        """Test 'branch' in guidance creates BRANCH thought type."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="branch"
        )

        assert second.type == ThoughtType.BRANCH


class TestSynthesis:
    """Tests for synthesizing causal understanding."""

    @pytest.mark.asyncio
    async def test_synthesis_stage_with_guidance(self, causal_method, session):
        """Test synthesis stage is triggered by guidance."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Analysis complete")
        guidance = "Synthesize the findings"

        result = await causal_method.continue_reasoning(
            session, first, guidance=guidance
        )

        assert result.metadata.get("stage") == "synthesis"

    @pytest.mark.asyncio
    async def test_synthesis_thought_type(self, causal_method, session):
        """Test synthesis stage creates SYNTHESIS thought type."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Data collected")
        second = await causal_method.continue_reasoning(
            session, first, guidance="synthesize"
        )

        assert second.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_synthesis_content(self, causal_method, session):
        """Test synthesis stage generates appropriate content."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Multiple paths analyzed")
        second = await causal_method.continue_reasoning(
            session, first, guidance="summary"
        )

        assert "Synthesis" in second.content or "synthesis" in second.content.lower()
        assert "Complete Causal Chain" in second.content or "Causal Map" in second.content

    @pytest.mark.asyncio
    async def test_synthesis_discusses_confidence(self, causal_method, session):
        """Test synthesis discusses confidence assessment."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Analysis done")
        second = await causal_method.continue_reasoning(
            session, first, guidance="synthesize"
        )

        assert "Confidence" in second.content or "confidence" in second.content.lower()


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_initial_thought_has_moderate_confidence(self, causal_method, session):
        """Test initial thought has moderate confidence (0.6)."""
        await causal_method.initialize()
        result = await causal_method.execute(session, "Effect")

        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_causal_tracing_increases_confidence(self, causal_method, session):
        """Test causal tracing stage increases confidence."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)

        assert second.confidence > first.confidence

    @pytest.mark.asyncio
    async def test_root_cause_increases_confidence_more(self, causal_method, session):
        """Test root cause identification increases confidence significantly."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(
            session, first, guidance="root cause"
        )

        # Root cause should have higher confidence boost
        assert second.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_validation_increases_confidence(self, causal_method, session):
        """Test validation stage increases confidence."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Hypothesis")
        second = await causal_method.continue_reasoning(
            session, first, guidance="validate"
        )

        assert second.confidence > first.confidence

    @pytest.mark.asyncio
    async def test_effect_prediction_decreases_confidence(self, causal_method, session):
        """Test effect prediction stage decreases confidence (predictions uncertain)."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Cause")
        second = await causal_method.continue_reasoning(
            session, first, guidance="predict"
        )

        # Predictions are less certain
        assert second.confidence <= first.confidence

    @pytest.mark.asyncio
    async def test_alternative_path_decreases_confidence(self, causal_method, session):
        """Test alternative path decreases confidence (introduces uncertainty)."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Main hypothesis")
        second = await causal_method.continue_reasoning(
            session, first, guidance="alternative"
        )

        # Alternatives introduce uncertainty
        assert second.confidence < first.confidence

    @pytest.mark.asyncio
    async def test_confidence_stays_in_valid_range(self, causal_method, session):
        """Test confidence stays within valid range [0.3, 0.95]."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")

        # Run multiple continuations
        current = first
        for _ in range(5):
            current = await causal_method.continue_reasoning(session, current)
            assert 0.3 <= current.confidence <= 0.95


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_input_text(self, causal_method, session):
        """Test handling of empty input text."""
        await causal_method.initialize()
        result = await causal_method.execute(session, "")

        assert result is not None
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_very_long_input_text(self, causal_method, session):
        """Test handling of very long input text."""
        await causal_method.initialize()
        long_effect = " ".join([
            "The system experienced multiple cascading failures",
            "starting with database connection pool exhaustion",
            "leading to API timeouts which triggered circuit breakers",
            "causing cache invalidation storms and eventual service degradation",
            "across all microservices in the production environment"
        ])
        result = await causal_method.execute(session, long_effect)

        assert result is not None

    @pytest.mark.asyncio
    async def test_direct_causation_scenario(self, causal_method, session):
        """Test handling of direct causation (no intermediate steps)."""
        await causal_method.initialize()
        effect = "Button was clicked and form submitted"
        result = await causal_method.execute(session, effect)

        assert result is not None
        assert "direct" in result.content.lower()

    @pytest.mark.asyncio
    async def test_complex_causal_chain_scenario(self, causal_method, session):
        """Test handling of complex multi-step causal chains."""
        await causal_method.initialize()
        effect = "Systemic failure across multiple dependent systems"
        result = await causal_method.execute(session, effect)

        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_causes_scenario(self, causal_method, session):
        """Test handling of effects with multiple contributing causes."""
        await causal_method.initialize()
        effect = "Performance degradation from multiple simultaneous issues"
        result = await causal_method.execute(session, effect)

        assert result is not None

    @pytest.mark.asyncio
    async def test_circular_causation(self, causal_method, session):
        """Test handling of circular/feedback loop causation."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Feedback loop detected")
        second = await causal_method.continue_reasoning(session, first)

        assert "feedback" in second.content.lower() or "circular" in second.content.lower()

    @pytest.mark.asyncio
    async def test_correlation_vs_causation(self, causal_method, session):
        """Test distinction between correlation and causation."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Two variables correlate")
        second = await causal_method.continue_reasoning(session, first)

        assert "correlation" in second.content.lower() or "causation" in second.content.lower()

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self, causal_method, session):
        """Test session metrics are properly updated during execution."""
        await causal_method.initialize()
        await causal_method.execute(session, "Effect to analyze")

        # Verify metrics are updated
        assert session.metrics.total_thoughts == 1
        assert session.metrics.max_depth_reached == 0
        assert session.metrics.average_confidence > 0.0
        assert session.metrics.methods_used[MethodIdentifier.CAUSAL_REASONING] == 1

    @pytest.mark.asyncio
    async def test_multiple_continuations(self, causal_method, session):
        """Test multiple continuations build causal chain."""
        await causal_method.initialize()

        first = await causal_method.execute(session, "Effect observed")
        second = await causal_method.continue_reasoning(session, first)
        third = await causal_method.continue_reasoning(session, second)
        fourth = await causal_method.continue_reasoning(session, third)

        assert session.thought_count == 4
        assert fourth.step_number == 4
        assert fourth.depth == 3

    @pytest.mark.asyncio
    async def test_thought_graph_structure(self, causal_method, session):
        """Test thought graph has proper structure."""
        await causal_method.initialize()

        first = await causal_method.execute(session, "Root effect")
        second = await causal_method.continue_reasoning(session, first)

        # Verify graph structure
        assert session.graph.node_count == 2
        assert session.graph.edge_count == 1

        # Root should be the initial thought
        assert session.graph.root_id is not None
        root_node = session.graph.get_node(session.graph.root_id)
        assert root_node is not None
        assert root_node.id == first.id

    @pytest.mark.asyncio
    async def test_counterfactual_reasoning(self, causal_method, session):
        """Test 'what if cause absent' counterfactual reasoning."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect with known cause")
        second = await causal_method.continue_reasoning(session, first)

        # Causal tracing should mention what would happen if cause absent
        content_lower = second.content.lower()
        assert "absent" in content_lower or "without" in content_lower or "prevent" in content_lower

    @pytest.mark.asyncio
    async def test_reasoning_type_preserved(self, causal_method, session):
        """Test reasoning_type metadata is preserved across continuations."""
        await causal_method.initialize()

        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)
        third = await causal_method.continue_reasoning(session, second)

        assert first.metadata.get("reasoning_type") == "causal"
        assert second.metadata.get("reasoning_type") == "causal"
        assert third.metadata.get("reasoning_type") == "causal"

    @pytest.mark.asyncio
    async def test_metadata_includes_input(self, causal_method, session):
        """Test initial thought metadata includes input text."""
        await causal_method.initialize()
        input_text = "Specific problem to analyze"
        result = await causal_method.execute(session, input_text)

        assert result.metadata.get("input") == input_text

    @pytest.mark.asyncio
    async def test_metadata_tracks_previous_step(self, causal_method, session):
        """Test continuation metadata tracks previous step number."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")
        second = await causal_method.continue_reasoning(session, first)

        assert second.metadata.get("previous_step") == first.step_number

    @pytest.mark.asyncio
    async def test_generic_continuation_fallback(self, causal_method, session):
        """Test continuation when stage is unknown falls back to causal_tracing."""
        await causal_method.initialize()
        first = await causal_method.execute(session, "Effect")

        # Manually set an unknown stage - implementation falls back to causal_tracing
        first.metadata["stage"] = "unknown_stage"

        second = await causal_method.continue_reasoning(session, first)

        # Unknown stage falls back to causal_tracing stage per _determine_stage
        assert second.metadata["stage"] == "causal_tracing"
        assert "Causal Chain" in second.content
