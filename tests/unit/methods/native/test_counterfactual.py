"""Unit tests for Counterfactual Reasoning method.

This module provides comprehensive test coverage for the Counterfactual class:
- Initialization and health checks
- Basic execution and baseline establishment
- Scenario generation and exploration
- Configuration options (scenario_count, divergence_point)
- Continue reasoning with different stages
- Antecedent modification (changing key conditions)
- Consequence tracing (effects of changes)
- Scenario comparison (actual vs counterfactual)
- Insight extraction from counterfactual analysis
- Edge cases (minimal/radical changes, multiple scenarios)
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.counterfactual import (
    COUNTERFACTUAL_METADATA,
    Counterfactual,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def counterfactual_method():
    """Create a Counterfactual method instance."""
    return Counterfactual()


@pytest.fixture
def session():
    """Create a fresh session for testing."""
    return Session().start()


# ============================================================================
# Metadata Tests
# ============================================================================


class TestCounterfactualMetadata:
    """Tests for COUNTERFACTUAL_METADATA."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert COUNTERFACTUAL_METADATA.identifier == MethodIdentifier.COUNTERFACTUAL

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert COUNTERFACTUAL_METADATA.name == "Counterfactual Reasoning"

    def test_metadata_category(self):
        """Test metadata is in ADVANCED category."""
        assert COUNTERFACTUAL_METADATA.category == MethodCategory.ADVANCED

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {
            "counterfactual",
            "what-if",
            "scenarios",
            "alternatives",
            "decision-analysis",
            "causality",
            "comparison",
            "hypothetical",
        }
        assert COUNTERFACTUAL_METADATA.tags == frozenset(expected_tags)

    def test_metadata_complexity(self):
        """Test metadata complexity level."""
        assert COUNTERFACTUAL_METADATA.complexity == 4

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert COUNTERFACTUAL_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert COUNTERFACTUAL_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context is not required."""
        assert COUNTERFACTUAL_METADATA.requires_context is False

    def test_metadata_thought_bounds(self):
        """Test metadata thought count bounds."""
        assert COUNTERFACTUAL_METADATA.min_thoughts == 5
        assert COUNTERFACTUAL_METADATA.max_thoughts == 0  # Unlimited

    def test_metadata_avg_tokens(self):
        """Test metadata average tokens per thought."""
        assert COUNTERFACTUAL_METADATA.avg_tokens_per_thought == 400

    def test_metadata_best_for(self):
        """Test metadata best use cases."""
        assert len(COUNTERFACTUAL_METADATA.best_for) > 0
        assert "decision analysis and evaluation" in COUNTERFACTUAL_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata not recommended use cases."""
        assert len(COUNTERFACTUAL_METADATA.not_recommended_for) > 0
        assert "simple yes/no questions" in COUNTERFACTUAL_METADATA.not_recommended_for


# ============================================================================
# Initialization Tests
# ============================================================================


class TestCounterfactualInitialization:
    """Tests for initialization and health check."""

    def test_initial_state(self, counterfactual_method):
        """Test method is not initialized on creation."""
        assert counterfactual_method._initialized is False
        assert counterfactual_method._step_counter == 0
        assert counterfactual_method._current_stage == "baseline"
        assert counterfactual_method._baseline_established is False
        assert counterfactual_method._variables_identified is False
        assert counterfactual_method._scenarios == []

    def test_identifier_property(self, counterfactual_method):
        """Test identifier property returns correct value."""
        assert counterfactual_method.identifier == MethodIdentifier.COUNTERFACTUAL

    def test_name_property(self, counterfactual_method):
        """Test name property returns correct value."""
        assert counterfactual_method.name == "Counterfactual Reasoning"

    def test_description_property(self, counterfactual_method):
        """Test description property returns correct value."""
        assert "What-if analysis" in counterfactual_method.description

    def test_category_property(self, counterfactual_method):
        """Test category property returns correct value."""
        assert counterfactual_method.category == MethodCategory.ADVANCED

    @pytest.mark.asyncio
    async def test_initialize(self, counterfactual_method):
        """Test initialize method sets initialized flag and resets state."""
        await counterfactual_method.initialize()
        assert counterfactual_method._initialized is True
        assert counterfactual_method._step_counter == 0
        assert counterfactual_method._current_stage == "baseline"
        assert counterfactual_method._baseline_established is False
        assert counterfactual_method._variables_identified is False
        assert counterfactual_method._scenarios == []

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, counterfactual_method):
        """Test health check returns False before initialization."""
        healthy = await counterfactual_method.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, counterfactual_method):
        """Test health check returns True after initialization."""
        await counterfactual_method.initialize()
        healthy = await counterfactual_method.health_check()
        assert healthy is True


# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestCounterfactualExecution:
    """Tests for basic execution flow."""

    @pytest.mark.asyncio
    async def test_execute_requires_initialization(self, counterfactual_method, session):
        """Test execute raises error if not initialized."""
        with pytest.raises(RuntimeError, match="must be initialized before execution"):
            await counterfactual_method.execute(session, "What if I took the job?")

    @pytest.mark.asyncio
    async def test_execute_returns_initial_thought(self, counterfactual_method, session):
        """Test execute returns a ThoughtNode with INITIAL type."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "What if I invested in stocks?")
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_execute_establishes_baseline(self, counterfactual_method, session):
        """Test execute establishes baseline scenario."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "What if I had studied abroad?")
        assert result.metadata["stage"] == "baseline"
        assert result.metadata["baseline_established"] is True
        assert counterfactual_method._baseline_established is True
        assert "baseline" in result.content.lower()

    @pytest.mark.asyncio
    async def test_execute_sets_step_number(self, counterfactual_method, session):
        """Test execute sets step number to 1."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "Decision question")
        assert result.step_number == 1
        assert counterfactual_method._step_counter == 1

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(self, counterfactual_method, session):
        """Test execute sets appropriate confidence for baseline."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "Investment decision")
        assert result.confidence == 0.8  # High confidence in baseline facts

    @pytest.mark.asyncio
    async def test_execute_with_context(self, counterfactual_method, session):
        """Test execute accepts and processes context parameter."""
        await counterfactual_method.initialize()
        context: dict[str, Any] = {"baseline": "Chose bonds over stocks in 2020"}
        result = await counterfactual_method.execute(
            session, "Investment decision", context=context
        )
        assert result.metadata["context"] == context
        assert "bonds over stocks" in result.content

    @pytest.mark.asyncio
    async def test_execute_without_context(self, counterfactual_method, session):
        """Test execute works without context parameter."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "Career decision")
        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self, counterfactual_method, session):
        """Test execute adds thought to session."""
        await counterfactual_method.initialize()
        await counterfactual_method.execute(session, "Decision analysis")
        assert session.thought_count == 1
        assert session.current_method == MethodIdentifier.COUNTERFACTUAL

    @pytest.mark.asyncio
    async def test_execute_resets_scenarios(self, counterfactual_method, session):
        """Test execute resets scenarios list."""
        await counterfactual_method.initialize()
        counterfactual_method._scenarios = ["old_scenario"]
        await counterfactual_method.execute(session, "New decision")
        assert counterfactual_method._scenarios == []


# ============================================================================
# Continue Reasoning Tests
# ============================================================================


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_requires_initialization(self, counterfactual_method, session):
        """Test continue_reasoning raises error if not initialized."""
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COUNTERFACTUAL,
            content="Test",
            metadata={"stage": "baseline"},
        )
        with pytest.raises(RuntimeError, match="must be initialized before continuation"):
            await counterfactual_method.continue_reasoning(session, thought)

    @pytest.mark.asyncio
    async def test_continue_from_baseline_identifies_variables(
        self, counterfactual_method, session
    ):
        """Test continuing from baseline identifies variables."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Investment decision")

        next_thought = await counterfactual_method.continue_reasoning(
            session, baseline, guidance="Identify key variables"
        )

        assert next_thought.metadata["stage"] == "variables"
        assert next_thought.type == ThoughtType.CONTINUATION
        assert counterfactual_method._variables_identified is True
        assert "variable" in next_thought.content.lower()

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(self, counterfactual_method, session):
        """Test continue_reasoning increments step counter."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        assert counterfactual_method._step_counter == 1

        next_thought = await counterfactual_method.continue_reasoning(session, baseline)
        assert counterfactual_method._step_counter == 2
        assert next_thought.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_sets_parent_id(self, counterfactual_method, session):
        """Test continue_reasoning sets parent_id correctly."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        next_thought = await counterfactual_method.continue_reasoning(session, baseline)
        assert next_thought.parent_id == baseline.id

    @pytest.mark.asyncio
    async def test_continue_increases_depth(self, counterfactual_method, session):
        """Test continue_reasoning increases depth."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        assert baseline.depth == 0

        next_thought = await counterfactual_method.continue_reasoning(session, baseline)
        assert next_thought.depth == 1


# ============================================================================
# Scenario Generation Tests
# ============================================================================


class TestScenarioGeneration:
    """Tests for scenario generation and branching."""

    @pytest.mark.asyncio
    async def test_branch_creates_scenario(self, counterfactual_method, session):
        """Test branching creates a counterfactual scenario."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Investment decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if interest rates were 2% higher?",
            context={"branch": True},
        )

        assert scenario.type == ThoughtType.BRANCH
        assert scenario.metadata["stage"] == "scenarios"
        assert scenario.metadata["is_branch"] is True
        assert len(counterfactual_method._scenarios) == 1

    @pytest.mark.asyncio
    async def test_multiple_scenarios(self, counterfactual_method, session):
        """Test generating multiple counterfactual scenarios."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Career decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario1 = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if I accepted the job offer?",
            context={"branch": True},
        )
        scenario2 = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if I pursued further education?",
            context={"branch": True},
        )

        assert scenario1.type == ThoughtType.BRANCH
        assert scenario2.type == ThoughtType.BRANCH
        assert len(counterfactual_method._scenarios) == 2
        assert counterfactual_method._scenarios[0] == "What if I accepted the job offer?"
        assert counterfactual_method._scenarios[1] == "What if I pursued further education?"

    @pytest.mark.asyncio
    async def test_scenario_without_branch_flag(self, counterfactual_method, session):
        """Test scenario generation without branch flag."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario = await counterfactual_method.continue_reasoning(
            session, variables, guidance="Generate scenarios"
        )

        assert scenario.type == ThoughtType.CONTINUATION
        assert scenario.metadata["stage"] == "scenarios"
        # Without branch flag, is_branch should be False or not set
        assert scenario.metadata.get("is_branch") is not True

    @pytest.mark.asyncio
    async def test_scenario_content_includes_guidance(self, counterfactual_method, session):
        """Test scenario content includes provided guidance."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        guidance_text = "What if market conditions changed?"
        scenario = await counterfactual_method.continue_reasoning(
            session, variables, guidance=guidance_text, context={"branch": True}
        )

        assert guidance_text in scenario.content


# ============================================================================
# Stage Progression Tests
# ============================================================================


class TestStageProgression:
    """Tests for stage progression through counterfactual reasoning."""

    @pytest.mark.asyncio
    async def test_baseline_to_variables(self, counterfactual_method, session):
        """Test progression from baseline to variables stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        assert baseline.metadata["stage"] == "baseline"
        assert variables.metadata["stage"] == "variables"
        assert counterfactual_method._current_stage == "variables"

    @pytest.mark.asyncio
    async def test_variables_to_scenarios(self, counterfactual_method, session):
        """Test progression from variables to scenarios stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)
        scenarios = await counterfactual_method.continue_reasoning(session, variables)

        assert scenarios.metadata["stage"] == "scenarios"

    @pytest.mark.asyncio
    async def test_scenarios_to_analysis(self, counterfactual_method, session):
        """Test progression from scenarios to analysis stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)
        scenarios = await counterfactual_method.continue_reasoning(
            session, variables, context={"branch": True}
        )
        analysis = await counterfactual_method.continue_reasoning(session, scenarios)

        assert analysis.metadata["stage"] == "analysis"
        assert analysis.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_analysis_to_synthesis(self, counterfactual_method, session):
        """Test progression from analysis to synthesis stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)
        scenarios = await counterfactual_method.continue_reasoning(
            session, variables, context={"branch": True}
        )
        analysis = await counterfactual_method.continue_reasoning(session, scenarios)
        synthesis = await counterfactual_method.continue_reasoning(session, analysis)

        assert synthesis.metadata["stage"] == "synthesis"
        assert synthesis.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_explicit_synthesis_guidance(self, counterfactual_method, session):
        """Test explicit synthesis guidance triggers synthesis stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        synthesis = await counterfactual_method.continue_reasoning(
            session, baseline, guidance="Synthesize insights"
        )

        assert synthesis.metadata["stage"] == "synthesis"
        assert synthesis.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_explicit_conclusion_guidance(self, counterfactual_method, session):
        """Test explicit conclusion guidance triggers conclusion stage."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        conclusion = await counterfactual_method.continue_reasoning(
            session, baseline, guidance="Conclude analysis"
        )

        assert conclusion.metadata["stage"] == "conclusion"
        assert conclusion.type == ThoughtType.CONCLUSION


# ============================================================================
# Antecedent Modification Tests
# ============================================================================


class TestAntecedentModification:
    """Tests for modifying key conditions (antecedents) in scenarios."""

    @pytest.mark.asyncio
    async def test_minimal_change_scenario(self, counterfactual_method, session):
        """Test scenario with minimal change from baseline."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(
            session, "What if I arrived 5 minutes earlier?"
        )
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        minimal = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if timing was 5 minutes earlier?",
            context={"branch": True},
        )

        assert minimal.type == ThoughtType.BRANCH
        assert "5 minutes earlier" in minimal.content

    @pytest.mark.asyncio
    async def test_radical_change_scenario(self, counterfactual_method, session):
        """Test scenario with radical change from baseline."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(
            session, "What if I had chosen a completely different career?"
        )
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        radical = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if I became an artist instead of engineer?",
            context={"branch": True},
        )

        assert radical.type == ThoughtType.BRANCH
        assert "artist instead of engineer" in radical.content

    @pytest.mark.asyncio
    async def test_multiple_variable_modification(self, counterfactual_method, session):
        """Test modifying multiple variables simultaneously."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Investment outcome")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        multi_var = await counterfactual_method.continue_reasoning(
            session,
            variables,
            guidance="What if timing, amount, and asset class all differed?",
            context={"branch": True},
        )

        assert multi_var.type == ThoughtType.BRANCH
        assert len(counterfactual_method._scenarios) == 1


# ============================================================================
# Consequence Tracing Tests
# ============================================================================


class TestConsequenceTracing:
    """Tests for tracing effects of changes through scenarios."""

    @pytest.mark.asyncio
    async def test_analysis_stage_traces_consequences(self, counterfactual_method, session):
        """Test analysis stage traces consequences of counterfactual changes."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)
        scenario = await counterfactual_method.continue_reasoning(
            session, variables, context={"branch": True}
        )

        analysis = await counterfactual_method.continue_reasoning(session, scenario)

        assert analysis.metadata["stage"] == "analysis"
        assert "outcome" in analysis.content.lower() or "differ" in analysis.content.lower()

    @pytest.mark.asyncio
    async def test_scenarios_explored_tracked(self, counterfactual_method, session):
        """Test that scenarios_explored is tracked in metadata."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario1 = await counterfactual_method.continue_reasoning(
            session, variables, guidance="Scenario 1", context={"branch": True}
        )
        assert scenario1.metadata["scenarios_explored"] == 1

        scenario2 = await counterfactual_method.continue_reasoning(
            session, variables, guidance="Scenario 2", context={"branch": True}
        )
        assert scenario2.metadata["scenarios_explored"] == 2


# ============================================================================
# Scenario Comparison Tests
# ============================================================================


class TestScenarioComparison:
    """Tests for comparing actual vs counterfactual scenarios."""

    @pytest.mark.asyncio
    async def test_comparison_in_scenario_content(self, counterfactual_method, session):
        """Test that scenario content includes comparison to baseline."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario = await counterfactual_method.continue_reasoning(
            session, variables, guidance="Alternative scenario", context={"branch": True}
        )

        assert "baseline" in scenario.content.lower()
        assert "comparison" in scenario.content.lower()

    @pytest.mark.asyncio
    async def test_analysis_compares_multiple_scenarios(self, counterfactual_method, session):
        """Test analysis stage compares multiple scenarios."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        await counterfactual_method.continue_reasoning(
            session, variables, guidance="S1", context={"branch": True}
        )
        scenario2 = await counterfactual_method.continue_reasoning(
            session, variables, guidance="S2", context={"branch": True}
        )

        analysis = await counterfactual_method.continue_reasoning(session, scenario2)

        assert analysis.metadata["scenarios_explored"] == 2
        assert "2" in analysis.content  # Should mention exploring 2 scenarios


# ============================================================================
# Insight Extraction Tests
# ============================================================================


class TestInsightExtraction:
    """Tests for extracting insights from counterfactual analysis."""

    @pytest.mark.asyncio
    async def test_synthesis_extracts_insights(self, counterfactual_method, session):
        """Test synthesis stage extracts key insights."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)
        scenario = await counterfactual_method.continue_reasoning(
            session, variables, context={"branch": True}
        )
        analysis = await counterfactual_method.continue_reasoning(session, scenario)

        synthesis = await counterfactual_method.continue_reasoning(session, analysis)

        assert synthesis.type == ThoughtType.SYNTHESIS
        assert "insight" in synthesis.content.lower()

    @pytest.mark.asyncio
    async def test_conclusion_provides_recommendations(self, counterfactual_method, session):
        """Test conclusion provides actionable recommendations."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        conclusion = await counterfactual_method.continue_reasoning(
            session, baseline, guidance="Conclude the analysis"
        )

        assert conclusion.type == ThoughtType.CONCLUSION
        assert (
            "recommendation" in conclusion.content.lower()
            or "assessment" in conclusion.content.lower()
        )


# ============================================================================
# Confidence Calculation Tests
# ============================================================================


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    @pytest.mark.asyncio
    async def test_baseline_high_confidence(self, counterfactual_method, session):
        """Test baseline has high confidence."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        assert baseline.confidence == 0.8

    @pytest.mark.asyncio
    async def test_scenarios_moderate_confidence(self, counterfactual_method, session):
        """Test counterfactual scenarios have moderate confidence."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        scenario = await counterfactual_method.continue_reasoning(
            session, variables, context={"branch": True}
        )

        # Scenarios are hypothetical, so lower confidence
        assert scenario.confidence < baseline.confidence

    @pytest.mark.asyncio
    async def test_confidence_with_multiple_scenarios(self, counterfactual_method, session):
        """Test confidence increases with more scenarios analyzed."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        # Create multiple scenarios
        await counterfactual_method.continue_reasoning(
            session, variables, guidance="S1", context={"branch": True}
        )
        await counterfactual_method.continue_reasoning(
            session, variables, guidance="S2", context={"branch": True}
        )
        scenario3 = await counterfactual_method.continue_reasoning(
            session, variables, guidance="S3", context={"branch": True}
        )

        # Analysis with 3 scenarios should have reasonable confidence
        analysis = await counterfactual_method.continue_reasoning(session, scenario3)
        assert analysis.confidence >= 0.6  # Decent confidence with multiple scenarios

    @pytest.mark.asyncio
    async def test_conclusion_high_confidence(self, counterfactual_method, session):
        """Test conclusion has high confidence."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        conclusion = await counterfactual_method.continue_reasoning(
            session, baseline, guidance="Conclude"
        )

        assert conclusion.confidence >= 0.7  # Conclusions have higher confidence


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_execute_with_empty_input(self, counterfactual_method, session):
        """Test execute handles empty input text."""
        await counterfactual_method.initialize()
        result = await counterfactual_method.execute(session, "")
        assert result is not None
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_continue_without_guidance(self, counterfactual_method, session):
        """Test continue_reasoning works without guidance."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        next_thought = await counterfactual_method.continue_reasoning(session, baseline)
        assert next_thought is not None
        assert next_thought.metadata["guidance"] == ""

    @pytest.mark.asyncio
    async def test_continue_without_context(self, counterfactual_method, session):
        """Test continue_reasoning works without context."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        next_thought = await counterfactual_method.continue_reasoning(session, baseline)
        assert next_thought.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_deep_reasoning_chain(self, counterfactual_method, session):
        """Test deep chain of reasoning steps."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        current = baseline
        for _i in range(5):
            current = await counterfactual_method.continue_reasoning(session, current)

        assert current.depth == 5
        assert session.thought_count == 6  # baseline + 5 continuations

    @pytest.mark.asyncio
    async def test_confidence_stays_in_bounds(self, counterfactual_method, session):
        """Test confidence values stay within valid range."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")

        current = baseline
        for _ in range(10):
            current = await counterfactual_method.continue_reasoning(session, current)
            assert 0.0 <= current.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_scenario_count_tracking(self, counterfactual_method, session):
        """Test accurate tracking of scenario count."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        assert len(counterfactual_method._scenarios) == 0

        for i in range(5):
            await counterfactual_method.continue_reasoning(
                session, variables, guidance=f"Scenario {i + 1}", context={"branch": True}
            )

        assert len(counterfactual_method._scenarios) == 5

    @pytest.mark.asyncio
    async def test_state_reset_on_execute(self, counterfactual_method, session):
        """Test that execute resets state even with prior scenarios."""
        await counterfactual_method.initialize()

        # First execution
        baseline1 = await counterfactual_method.execute(session, "Decision 1")
        variables1 = await counterfactual_method.continue_reasoning(session, baseline1)
        await counterfactual_method.continue_reasoning(
            session, variables1, context={"branch": True}
        )
        assert len(counterfactual_method._scenarios) == 1

        # Second execution should reset
        session2 = Session().start()
        baseline2 = await counterfactual_method.execute(session2, "Decision 2")
        assert len(counterfactual_method._scenarios) == 0
        assert counterfactual_method._step_counter == 1
        assert baseline2.step_number == 1

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, counterfactual_method, session):
        """Test that metadata is properly preserved across thoughts."""
        await counterfactual_method.initialize()
        baseline = await counterfactual_method.execute(session, "Decision")
        variables = await counterfactual_method.continue_reasoning(session, baseline)

        assert variables.metadata["reasoning_type"] == "counterfactual"
        assert "stage" in variables.metadata
        assert "scenarios_explored" in variables.metadata
