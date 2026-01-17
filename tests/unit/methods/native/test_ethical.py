"""Unit tests for EthicalReasoning method."""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.ethical import (
    ETHICAL_REASONING_METADATA,
    EthicalReasoning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def ethical_method():
    """Create an EthicalReasoning method instance."""
    return EthicalReasoning()


@pytest.fixture
def session():
    """Create a fresh session for testing."""
    return Session().start()


class TestEthicalReasoningMetadata:
    """Tests for ETHICAL_REASONING_METADATA."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert ETHICAL_REASONING_METADATA.identifier == MethodIdentifier.ETHICAL_REASONING

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert ETHICAL_REASONING_METADATA.name == "Ethical Reasoning"

    def test_metadata_category(self):
        """Test metadata is in HIGH_VALUE category."""
        assert ETHICAL_REASONING_METADATA.category == MethodCategory.HIGH_VALUE

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {
            "ethics",
            "moral",
            "stakeholder",
            "multi-framework",
            "balanced",
            "decision-making",
        }
        assert ETHICAL_REASONING_METADATA.tags == frozenset(expected_tags)

    def test_metadata_complexity(self):
        """Test metadata complexity level."""
        assert ETHICAL_REASONING_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test metadata indicates branching support."""
        assert ETHICAL_REASONING_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test metadata indicates no revision support."""
        assert ETHICAL_REASONING_METADATA.supports_revision is False

    def test_metadata_requires_context(self):
        """Test metadata indicates context is not required."""
        assert ETHICAL_REASONING_METADATA.requires_context is False

    def test_metadata_thought_bounds(self):
        """Test metadata thought count bounds."""
        assert ETHICAL_REASONING_METADATA.min_thoughts == 6
        assert ETHICAL_REASONING_METADATA.max_thoughts == 15

    def test_metadata_avg_tokens(self):
        """Test metadata average tokens per thought."""
        assert ETHICAL_REASONING_METADATA.avg_tokens_per_thought == 600

    def test_metadata_best_for(self):
        """Test metadata best use cases."""
        assert len(ETHICAL_REASONING_METADATA.best_for) > 0
        assert "Ethical dilemmas and moral questions" in ETHICAL_REASONING_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata not recommended use cases."""
        assert len(ETHICAL_REASONING_METADATA.not_recommended_for) > 0
        assert "Pure technical problems" in ETHICAL_REASONING_METADATA.not_recommended_for


class TestEthicalReasoningInitialization:
    """Tests for initialization and health check."""

    def test_initial_state(self, ethical_method):
        """Test method is not initialized on creation."""
        assert ethical_method._initialized is False

    def test_identifier_property(self, ethical_method):
        """Test identifier property returns correct value."""
        assert ethical_method.identifier == str(MethodIdentifier.ETHICAL_REASONING)

    def test_name_property(self, ethical_method):
        """Test name property returns correct value."""
        assert ethical_method.name == "Ethical Reasoning"

    def test_description_property(self, ethical_method):
        """Test description property returns correct value."""
        assert ethical_method.description == "Multi-framework ethical analysis"

    def test_category_property(self, ethical_method):
        """Test category property returns correct value."""
        assert ethical_method.category == str(MethodCategory.HIGH_VALUE)

    @pytest.mark.asyncio
    async def test_initialize(self, ethical_method):
        """Test initialize method sets initialized flag."""
        await ethical_method.initialize()
        assert ethical_method._initialized is True

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, ethical_method):
        """Test health check returns False before initialization."""
        healthy = await ethical_method.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, ethical_method):
        """Test health check returns True after initialization."""
        await ethical_method.initialize()
        healthy = await ethical_method.health_check()
        assert healthy is True


class TestEthicalReasoningExecution:
    """Tests for basic execution flow."""

    @pytest.mark.asyncio
    async def test_execute_auto_initializes(self, ethical_method, session):
        """Test execute auto-initializes if not already initialized."""
        assert ethical_method._initialized is False
        scenario = "Should companies mandate vaccination for employees?"
        result = await ethical_method.execute(session, scenario)
        assert ethical_method._initialized is True
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_returns_conclusion(self, ethical_method, session):
        """Test execute returns a ThoughtNode with CONCLUSION type."""
        scenario = "Is it ethical to use AI for hiring decisions?"
        result = await ethical_method.execute(session, scenario)
        assert isinstance(result, ThoughtNode)
        assert result.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_execute_creates_expected_thought_count(self, ethical_method, session):
        """Test execute creates expected number of thoughts (8 steps)."""
        scenario = "Should autonomous vehicles prioritize passenger or pedestrian safety?"
        await ethical_method.execute(session, scenario)
        # Execute should create 8 thoughts: understanding, stakeholders, 4 frameworks, comparison, recommendation
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_execute_with_context(self, ethical_method, session):
        """Test execute accepts and processes context parameter."""
        scenario = "Should we prioritize profit or environmental protection?"
        context: dict[str, Any] = {
            "industry": "manufacturing",
            "stakeholders": ["workers", "community"],
        }
        result = await ethical_method.execute(session, scenario, context=context)
        assert result is not None
        assert result.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_execute_without_context(self, ethical_method, session):
        """Test execute works without context parameter."""
        scenario = "Is it ethical to conduct animal testing for cosmetics?"
        result = await ethical_method.execute(session, scenario, context=None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_conclusion_has_high_confidence(self, ethical_method, session):
        """Test final conclusion has expected confidence level."""
        scenario = "Should we implement universal basic income?"
        result = await ethical_method.execute(session, scenario)
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_conclusion_has_quality_score(self, ethical_method, session):
        """Test final conclusion has quality score."""
        scenario = "Is it ethical to use facial recognition in public spaces?"
        result = await ethical_method.execute(session, scenario)
        assert result.quality_score is not None
        assert result.quality_score == 0.9


class TestFrameworkAnalysis:
    """Tests for individual ethical framework analyses."""

    @pytest.mark.asyncio
    async def test_consequentialist_framework_applied(self, ethical_method, session):
        """Test consequentialist framework is applied."""
        scenario = "Should we ban single-use plastics?"
        await ethical_method.execute(session, scenario)

        # Find the consequentialist thought
        thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("framework") == "consequentialist"
        ]
        assert len(thoughts) == 1
        assert "Consequentialist" in thoughts[0].content

    @pytest.mark.asyncio
    async def test_deontological_framework_applied(self, ethical_method, session):
        """Test deontological framework is applied."""
        scenario = "Is it ethical to lie to protect someone's feelings?"
        await ethical_method.execute(session, scenario)

        # Find the deontological thought
        thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("framework") == "deontological"
        ]
        assert len(thoughts) == 1
        assert "Deontological" in thoughts[0].content

    @pytest.mark.asyncio
    async def test_virtue_ethics_framework_applied(self, ethical_method, session):
        """Test virtue ethics framework is applied."""
        scenario = "How should I handle conflicts at work?"
        await ethical_method.execute(session, scenario)

        # Find the virtue ethics thought
        thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("framework") == "virtue_ethics"
        ]
        assert len(thoughts) == 1
        assert "Virtue Ethics" in thoughts[0].content

    @pytest.mark.asyncio
    async def test_care_ethics_framework_applied(self, ethical_method, session):
        """Test care ethics framework is applied."""
        scenario = "Should elderly parents move in with their children?"
        await ethical_method.execute(session, scenario)

        # Find the care ethics thought
        thoughts = [
            t for t in session.graph.nodes.values() if t.metadata.get("framework") == "care_ethics"
        ]
        assert len(thoughts) == 1
        assert "Care Ethics" in thoughts[0].content


class TestMultiFrameworkAnalysis:
    """Tests for multi-framework analysis."""

    @pytest.mark.asyncio
    async def test_all_four_frameworks_applied(self, ethical_method, session):
        """Test all 4 frameworks are applied independently."""
        scenario = "Should we implement mandatory organ donation?"
        await ethical_method.execute(session, scenario)

        # Check all 4 frameworks are present
        frameworks = {
            t.metadata.get("framework")
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "framework_analysis"
        }
        assert frameworks == {"consequentialist", "deontological", "virtue_ethics", "care_ethics"}

    @pytest.mark.asyncio
    async def test_framework_thoughts_are_branches(self, ethical_method, session):
        """Test framework thoughts are marked as branches."""
        scenario = "Is capital punishment ethical?"
        await ethical_method.execute(session, scenario)

        # All framework thoughts should be branches
        framework_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "framework_analysis"
        ]
        assert len(framework_thoughts) == 4
        for thought in framework_thoughts:
            assert thought.type == ThoughtType.BRANCH
            assert thought.branch_id is not None

    @pytest.mark.asyncio
    async def test_framework_thoughts_share_parent(self, ethical_method, session):
        """Test all framework thoughts branch from stakeholder analysis."""
        scenario = "Should gene editing be allowed for human embryos?"
        await ethical_method.execute(session, scenario)

        # Get stakeholder thought
        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        assert len(stakeholder_thoughts) == 1
        stakeholder_id = stakeholder_thoughts[0].id

        # All framework thoughts should have same parent
        framework_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "framework_analysis"
        ]
        assert len(framework_thoughts) == 4
        for thought in framework_thoughts:
            assert thought.parent_id == stakeholder_id

    @pytest.mark.asyncio
    async def test_frameworks_have_consistent_confidence(self, ethical_method, session):
        """Test framework thoughts have consistent confidence scores."""
        scenario = "Is it ethical to use drones in warfare?"
        await ethical_method.execute(session, scenario)

        framework_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "framework_analysis"
        ]
        confidences = {t.confidence for t in framework_thoughts}
        # All frameworks should have same confidence (0.75)
        assert confidences == {0.75}


class TestStakeholderIdentification:
    """Tests for stakeholder identification."""

    @pytest.mark.asyncio
    async def test_stakeholder_identification_with_employees(self, ethical_method, session):
        """Test stakeholder identification detects employees."""
        scenario = "Should employees be required to work overtime?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        assert len(stakeholder_thoughts) == 1
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        assert "Employees" in stakeholders

    @pytest.mark.asyncio
    async def test_stakeholder_identification_with_customers(self, ethical_method, session):
        """Test stakeholder identification detects customers."""
        scenario = "Should we collect customer data for targeted advertising?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        assert "Customers/Clients" in stakeholders

    @pytest.mark.asyncio
    async def test_stakeholder_identification_with_public(self, ethical_method, session):
        """Test stakeholder identification detects public/community."""
        scenario = "Should the government increase taxes for public services?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        assert "Public/Community" in stakeholders or "Regulators/Government" in stakeholders

    @pytest.mark.asyncio
    async def test_stakeholder_identification_with_company(self, ethical_method, session):
        """Test stakeholder identification detects organizations."""
        scenario = "Should the company invest in renewable energy?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        assert "Organization/Company" in stakeholders

    @pytest.mark.asyncio
    async def test_stakeholder_identification_with_patients(self, ethical_method, session):
        """Test stakeholder identification detects healthcare recipients."""
        scenario = "Should patients have access to experimental treatments?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        assert "Patients/Healthcare Recipients" in stakeholders

    @pytest.mark.asyncio
    async def test_stakeholder_identification_generic_fallback(self, ethical_method, session):
        """Test stakeholder identification uses generic categories as fallback."""
        scenario = "What is the right thing to do in this situation?"
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        # Should have generic categories
        assert len(stakeholders) > 0
        assert any(
            s in ["Directly Affected Individuals", "Decision Makers", "Broader Community"]
            for s in stakeholders
        )


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self, ethical_method, session):
        """Test continue_reasoning with user guidance."""
        scenario = "Should we implement workplace surveillance?"
        result = await ethical_method.execute(session, scenario)

        guidance = "Consider the perspective of remote workers"
        continuation = await ethical_method.continue_reasoning(session, result, guidance=guidance)

        assert continuation is not None
        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == result.id
        assert guidance in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(self, ethical_method, session):
        """Test continue_reasoning without guidance (automatic)."""
        scenario = "Is it ethical to use child labor in supply chains?"
        result = await ethical_method.execute(session, scenario)

        continuation = await ethical_method.continue_reasoning(session, result)

        assert continuation is not None
        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.parent_id == result.id
        assert "Deepening ethical analysis" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self, ethical_method, session):
        """Test continue_reasoning accepts context parameter."""
        scenario = "Should we automate jobs with AI?"
        result = await ethical_method.execute(session, scenario)

        context: dict[str, Any] = {"new_info": "unemployment concerns"}
        continuation = await ethical_method.continue_reasoning(session, result, context=context)

        assert continuation is not None

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_depth(self, ethical_method, session):
        """Test continue_reasoning increases depth."""
        scenario = "Should we legalize euthanasia?"
        result = await ethical_method.execute(session, scenario)

        continuation = await ethical_method.continue_reasoning(session, result)

        assert continuation.depth == result.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step_number(self, ethical_method, session):
        """Test continue_reasoning increases step number."""
        scenario = "Is it ethical to clone humans?"
        result = await ethical_method.execute(session, scenario)

        continuation = await ethical_method.continue_reasoning(session, result)

        assert continuation.step_number == result.step_number + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_added_to_session(self, ethical_method, session):
        """Test continue_reasoning adds thought to session."""
        scenario = "Should we ban cryptocurrency?"
        result = await ethical_method.execute(session, scenario)
        initial_count = session.thought_count

        await ethical_method.continue_reasoning(session, result)

        assert session.thought_count == initial_count + 1


class TestRecommendationSynthesis:
    """Tests for final recommendation synthesis."""

    @pytest.mark.asyncio
    async def test_recommendation_contains_balanced_approach(self, ethical_method, session):
        """Test recommendation synthesizes multiple frameworks."""
        scenario = "Should we implement carbon taxes?"
        result = await ethical_method.execute(session, scenario)

        # Recommendation should mention balanced approach
        assert "Balanced" in result.content or "balanced" in result.content

    @pytest.mark.asyncio
    async def test_recommendation_metadata_tracks_frameworks(self, ethical_method, session):
        """Test recommendation metadata tracks framework count."""
        scenario = "Is it ethical to use surveillance cameras in workplaces?"
        result = await ethical_method.execute(session, scenario)

        assert result.metadata.get("frameworks_synthesized") == 4

    @pytest.mark.asyncio
    async def test_recommendation_metadata_tracks_stakeholders(self, ethical_method, session):
        """Test recommendation metadata tracks stakeholder count."""
        scenario = "Should companies be required to disclose their environmental impact?"
        result = await ethical_method.execute(session, scenario)

        stakeholder_count = result.metadata.get("stakeholders_considered")
        assert stakeholder_count is not None
        assert stakeholder_count > 0

    @pytest.mark.asyncio
    async def test_synthesis_step_exists(self, ethical_method, session):
        """Test framework comparison/synthesis step exists."""
        scenario = "Should we allow genetic modification of crops?"
        await ethical_method.execute(session, scenario)

        # Find synthesis/comparison thought
        synthesis_thoughts = [
            t for t in session.graph.nodes.values() if t.type == ThoughtType.SYNTHESIS
        ]
        assert len(synthesis_thoughts) == 1
        assert "Cross-Framework" in synthesis_thoughts[0].content


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_clear_cut_ethical_case(self, ethical_method, session):
        """Test handling of clear-cut ethical scenarios."""
        scenario = "Is murder wrong?"
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        assert result.type == ThoughtType.CONCLUSION
        # Should still apply all frameworks even for clear cases
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_genuine_ethical_dilemma(self, ethical_method, session):
        """Test handling of genuine ethical dilemmas with conflicting values."""
        scenario = "Should a doctor break patient confidentiality to prevent harm?"
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        # Dilemma should still get full analysis
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_multiple_stakeholder_scenario(self, ethical_method, session):
        """Test scenario with many stakeholders."""
        scenario = (
            "Should the government mandate vaccines for employees, customers, and the public?"
        )
        await ethical_method.execute(session, scenario)

        stakeholder_thoughts = [
            t
            for t in session.graph.nodes.values()
            if t.metadata.get("phase") == "stakeholder_analysis"
        ]
        stakeholders = stakeholder_thoughts[0].metadata.get("stakeholders", [])
        # Should identify multiple stakeholder groups
        assert len(stakeholders) >= 3

    @pytest.mark.asyncio
    async def test_empty_scenario(self, ethical_method, session):
        """Test handling of empty scenario."""
        scenario = ""
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        # Should still complete analysis even with empty input
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_very_long_scenario(self, ethical_method, session):
        """Test handling of very long scenario text."""
        scenario = " ".join(
            [
                "Should we implement a policy that balances economic growth",
                "with environmental protection, considering the needs of current",
                "and future generations, while respecting individual freedoms",
                "and promoting social justice across all demographic groups?",
            ]
        )
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_scenario_with_technical_question(self, ethical_method, session):
        """Test ethical method applied to question with technical elements."""
        scenario = "Should AI systems be allowed to make life-or-death decisions?"
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        # Even technical questions get full ethical treatment
        assert session.thought_count == 8

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self, ethical_method, session):
        """Test session metrics are properly updated during execution."""
        scenario = "Is it ethical to use animals in scientific research?"
        await ethical_method.execute(session, scenario)

        # Verify metrics are updated
        assert session.metrics.total_thoughts == 8
        assert session.metrics.max_depth_reached >= 0
        assert session.metrics.average_confidence > 0.0
        assert session.metrics.methods_used[MethodIdentifier.ETHICAL_REASONING] == 8

    @pytest.mark.asyncio
    async def test_thought_graph_structure(self, ethical_method, session):
        """Test thought graph has proper structure."""
        scenario = "Should we allow assisted suicide?"
        await ethical_method.execute(session, scenario)

        # Verify graph structure
        assert session.graph.node_count == 8
        assert session.graph.edge_count > 0
        # Root should be the initial understanding thought
        assert session.graph.root_id is not None
        root_node = session.graph.get_node(session.graph.root_id)
        assert root_node is not None
        assert root_node.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_scenario_with_question_mark(self, ethical_method, session):
        """Test scenario phrased as a question."""
        scenario = "Is it ethical to prioritize profit over people?"
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        assert result.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_scenario_as_statement(self, ethical_method, session):
        """Test scenario phrased as a statement."""
        scenario = "The company wants to reduce employee benefits to increase profits."
        result = await ethical_method.execute(session, scenario)

        assert result is not None
        assert result.type == ThoughtType.CONCLUSION
