"""Unit tests for MathematicalReasoning method."""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.mathematical import (
    MATHEMATICAL_REASONING_METADATA,
    MathematicalReasoning,
)
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode


@pytest.fixture
def math_method():
    """Create a MathematicalReasoning method instance."""
    return MathematicalReasoning()


@pytest.fixture
def session():
    """Create a fresh session for testing."""
    return Session().start()


class TestMathematicalReasoningMetadata:
    """Tests for MATHEMATICAL_REASONING_METADATA."""

    def test_metadata_identifier(self):
        """Test metadata has correct identifier."""
        assert MATHEMATICAL_REASONING_METADATA.identifier == MethodIdentifier.MATHEMATICAL_REASONING

    def test_metadata_name(self):
        """Test metadata has correct name."""
        assert MATHEMATICAL_REASONING_METADATA.name == "Mathematical Reasoning"

    def test_metadata_category(self):
        """Test metadata is in SPECIALIZED category."""
        assert MATHEMATICAL_REASONING_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_tags(self):
        """Test metadata has expected tags."""
        expected_tags = {
            "mathematical",
            "formal",
            "proof",
            "logic",
            "verification",
            "theorem",
            "rigorous",
            "symbolic",
        }
        assert MATHEMATICAL_REASONING_METADATA.tags == frozenset(expected_tags)

    def test_metadata_complexity(self):
        """Test metadata complexity level."""
        assert MATHEMATICAL_REASONING_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test metadata indicates no branching support."""
        assert MATHEMATICAL_REASONING_METADATA.supports_branching is False

    def test_metadata_supports_revision(self):
        """Test metadata indicates revision support."""
        assert MATHEMATICAL_REASONING_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test metadata indicates context is not required."""
        assert MATHEMATICAL_REASONING_METADATA.requires_context is False

    def test_metadata_thought_bounds(self):
        """Test metadata thought count bounds."""
        assert MATHEMATICAL_REASONING_METADATA.min_thoughts == 3
        assert MATHEMATICAL_REASONING_METADATA.max_thoughts == 0  # Unlimited

    def test_metadata_avg_tokens(self):
        """Test metadata average tokens per thought."""
        assert MATHEMATICAL_REASONING_METADATA.avg_tokens_per_thought == 400

    def test_metadata_best_for(self):
        """Test metadata best use cases."""
        assert len(MATHEMATICAL_REASONING_METADATA.best_for) > 0
        assert "mathematical proofs" in MATHEMATICAL_REASONING_METADATA.best_for
        assert "formal logic problems" in MATHEMATICAL_REASONING_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test metadata not recommended use cases."""
        assert len(MATHEMATICAL_REASONING_METADATA.not_recommended_for) > 0
        assert "creative brainstorming" in MATHEMATICAL_REASONING_METADATA.not_recommended_for


class TestMathematicalReasoningInitialization:
    """Tests for initialization and health check."""

    def test_initial_state(self, math_method):
        """Test method is not initialized on creation."""
        assert math_method._initialized is False
        assert math_method._step_counter == 0
        assert math_method._current_phase == MathematicalReasoning.PHASE_SETUP

    def test_identifier_property(self, math_method):
        """Test identifier property returns correct value."""
        assert math_method.identifier == str(MethodIdentifier.MATHEMATICAL_REASONING)

    def test_name_property(self, math_method):
        """Test name property returns correct value."""
        assert math_method.name == "Mathematical Reasoning"

    def test_description_property(self, math_method):
        """Test description property returns correct value."""
        assert "formal mathematical reasoning" in math_method.description.lower()

    def test_category_property(self, math_method):
        """Test category property returns correct value."""
        assert math_method.category == str(MethodCategory.SPECIALIZED)

    @pytest.mark.asyncio
    async def test_initialize(self, math_method):
        """Test initialize method sets initialized flag and resets state."""
        await math_method.initialize()
        assert math_method._initialized is True
        assert math_method._step_counter == 0
        assert math_method._current_phase == MathematicalReasoning.PHASE_SETUP
        assert math_method._theorems_used == []
        assert math_method._definitions_used == []

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self, math_method):
        """Test initialize resets state even after prior use."""
        # Modify state
        math_method._step_counter = 5
        math_method._current_phase = MathematicalReasoning.PHASE_CONCLUSION
        math_method._theorems_used = ["theorem1"]
        math_method._definitions_used = ["def1"]

        # Re-initialize
        await math_method.initialize()

        assert math_method._initialized is True
        assert math_method._step_counter == 0
        assert math_method._current_phase == MathematicalReasoning.PHASE_SETUP
        assert math_method._theorems_used == []
        assert math_method._definitions_used == []

    @pytest.mark.asyncio
    async def test_health_check_before_init(self, math_method):
        """Test health check returns False before initialization."""
        healthy = await math_method.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_health_check_after_init(self, math_method):
        """Test health check returns True after initialization."""
        await math_method.initialize()
        healthy = await math_method.health_check()
        assert healthy is True


class TestMathematicalReasoningExecution:
    """Tests for basic execution flow."""

    @pytest.mark.asyncio
    async def test_execute_requires_initialization(self, math_method, session):
        """Test execute raises error if not initialized."""
        assert math_method._initialized is False
        problem = "Prove that the square root of 2 is irrational"

        with pytest.raises(RuntimeError, match="must be initialized"):
            await math_method.execute(session, problem)

    @pytest.mark.asyncio
    async def test_execute_returns_thought_node(self, math_method, session):
        """Test execute returns a ThoughtNode."""
        await math_method.initialize()
        problem = "Solve for x: 2x + 5 = 15"
        result = await math_method.execute(session, problem)

        assert isinstance(result, ThoughtNode)

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self, math_method, session):
        """Test execute creates INITIAL type thought."""
        await math_method.initialize()
        problem = "Prove that n^2 - n is always even for any integer n"
        result = await math_method.execute(session, problem)

        assert result.type == ThoughtType.INITIAL
        assert result.step_number == 1
        assert result.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_correct_method_id(self, math_method, session):
        """Test execute sets correct method identifier."""
        await math_method.initialize()
        problem = "What is the derivative of x^2?"
        result = await math_method.execute(session, problem)

        assert result.method_id == MethodIdentifier.MATHEMATICAL_REASONING

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self, math_method, session):
        """Test execute adds thought to session."""
        await math_method.initialize()
        problem = "Factor x^2 - 9"

        initial_count = session.thought_count
        await math_method.execute(session, problem)

        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_sets_session_method(self, math_method, session):
        """Test execute updates session current_method."""
        await math_method.initialize()
        problem = "Integrate x^3 dx"
        await math_method.execute(session, problem)

        assert session.current_method == MethodIdentifier.MATHEMATICAL_REASONING

    @pytest.mark.asyncio
    async def test_execute_with_context(self, math_method, session):
        """Test execute accepts and processes context parameter."""
        await math_method.initialize()
        problem = "Apply the Pythagorean theorem"
        context: dict[str, Any] = {"given": "a=3, b=4", "find": "c"}
        result = await math_method.execute(session, problem, context=context)

        assert result is not None
        assert result.metadata.get("context") == context

    @pytest.mark.asyncio
    async def test_execute_without_context(self, math_method, session):
        """Test execute works without context parameter."""
        await math_method.initialize()
        problem = "Prove the triangle inequality"
        result = await math_method.execute(session, problem, context=None)

        assert result is not None
        assert result.metadata.get("context") == {}

    @pytest.mark.asyncio
    async def test_execute_has_high_confidence(self, math_method, session):
        """Test initial thought has high confidence."""
        await math_method.initialize()
        problem = "Calculate 2 + 2"
        result = await math_method.execute(session, problem)

        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_execute_content_structure(self, math_method, session):
        """Test execute creates properly structured content."""
        await math_method.initialize()
        problem = "Prove that 1 + 1 = 2"
        result = await math_method.execute(session, problem)

        # Check content includes key sections
        assert "Problem Statement:" in result.content
        assert problem in result.content
        assert "Strategy:" in result.content
        assert "Step 1:" in result.content


class TestProofStructure:
    """Tests for proof structure and thought types."""

    @pytest.mark.asyncio
    async def test_initial_phase_is_setup(self, math_method, session):
        """Test initial execution starts in setup phase."""
        await math_method.initialize()
        problem = "Prove sum of two even numbers is even"
        result = await math_method.execute(session, problem)

        assert result.metadata.get("phase") == MathematicalReasoning.PHASE_SETUP

    @pytest.mark.asyncio
    async def test_phase_progression_to_given(self, math_method, session):
        """Test phase progresses to GIVEN after setup."""
        await math_method.initialize()
        problem = "Prove that x^2 >= 0 for all real x"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second.metadata.get("phase") == MathematicalReasoning.PHASE_GIVEN

    @pytest.mark.asyncio
    async def test_phase_progression_to_theorem_application(self, math_method, session):
        """Test phase progresses to theorem application."""
        await math_method.initialize()
        problem = "Use Fermat's Last Theorem"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)
        third = await math_method.continue_reasoning(session, second)

        assert third.metadata.get("phase") == MathematicalReasoning.PHASE_THEOREM_APPLICATION

    @pytest.mark.asyncio
    async def test_phase_progression_to_derivation(self, math_method, session):
        """Test phase progresses to derivation."""
        await math_method.initialize()
        problem = "Derive formula for quadratic roots"
        first = await math_method.execute(session, problem)

        # Progress through phases
        current = first
        for _ in range(3):
            current = await math_method.continue_reasoning(session, current)

        assert current.metadata.get("phase") == MathematicalReasoning.PHASE_DERIVATION

    @pytest.mark.asyncio
    async def test_phase_progression_to_verification(self, math_method, session):
        """Test phase progresses to verification."""
        await math_method.initialize()
        problem = "Verify the proof"
        first = await math_method.execute(session, problem)

        # Progress to verification phase (step 7)
        current = first
        for _ in range(6):
            current = await math_method.continue_reasoning(session, current)

        assert current.metadata.get("phase") == MathematicalReasoning.PHASE_VERIFICATION

    @pytest.mark.asyncio
    async def test_verification_thought_type(self, math_method, session):
        """Test verification phase creates VERIFICATION thought type."""
        await math_method.initialize()
        problem = "Check the solution"
        first = await math_method.execute(session, problem)

        # Progress to verification phase
        current = first
        for _ in range(6):
            current = await math_method.continue_reasoning(session, current)

        assert current.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_conclusion_thought_type(self, math_method, session):
        """Test conclusion phase creates CONCLUSION thought type."""
        await math_method.initialize()
        problem = "Prove the theorem"
        first = await math_method.execute(session, problem)

        # Progress to conclusion phase (step 8+)
        current = first
        for _ in range(7):
            current = await math_method.continue_reasoning(session, current)

        assert current.type == ThoughtType.CONCLUSION

    @pytest.mark.asyncio
    async def test_continuation_thought_type(self, math_method, session):
        """Test derivation phase creates CONTINUATION thought type."""
        await math_method.initialize()
        problem = "Derive the result"
        first = await math_method.execute(session, problem)

        # Step 2-6 should be continuations
        current = first
        for i in range(5):
            current = await math_method.continue_reasoning(session, current)
            if i < 4:  # Steps 2-5
                assert current.type == ThoughtType.CONTINUATION


class TestConfigurationOptions:
    """Tests for configuration options like rigor_level and allow_approximation."""

    @pytest.mark.asyncio
    async def test_execute_accepts_rigor_level_in_context(self, math_method, session):
        """Test rigor_level can be specified in context."""
        await math_method.initialize()
        problem = "Prove using high rigor"
        context: dict[str, Any] = {"rigor_level": "high"}
        result = await math_method.execute(session, problem, context=context)

        assert result.metadata.get("context", {}).get("rigor_level") == "high"

    @pytest.mark.asyncio
    async def test_execute_accepts_allow_approximation_in_context(self, math_method, session):
        """Test allow_approximation can be specified in context."""
        await math_method.initialize()
        problem = "Approximate pi"
        context: dict[str, Any] = {"allow_approximation": True}
        result = await math_method.execute(session, problem, context=context)

        assert result.metadata.get("context", {}).get("allow_approximation") is True

    @pytest.mark.asyncio
    async def test_execute_with_multiple_config_options(self, math_method, session):
        """Test multiple configuration options together."""
        await math_method.initialize()
        problem = "Solve approximately with medium rigor"
        context: dict[str, Any] = {
            "rigor_level": "medium",
            "allow_approximation": True,
            "precision": 0.001,
        }
        result = await math_method.execute(session, problem, context=context)

        ctx = result.metadata.get("context", {})
        assert ctx.get("rigor_level") == "medium"
        assert ctx.get("allow_approximation") is True
        assert ctx.get("precision") == 0.001


class TestContinueReasoning:
    """Tests for continue_reasoning method."""

    @pytest.mark.asyncio
    async def test_continue_reasoning_requires_initialization(self, math_method, session):
        """Test continue_reasoning raises error if not initialized."""
        await math_method.initialize()
        problem = "Prove something"
        first = await math_method.execute(session, problem)

        # Reset initialization
        math_method._initialized = False

        with pytest.raises(RuntimeError, match="must be initialized"):
            await math_method.continue_reasoning(session, first)

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_step(self, math_method, session):
        """Test continue_reasoning increments step counter."""
        await math_method.initialize()
        problem = "Prove by induction"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second.step_number == first.step_number + 1
        assert second.step_number == 2

    @pytest.mark.asyncio
    async def test_continue_reasoning_increments_depth(self, math_method, session):
        """Test continue_reasoning increments depth."""
        await math_method.initialize()
        problem = "Continue the proof"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second.depth == first.depth + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_sets_parent_id(self, math_method, session):
        """Test continue_reasoning sets parent_id correctly."""
        await math_method.initialize()
        problem = "Multi-step proof"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second.parent_id == first.id

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_guidance(self, math_method, session):
        """Test continue_reasoning with user guidance."""
        await math_method.initialize()
        problem = "Solve the equation"
        first = await math_method.execute(session, problem)

        guidance = "Apply the quadratic formula"
        second = await math_method.continue_reasoning(session, first, guidance=guidance)

        assert guidance in second.content
        assert second.metadata.get("guidance") == guidance

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(self, math_method, session):
        """Test continue_reasoning without guidance proceeds automatically."""
        await math_method.initialize()
        problem = "Prove step by step"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second is not None
        assert second.metadata.get("guidance") == ""

    @pytest.mark.asyncio
    async def test_continue_reasoning_guidance_affects_phase(self, math_method, session):
        """Test guidance with keywords affects phase selection."""
        await math_method.initialize()
        problem = "Prove with theorems"
        first = await math_method.execute(session, problem)

        # Guidance with "apply theorem" should trigger theorem application phase
        guidance = "apply the Pythagorean theorem"
        second = await math_method.continue_reasoning(session, first, guidance=guidance)

        assert second.metadata.get("phase") == MathematicalReasoning.PHASE_THEOREM_APPLICATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_verify_guidance(self, math_method, session):
        """Test guidance with 'verify' triggers verification phase."""
        await math_method.initialize()
        problem = "Solve and verify"
        first = await math_method.execute(session, problem)

        guidance = "verify the solution"
        second = await math_method.continue_reasoning(session, first, guidance=guidance)

        assert second.metadata.get("phase") == MathematicalReasoning.PHASE_VERIFICATION

    @pytest.mark.asyncio
    async def test_continue_reasoning_conclude_guidance(self, math_method, session):
        """Test guidance with 'conclude' triggers conclusion phase."""
        await math_method.initialize()
        problem = "Finish the proof"
        first = await math_method.execute(session, problem)

        guidance = "conclude the proof"
        second = await math_method.continue_reasoning(session, first, guidance=guidance)

        assert second.metadata.get("phase") == MathematicalReasoning.PHASE_CONCLUSION

    @pytest.mark.asyncio
    async def test_continue_reasoning_increases_confidence(self, math_method, session):
        """Test continue_reasoning slightly increases confidence."""
        await math_method.initialize()
        problem = "Build confidence"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert second.confidence > first.confidence
        assert second.confidence <= 0.95  # Capped at 0.95

    @pytest.mark.asyncio
    async def test_continue_reasoning_confidence_capped(self, math_method, session):
        """Test confidence is capped at 0.95."""
        await math_method.initialize()
        problem = "High confidence proof"
        first = await math_method.execute(session, problem)

        # Continue many times
        current = first
        for _ in range(10):
            current = await math_method.continue_reasoning(session, current)

        assert current.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_continue_reasoning_adds_to_session(self, math_method, session):
        """Test continue_reasoning adds thought to session."""
        await math_method.initialize()
        problem = "Multi-step solution"
        first = await math_method.execute(session, problem)
        initial_count = session.thought_count

        await math_method.continue_reasoning(session, first)

        assert session.thought_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_context(self, math_method, session):
        """Test continue_reasoning accepts context parameter."""
        await math_method.initialize()
        problem = "Continued reasoning with context"
        first = await math_method.execute(session, problem)

        context: dict[str, Any] = {"new_constraint": "x > 0"}
        second = await math_method.continue_reasoning(session, first, context=context)

        assert second.metadata.get("context") == context


class TestSymbolicManipulation:
    """Tests for mathematical expression and symbolic manipulation handling."""

    @pytest.mark.asyncio
    async def test_simple_arithmetic_expression(self, math_method, session):
        """Test handling of simple arithmetic."""
        await math_method.initialize()
        problem = "Calculate 5 + 3 * 2"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "5 + 3 * 2" in result.content or "5 + 3 \\* 2" in result.content

    @pytest.mark.asyncio
    async def test_algebraic_expression(self, math_method, session):
        """Test handling of algebraic expressions."""
        await math_method.initialize()
        problem = "Simplify (x + 2)(x - 2)"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "(x + 2)(x - 2)" in result.content or problem in result.content

    @pytest.mark.asyncio
    async def test_equation_solving(self, math_method, session):
        """Test equation solving problems."""
        await math_method.initialize()
        problem = "Solve 3x - 7 = 14"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "3x - 7 = 14" in result.content or problem in result.content

    @pytest.mark.asyncio
    async def test_calculus_problem(self, math_method, session):
        """Test calculus-related problems."""
        await math_method.initialize()
        problem = "Find the derivative of f(x) = x^3 + 2x"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "derivative" in result.content.lower()

    @pytest.mark.asyncio
    async def test_geometric_proof(self, math_method, session):
        """Test geometric proof problems."""
        await math_method.initialize()
        problem = "Prove that the angles in a triangle sum to 180 degrees"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "triangle" in result.content.lower()


class TestProofValidation:
    """Tests for verification of proof steps."""

    @pytest.mark.asyncio
    async def test_verification_step_content(self, math_method, session):
        """Test verification step has appropriate content."""
        await math_method.initialize()
        problem = "Verify proof correctness"
        first = await math_method.execute(session, problem)

        # Progress to verification phase
        current = first
        for _ in range(6):
            current = await math_method.continue_reasoning(session, current)

        assert "Verification" in current.content
        assert current.type == ThoughtType.VERIFICATION

    @pytest.mark.asyncio
    async def test_justification_in_proof_steps(self, math_method, session):
        """Test each proof step includes justification."""
        await math_method.initialize()
        problem = "Prove with justification"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert "Justification:" in second.content

    @pytest.mark.asyncio
    async def test_logical_consistency_check(self, math_method, session):
        """Test verification mentions logical consistency."""
        await math_method.initialize()
        problem = "Check logical consistency"
        first = await math_method.execute(session, problem)

        # Progress to verification
        current = first
        for _ in range(6):
            current = await math_method.continue_reasoning(session, current)

        assert "valid" in current.content.lower() or "consistency" in current.content.lower()


class TestMultipleApproaches:
    """Tests for different proof strategies."""

    @pytest.mark.asyncio
    async def test_direct_proof_strategy(self, math_method, session):
        """Test direct proof problems."""
        await math_method.initialize()
        problem = "Prove directly that if n is even, then n^2 is even"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "Strategy:" in result.content

    @pytest.mark.asyncio
    async def test_contradiction_proof_strategy(self, math_method, session):
        """Test proof by contradiction."""
        await math_method.initialize()
        problem = "Prove by contradiction that sqrt(2) is irrational"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "contradiction" in result.content.lower() or "Strategy:" in result.content

    @pytest.mark.asyncio
    async def test_induction_proof_strategy(self, math_method, session):
        """Test proof by induction."""
        await math_method.initialize()
        problem = "Prove by induction that 1 + 2 + ... + n = n(n+1)/2"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "induction" in result.content.lower() or "Strategy:" in result.content

    @pytest.mark.asyncio
    async def test_constructive_proof_strategy(self, math_method, session):
        """Test constructive proofs."""
        await math_method.initialize()
        problem = "Construct a solution to x^2 = 4"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert "Strategy:" in result.content


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_simple_arithmetic(self, math_method, session):
        """Test handling of very simple arithmetic."""
        await math_method.initialize()
        problem = "What is 2 + 2?"
        result = await math_method.execute(session, problem)

        assert result is not None
        assert result.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_complex_proof(self, math_method, session):
        """Test handling of complex mathematical proofs."""
        await math_method.initialize()
        problem = (
            "Prove that there are infinitely many prime numbers using "
            "Euclid's classical proof by contradiction"
        )
        result = await math_method.execute(session, problem)

        assert result is not None
        assert len(result.content) > 100  # Complex problem should have substantial content

    @pytest.mark.asyncio
    async def test_approximation_problem(self, math_method, session):
        """Test problems involving approximations."""
        await math_method.initialize()
        problem = "Approximate e to 3 decimal places"
        context: dict[str, Any] = {"allow_approximation": True}
        result = await math_method.execute(session, problem, context=context)

        assert result is not None
        assert result.metadata.get("context", {}).get("allow_approximation") is True

    @pytest.mark.asyncio
    async def test_theorem_application(self, math_method, session):
        """Test applying named theorems."""
        await math_method.initialize()
        problem = "Apply the Fundamental Theorem of Calculus"
        first = await math_method.execute(session, problem)

        guidance = "apply the theorem to integrate"
        second = await math_method.continue_reasoning(session, first, guidance=guidance)

        assert second.metadata.get("phase") == MathematicalReasoning.PHASE_THEOREM_APPLICATION

    @pytest.mark.asyncio
    async def test_empty_problem(self, math_method, session):
        """Test handling of empty problem statement."""
        await math_method.initialize()
        problem = ""
        result = await math_method.execute(session, problem)

        assert result is not None
        # Should still create structured response

    @pytest.mark.asyncio
    async def test_very_long_problem(self, math_method, session):
        """Test handling of very long problem statements."""
        await math_method.initialize()
        problem = " ".join(
            [
                "Given a complex mathematical system with multiple constraints,",
                "variables x, y, z subject to the conditions x^2 + y^2 = z^2,",
                "x > 0, y > 0, z > 0, and the additional requirement that x + y + z = 12,",
                "prove that there exists at least one solution satisfying all constraints.",
            ]
        )
        result = await math_method.execute(session, problem)

        assert result is not None
        assert problem in result.content

    @pytest.mark.asyncio
    async def test_session_metrics_updated(self, math_method, session):
        """Test session metrics are properly updated during execution."""
        await math_method.initialize()
        problem = "Test metrics update"
        await math_method.execute(session, problem)

        assert session.metrics.total_thoughts >= 1
        assert session.metrics.methods_used[MethodIdentifier.MATHEMATICAL_REASONING] >= 1

    @pytest.mark.asyncio
    async def test_thought_graph_structure(self, math_method, session):
        """Test thought graph has proper structure."""
        await math_method.initialize()
        problem = "Build a proof graph"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)
        await math_method.continue_reasoning(session, second)

        # Verify graph structure
        assert session.graph.node_count == 3
        assert session.graph.root_id is not None
        root_node = session.graph.get_node(session.graph.root_id)
        assert root_node is not None
        assert root_node.id == first.id

    @pytest.mark.asyncio
    async def test_metadata_tracks_phase_progression(self, math_method, session):
        """Test metadata properly tracks phase changes."""
        await math_method.initialize()
        problem = "Track phase changes"
        first = await math_method.execute(session, problem)

        phases = [first.metadata.get("phase")]
        current = first
        for _ in range(7):
            current = await math_method.continue_reasoning(session, current)
            phases.append(current.metadata.get("phase"))

        # Should have progressed through multiple phases
        unique_phases = set(phases)
        assert len(unique_phases) > 1
        assert MathematicalReasoning.PHASE_SETUP in unique_phases

    @pytest.mark.asyncio
    async def test_metadata_tracks_theorems_used(self, math_method, session):
        """Test metadata includes theorems_used tracking."""
        await math_method.initialize()
        problem = "Use theorems"
        result = await math_method.execute(session, problem)

        assert "theorems_used" in result.metadata
        assert isinstance(result.metadata["theorems_used"], list)

    @pytest.mark.asyncio
    async def test_metadata_tracks_definitions_used(self, math_method, session):
        """Test metadata includes definitions_used tracking."""
        await math_method.initialize()
        problem = "Define and use terms"
        result = await math_method.execute(session, problem)

        assert "definitions_used" in result.metadata
        assert isinstance(result.metadata["definitions_used"], list)

    @pytest.mark.asyncio
    async def test_multiple_continue_calls_chain_properly(self, math_method, session):
        """Test multiple continue_reasoning calls create proper chain."""
        await math_method.initialize()
        problem = "Multi-step chain"
        thoughts = [await math_method.execute(session, problem)]

        for _ in range(5):
            thoughts.append(await math_method.continue_reasoning(session, thoughts[-1]))

        # Verify chain
        for i in range(1, len(thoughts)):
            assert thoughts[i].parent_id == thoughts[i - 1].id
            assert thoughts[i].step_number == i + 1
            assert thoughts[i].depth == i

    @pytest.mark.asyncio
    async def test_reasoning_type_metadata(self, math_method, session):
        """Test all thoughts have reasoning_type metadata."""
        await math_method.initialize()
        problem = "Check reasoning type"
        first = await math_method.execute(session, problem)
        second = await math_method.continue_reasoning(session, first)

        assert first.metadata.get("reasoning_type") == "mathematical"
        assert second.metadata.get("reasoning_type") == "mathematical"

    @pytest.mark.asyncio
    async def test_step_intro_varies_by_phase(self, math_method, session):
        """Test step introductions vary based on phase."""
        await math_method.initialize()
        problem = "Test phase intros"
        first = await math_method.execute(session, problem)

        # Get a few different phase steps
        current = first
        contents = [first.content]
        for _ in range(4):
            current = await math_method.continue_reasoning(session, current)
            contents.append(current.content)

        # Different phases should have different step structures
        # At minimum, step numbers should increment
        assert "Step 1:" in contents[0]
        assert "Step 2:" in contents[1]
