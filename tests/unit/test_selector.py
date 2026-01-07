"""Tests for the selector module."""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.selector import (
    SELECTION_RULES,
    MethodRecommendation,
    MethodSelector,
    SelectionConstraint,
    SelectionHint,
    SelectionRule,
    detect_problem_patterns,
    estimate_complexity,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockMethod:
    """Mock implementation of ReasoningMethod protocol for testing."""

    def __init__(
        self,
        identifier: str = "chain_of_thought",
        name: str = "Chain of Thought",
        description: str = "Step-by-step reasoning",
        category: str = "core",
    ) -> None:
        self._identifier = identifier
        self._name = name
        self._description = description
        self._category = category
        self._initialized = False
        self._healthy = True

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    async def initialize(self) -> None:
        self._initialized = True

    async def execute(
        self,
        session: Any,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        return {"thought": f"Thinking about: {input_text}"}

    async def continue_reasoning(
        self,
        session: Any,
        previous_thought: Any,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        return {"thought": "Continuing..."}

    async def health_check(self) -> bool:
        return self._healthy


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Create a registry with mock methods for testing."""
    registry = MethodRegistry()

    # Register chain_of_thought
    cot_method = MockMethod(
        identifier="chain_of_thought",
        name="Chain of Thought",
        description="Step-by-step reasoning",
        category="core",
    )
    cot_metadata = MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Chain of Thought",
        description="Step-by-step reasoning",
        category=MethodCategory.CORE,
        tags=frozenset({"analytical", "multi_step"}),
        complexity=3,
        supports_branching=False,
        supports_revision=False,
        best_for=("step-by-step problems", "logical reasoning"),
        not_recommended_for=("creative tasks",),
    )
    registry.register(cot_method, cot_metadata)

    # Register tree_of_thoughts
    tot_method = MockMethod(
        identifier="tree_of_thoughts",
        name="Tree of Thoughts",
        description="Branching exploration",
        category="core",
    )
    tot_metadata = MethodMetadata(
        identifier=MethodIdentifier.TREE_OF_THOUGHTS,
        name="Tree of Thoughts",
        description="Branching exploration",
        category=MethodCategory.CORE,
        tags=frozenset({"creative", "decision"}),
        complexity=6,
        supports_branching=True,
        supports_revision=True,
        best_for=("creative problems", "decision making"),
        not_recommended_for=("simple calculations",),
    )
    registry.register(tot_method, tot_metadata)

    # Register mathematical_reasoning
    math_method = MockMethod(
        identifier="mathematical_reasoning",
        name="Mathematical Reasoning",
        description="Formal math reasoning",
        category="specialized",
    )
    math_metadata = MethodMetadata(
        identifier=MethodIdentifier.MATHEMATICAL_REASONING,
        name="Mathematical Reasoning",
        description="Formal mathematical reasoning",
        category=MethodCategory.SPECIALIZED,
        tags=frozenset({"mathematical"}),
        complexity=7,
        supports_branching=False,
        supports_revision=True,
        best_for=("calculations", "proofs"),
        not_recommended_for=("creative tasks",),
    )
    registry.register(math_method, math_metadata)

    # Register code_reasoning
    code_method = MockMethod(
        identifier="code_reasoning",
        name="Code Reasoning",
        description="Code analysis and debugging",
        category="high_value",
    )
    code_metadata = MethodMetadata(
        identifier=MethodIdentifier.CODE_REASONING,
        name="Code Reasoning",
        description="Code analysis and debugging",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"code"}),
        complexity=5,
        supports_branching=True,
        supports_revision=True,
        best_for=("debugging", "code review"),
        not_recommended_for=("non-code tasks",),
    )
    registry.register(code_method, code_metadata)

    # Register ethical_reasoning
    ethics_method = MockMethod(
        identifier="ethical_reasoning",
        name="Ethical Reasoning",
        description="Multi-framework ethical analysis",
        category="high_value",
    )
    ethics_metadata = MethodMetadata(
        identifier=MethodIdentifier.ETHICAL_REASONING,
        name="Ethical Reasoning",
        description="Multi-framework ethical analysis",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"ethical"}),
        complexity=6,
        supports_branching=False,
        supports_revision=True,
        best_for=("ethical dilemmas", "moral reasoning"),
        not_recommended_for=("technical problems",),
    )
    registry.register(ethics_method, ethics_metadata)

    return registry


# =============================================================================
# Tests for detect_problem_patterns
# =============================================================================


class TestDetectProblemPatterns:
    """Tests for detect_problem_patterns function."""

    def test_detect_mathematical_pattern(self):
        """Test detection of mathematical patterns."""
        text = "Calculate the sum of 2 + 2"
        patterns = detect_problem_patterns(text)
        assert patterns["mathematical"] is True

    def test_detect_code_pattern(self):
        """Test detection of code patterns."""
        text = "Debug this function that throws an error"
        patterns = detect_problem_patterns(text)
        assert patterns["code"] is True

    def test_detect_ethical_pattern(self):
        """Test detection of ethical patterns."""
        text = "Is it ethical to use this data?"
        patterns = detect_problem_patterns(text)
        assert patterns["ethical"] is True

    def test_detect_creative_pattern(self):
        """Test detection of creative patterns."""
        text = "Design a novel solution to this problem"
        patterns = detect_problem_patterns(text)
        assert patterns["creative"] is True

    def test_detect_analytical_pattern(self):
        """Test detection of analytical patterns."""
        text = "Analyze and compare these two approaches"
        patterns = detect_problem_patterns(text)
        assert patterns["analytical"] is True

    def test_detect_causal_pattern(self):
        """Test detection of causal patterns."""
        text = "Why does this cause that effect?"
        patterns = detect_problem_patterns(text)
        assert patterns["causal"] is True

    def test_detect_decision_pattern(self):
        """Test detection of decision patterns."""
        text = "Help me decide between these options"
        patterns = detect_problem_patterns(text)
        assert patterns["decision"] is True

    def test_detect_multi_step_pattern(self):
        """Test detection of multi-step patterns."""
        text = "First do this, then do that, next continue"
        patterns = detect_problem_patterns(text)
        assert patterns["multi_step"] is True

    def test_no_patterns_detected(self):
        """Test when no patterns are detected."""
        text = "Hello world"
        patterns = detect_problem_patterns(text)
        assert all(not v for v in patterns.values())

    def test_multiple_patterns_detected(self):
        """Test detection of multiple patterns."""
        text = "Calculate and analyze this mathematical equation step by step"
        patterns = detect_problem_patterns(text)
        assert patterns["mathematical"] is True
        assert patterns["analytical"] is True
        assert patterns["multi_step"] is True

    def test_case_insensitive(self):
        """Test that pattern detection is case insensitive."""
        text = "CALCULATE the SUM"
        patterns = detect_problem_patterns(text)
        assert patterns["mathematical"] is True

    def test_all_patterns_returned(self):
        """Test that all pattern types are returned."""
        patterns = detect_problem_patterns("test")
        expected_patterns = {
            "mathematical",
            "code",
            "ethical",
            "creative",
            "analytical",
            "causal",
            "decision",
            "multi_step",
        }
        assert set(patterns.keys()) == expected_patterns


# =============================================================================
# Tests for estimate_complexity
# =============================================================================


class TestEstimateComplexity:
    """Tests for estimate_complexity function."""

    def test_base_complexity(self):
        """Test base complexity for simple text."""
        patterns = detect_problem_patterns("simple")
        complexity = estimate_complexity("simple", patterns)
        assert 1 <= complexity <= 10

    def test_complexity_increases_with_patterns(self):
        """Test complexity increases with more patterns."""
        text_simple = "hello world"
        patterns_simple = detect_problem_patterns(text_simple)
        complexity_simple = estimate_complexity(text_simple, patterns_simple)

        text_complex = "calculate and analyze the code step by step"
        patterns_complex = detect_problem_patterns(text_complex)
        complexity_complex = estimate_complexity(text_complex, patterns_complex)

        assert complexity_complex > complexity_simple

    def test_complexity_increases_with_length(self):
        """Test complexity increases with longer text."""
        short_text = "hello"
        patterns = detect_problem_patterns(short_text)
        short_complexity = estimate_complexity(short_text, patterns)

        long_text = " ".join(["word"] * 150)  # > 100 words
        patterns = detect_problem_patterns(long_text)
        long_complexity = estimate_complexity(long_text, patterns)

        assert long_complexity >= short_complexity

    def test_complexity_max_bound(self):
        """Test complexity doesn't exceed 10."""
        # Create text with many patterns and long length
        text = " ".join(
            [
                "calculate analyze code ethical creative cause decide step"
            ]
            * 50
        )
        patterns = detect_problem_patterns(text)
        complexity = estimate_complexity(text, patterns)
        assert complexity <= 10

    def test_complexity_min_bound(self):
        """Test complexity is at least 1."""
        text = "a"
        patterns = {"p": False for p in range(10)}
        complexity = estimate_complexity(text, patterns)
        assert complexity >= 1


# =============================================================================
# Tests for SelectionHint
# =============================================================================


class TestSelectionHint:
    """Tests for SelectionHint dataclass."""

    def test_create_with_defaults(self):
        """Test creating SelectionHint with default values."""
        hint = SelectionHint(problem_type="general")
        assert hint.problem_type == "general"
        assert hint.complexity_estimate == 5
        assert hint.requires_branching is False
        assert hint.requires_iteration is False
        assert hint.requires_external_tools is False
        assert hint.domain_tags == frozenset()
        assert hint.keywords == frozenset()
        assert hint.suggested_min_depth == 1
        assert hint.confidence == 0.5

    def test_create_with_all_fields(self):
        """Test creating SelectionHint with all fields."""
        hint = SelectionHint(
            problem_type="mathematical",
            complexity_estimate=8,
            requires_branching=True,
            requires_iteration=True,
            requires_external_tools=True,
            domain_tags=frozenset({"math", "analysis"}),
            keywords=frozenset({"calculate", "prove"}),
            suggested_min_depth=3,
            confidence=0.9,
        )
        assert hint.problem_type == "mathematical"
        assert hint.complexity_estimate == 8
        assert hint.requires_branching is True
        assert hint.domain_tags == frozenset({"math", "analysis"})

    def test_immutability(self):
        """Test that SelectionHint is immutable."""
        hint = SelectionHint(problem_type="test")
        with pytest.raises(AttributeError):
            hint.problem_type = "modified"  # type: ignore[misc]

    def test_complexity_validation_min(self):
        """Test complexity validation minimum bound."""
        with pytest.raises(ValueError, match="complexity_estimate must be 1-10"):
            SelectionHint(problem_type="test", complexity_estimate=0)

    def test_complexity_validation_max(self):
        """Test complexity validation maximum bound."""
        with pytest.raises(ValueError, match="complexity_estimate must be 1-10"):
            SelectionHint(problem_type="test", complexity_estimate=11)

    def test_confidence_validation_min(self):
        """Test confidence validation minimum bound."""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            SelectionHint(problem_type="test", confidence=-0.1)

    def test_confidence_validation_max(self):
        """Test confidence validation maximum bound."""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            SelectionHint(problem_type="test", confidence=1.1)

    def test_suggested_min_depth_validation(self):
        """Test suggested_min_depth validation."""
        with pytest.raises(ValueError, match="suggested_min_depth must be >= 1"):
            SelectionHint(problem_type="test", suggested_min_depth=0)


# =============================================================================
# Tests for SelectionConstraint
# =============================================================================


class TestSelectionConstraint:
    """Tests for SelectionConstraint dataclass."""

    def test_create_with_defaults(self):
        """Test creating SelectionConstraint with default values."""
        constraint = SelectionConstraint()
        assert constraint.allowed_methods is None
        assert constraint.excluded_methods == frozenset()
        assert constraint.allowed_categories is None
        assert constraint.excluded_categories == frozenset()
        assert constraint.max_complexity is None
        assert constraint.require_branching is False
        assert constraint.require_revision is False
        assert constraint.max_tokens_budget is None
        assert constraint.preferred_methods == frozenset()

    def test_create_with_all_fields(self):
        """Test creating SelectionConstraint with all fields."""
        constraint = SelectionConstraint(
            allowed_methods=frozenset({"method_a", "method_b"}),
            excluded_methods=frozenset({"method_c"}),
            allowed_categories=frozenset({"core"}),
            excluded_categories=frozenset({"advanced"}),
            max_complexity=7,
            require_branching=True,
            require_revision=True,
            max_tokens_budget=5000,
            preferred_methods=frozenset({"method_a"}),
        )
        assert constraint.allowed_methods == frozenset({"method_a", "method_b"})
        assert constraint.max_complexity == 7

    def test_immutability(self):
        """Test that SelectionConstraint is immutable."""
        constraint = SelectionConstraint()
        with pytest.raises(AttributeError):
            constraint.max_complexity = 5  # type: ignore[misc]

    def test_max_complexity_validation_min(self):
        """Test max_complexity validation minimum bound."""
        with pytest.raises(ValueError, match="max_complexity must be 1-10"):
            SelectionConstraint(max_complexity=0)

    def test_max_complexity_validation_max(self):
        """Test max_complexity validation maximum bound."""
        with pytest.raises(ValueError, match="max_complexity must be 1-10"):
            SelectionConstraint(max_complexity=11)

    def test_max_tokens_budget_validation(self):
        """Test max_tokens_budget validation."""
        with pytest.raises(ValueError, match="max_tokens_budget must be >= 100"):
            SelectionConstraint(max_tokens_budget=50)

    def test_is_method_allowed_no_restrictions(self):
        """Test is_method_allowed with no restrictions."""
        constraint = SelectionConstraint()
        assert constraint.is_method_allowed("any_method") is True

    def test_is_method_allowed_excluded(self):
        """Test is_method_allowed with excluded methods."""
        constraint = SelectionConstraint(
            excluded_methods=frozenset({"excluded_method"})
        )
        assert constraint.is_method_allowed("excluded_method") is False
        assert constraint.is_method_allowed("other_method") is True

    def test_is_method_allowed_whitelist(self):
        """Test is_method_allowed with allowed methods list."""
        constraint = SelectionConstraint(
            allowed_methods=frozenset({"allowed_method"})
        )
        assert constraint.is_method_allowed("allowed_method") is True
        assert constraint.is_method_allowed("other_method") is False

    def test_is_method_allowed_excluded_takes_precedence(self):
        """Test that excluded_methods takes precedence over allowed_methods."""
        constraint = SelectionConstraint(
            allowed_methods=frozenset({"method_a"}),
            excluded_methods=frozenset({"method_a"}),
        )
        assert constraint.is_method_allowed("method_a") is False

    def test_is_category_allowed_no_restrictions(self):
        """Test is_category_allowed with no restrictions."""
        constraint = SelectionConstraint()
        assert constraint.is_category_allowed("any_category") is True

    def test_is_category_allowed_excluded(self):
        """Test is_category_allowed with excluded categories."""
        constraint = SelectionConstraint(
            excluded_categories=frozenset({"excluded_category"})
        )
        assert constraint.is_category_allowed("excluded_category") is False
        assert constraint.is_category_allowed("other_category") is True

    def test_is_category_allowed_whitelist(self):
        """Test is_category_allowed with allowed categories list."""
        constraint = SelectionConstraint(
            allowed_categories=frozenset({"allowed_category"})
        )
        assert constraint.is_category_allowed("allowed_category") is True
        assert constraint.is_category_allowed("other_category") is False


# =============================================================================
# Tests for MethodRecommendation
# =============================================================================


class TestMethodRecommendation:
    """Tests for MethodRecommendation dataclass."""

    def test_create_with_required_fields(self):
        """Test creating MethodRecommendation with required fields."""
        rec = MethodRecommendation(
            identifier="chain_of_thought",
            score=0.8,
            confidence=0.9,
            reasoning="Good for step-by-step",
        )
        assert rec.identifier == "chain_of_thought"
        assert rec.score == 0.8
        assert rec.confidence == 0.9
        assert rec.reasoning == "Good for step-by-step"

    def test_create_with_all_fields(self):
        """Test creating MethodRecommendation with all fields."""
        rec = MethodRecommendation(
            identifier="tree_of_thoughts",
            score=0.9,
            confidence=0.85,
            reasoning="Good for exploration",
            matched_tags=frozenset({"creative", "decision"}),
            strengths=("explores alternatives", "finds novel solutions"),
            weaknesses=("slower",),
            alternative_to="chain_of_thought",
        )
        assert rec.matched_tags == frozenset({"creative", "decision"})
        assert rec.alternative_to == "chain_of_thought"

    def test_immutability(self):
        """Test that MethodRecommendation is immutable."""
        rec = MethodRecommendation(
            identifier="test", score=0.5, confidence=0.5, reasoning="test"
        )
        with pytest.raises(AttributeError):
            rec.score = 1.0  # type: ignore[misc]

    def test_score_validation_min(self):
        """Test score validation minimum bound."""
        with pytest.raises(ValueError, match="score must be 0.0-1.0"):
            MethodRecommendation(
                identifier="test", score=-0.1, confidence=0.5, reasoning="test"
            )

    def test_score_validation_max(self):
        """Test score validation maximum bound."""
        with pytest.raises(ValueError, match="score must be 0.0-1.0"):
            MethodRecommendation(
                identifier="test", score=1.1, confidence=0.5, reasoning="test"
            )

    def test_confidence_validation_min(self):
        """Test confidence validation minimum bound."""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            MethodRecommendation(
                identifier="test", score=0.5, confidence=-0.1, reasoning="test"
            )

    def test_confidence_validation_max(self):
        """Test confidence validation maximum bound."""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            MethodRecommendation(
                identifier="test", score=0.5, confidence=1.1, reasoning="test"
            )

    def test_reasoning_validation_empty(self):
        """Test reasoning validation for empty string."""
        with pytest.raises(ValueError, match="reasoning cannot be empty"):
            MethodRecommendation(
                identifier="test", score=0.5, confidence=0.5, reasoning=""
            )

    def test_comparison_by_score(self):
        """Test that recommendations can be compared by score."""
        rec_low = MethodRecommendation(
            identifier="low", score=0.3, confidence=0.5, reasoning="low"
        )
        rec_high = MethodRecommendation(
            identifier="high", score=0.9, confidence=0.5, reasoning="high"
        )
        assert rec_low < rec_high
        assert not rec_high < rec_low

    def test_equality_by_identifier(self):
        """Test that equality is based on identifier."""
        rec1 = MethodRecommendation(
            identifier="same", score=0.3, confidence=0.5, reasoning="a"
        )
        rec2 = MethodRecommendation(
            identifier="same", score=0.9, confidence=0.9, reasoning="b"
        )
        assert rec1 == rec2

    def test_hash_by_identifier(self):
        """Test that hash is based on identifier."""
        rec1 = MethodRecommendation(
            identifier="same", score=0.3, confidence=0.5, reasoning="a"
        )
        rec2 = MethodRecommendation(
            identifier="same", score=0.9, confidence=0.9, reasoning="b"
        )
        assert hash(rec1) == hash(rec2)


# =============================================================================
# Tests for SelectionRule
# =============================================================================


class TestSelectionRule:
    """Tests for SelectionRule dataclass."""

    def test_create_rule(self):
        """Test creating a selection rule."""
        rule = SelectionRule(
            pattern="mathematical",
            method="chain_of_thought",
            score_boost=0.2,
            reason="Step-by-step helps with math",
        )
        assert rule.pattern == "mathematical"
        assert rule.method == "chain_of_thought"
        assert rule.score_boost == 0.2
        assert rule.reason == "Step-by-step helps with math"

    def test_immutability(self):
        """Test that SelectionRule is immutable."""
        rule = SelectionRule(
            pattern="test", method="test", score_boost=0.1, reason="test"
        )
        with pytest.raises(AttributeError):
            rule.score_boost = 0.5  # type: ignore[misc]

    def test_score_boost_validation_min(self):
        """Test score_boost validation minimum bound."""
        with pytest.raises(ValueError, match="score_boost must be 0.0-1.0"):
            SelectionRule(
                pattern="test", method="test", score_boost=-0.1, reason="test"
            )

    def test_score_boost_validation_max(self):
        """Test score_boost validation maximum bound."""
        with pytest.raises(ValueError, match="score_boost must be 0.0-1.0"):
            SelectionRule(
                pattern="test", method="test", score_boost=1.1, reason="test"
            )


# =============================================================================
# Tests for SELECTION_RULES
# =============================================================================


class TestSelectionRules:
    """Tests for SELECTION_RULES list."""

    def test_selection_rules_not_empty(self):
        """Test that SELECTION_RULES contains rules."""
        assert len(SELECTION_RULES) > 0

    def test_selection_rules_are_valid(self):
        """Test that all SELECTION_RULES are valid SelectionRule instances."""
        for rule in SELECTION_RULES:
            assert isinstance(rule, SelectionRule)
            assert rule.pattern
            assert rule.method
            assert 0 <= rule.score_boost <= 1
            assert rule.reason

    def test_selection_rules_cover_key_patterns(self):
        """Test that rules cover key pattern types."""
        patterns_covered = {rule.pattern for rule in SELECTION_RULES}
        expected_patterns = {"mathematical", "code", "ethical", "creative"}
        assert expected_patterns.issubset(patterns_covered)


# =============================================================================
# Tests for MethodSelector
# =============================================================================


class TestMethodSelector:
    """Tests for MethodSelector class."""

    def test_create_selector(self, mock_registry: MethodRegistry):
        """Test creating a MethodSelector."""
        selector = MethodSelector(mock_registry)
        assert selector is not None

    def test_analyze_mathematical(self, mock_registry: MethodRegistry):
        """Test analyzing a mathematical problem."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze("Calculate the sum of 2 and 3")
        assert hint.problem_type == "mathematical"
        assert "mathematical" in hint.domain_tags

    def test_analyze_code(self, mock_registry: MethodRegistry):
        """Test analyzing a code problem."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze("Debug this function that returns an error")
        assert hint.problem_type == "code"
        assert "code" in hint.domain_tags

    def test_analyze_general(self, mock_registry: MethodRegistry):
        """Test analyzing a general problem."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze("Hello world")
        assert hint.problem_type == "general"
        assert len(hint.domain_tags) == 0

    def test_analyze_complexity(self, mock_registry: MethodRegistry):
        """Test that analysis estimates complexity."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze(
            "Calculate, analyze, and debug this code step by step"
        )
        assert hint.complexity_estimate > 3  # Multiple patterns detected

    def test_analyze_requires_branching(self, mock_registry: MethodRegistry):
        """Test analysis detects branching requirements."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze("Decide between option A and option B")
        assert hint.requires_branching is True

    def test_analyze_requires_iteration(self, mock_registry: MethodRegistry):
        """Test analysis detects iteration requirements."""
        selector = MethodSelector(mock_registry)
        hint = selector.analyze("First do this, then do that, next continue")
        assert hint.requires_iteration is True

    def test_recommend_returns_list(self, mock_registry: MethodRegistry):
        """Test that recommend returns a list of recommendations."""
        selector = MethodSelector(mock_registry)
        recommendations = selector.recommend("Calculate 2 + 2")
        assert isinstance(recommendations, list)

    def test_recommend_max_recommendations(self, mock_registry: MethodRegistry):
        """Test that recommend respects max_recommendations."""
        selector = MethodSelector(mock_registry)
        recommendations = selector.recommend(
            "Calculate 2 + 2", max_recommendations=2
        )
        assert len(recommendations) <= 2

    def test_recommend_sorted_by_score(self, mock_registry: MethodRegistry):
        """Test that recommendations are sorted by score (descending)."""
        selector = MethodSelector(mock_registry)
        recommendations = selector.recommend("Calculate 2 + 2")
        if len(recommendations) > 1:
            scores = [r.score for r in recommendations]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_with_constraints_excluded_method(
        self, mock_registry: MethodRegistry
    ):
        """Test recommendations with excluded method."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(
            excluded_methods=frozenset({"chain_of_thought"})
        )
        recommendations = selector.recommend(
            "Calculate step by step", constraints=constraints
        )
        identifiers = {r.identifier for r in recommendations}
        assert "chain_of_thought" not in identifiers

    def test_recommend_with_constraints_allowed_category(
        self, mock_registry: MethodRegistry
    ):
        """Test recommendations with allowed category."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(
            allowed_categories=frozenset({"core"})
        )
        recommendations = selector.recommend(
            "Some problem", constraints=constraints
        )
        # All recommendations should be from core category
        for rec in recommendations:
            metadata = mock_registry.get_metadata(rec.identifier)
            assert metadata is not None
            assert str(metadata.category) == "core"

    def test_recommend_with_constraints_max_complexity(
        self, mock_registry: MethodRegistry
    ):
        """Test recommendations with max complexity."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(max_complexity=4)
        recommendations = selector.recommend(
            "Some problem", constraints=constraints
        )
        for rec in recommendations:
            metadata = mock_registry.get_metadata(rec.identifier)
            assert metadata is not None
            assert metadata.complexity <= 4

    def test_recommend_with_constraints_require_branching(
        self, mock_registry: MethodRegistry
    ):
        """Test recommendations with branching requirement."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(require_branching=True)
        recommendations = selector.recommend(
            "Some problem", constraints=constraints
        )
        for rec in recommendations:
            metadata = mock_registry.get_metadata(rec.identifier)
            assert metadata is not None
            assert metadata.supports_branching is True

    def test_recommend_with_constraints_preferred_methods(
        self, mock_registry: MethodRegistry
    ):
        """Test recommendations with preferred methods."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(
            preferred_methods=frozenset({"tree_of_thoughts"})
        )
        recommendations = selector.recommend(
            "Some problem", constraints=constraints
        )
        # tree_of_thoughts should have boosted score
        tot_rec = next(
            (r for r in recommendations if r.identifier == "tree_of_thoughts"),
            None,
        )
        if tot_rec:
            assert "User preferred method" in tot_rec.reasoning

    def test_recommend_boosts_matching_methods(self, mock_registry: MethodRegistry):
        """Test that recommendations boost methods matching patterns."""
        selector = MethodSelector(mock_registry)
        recommendations = selector.recommend("Calculate the equation")
        # mathematical_reasoning should be recommended for math problems
        math_rec = next(
            (
                r
                for r in recommendations
                if r.identifier == "mathematical_reasoning"
            ),
            None,
        )
        assert math_rec is not None

    def test_select_best_returns_identifier(self, mock_registry: MethodRegistry):
        """Test that select_best returns a method identifier."""
        selector = MethodSelector(mock_registry)
        best = selector.select_best("Calculate 2 + 2")
        assert best is not None
        assert isinstance(best, str)

    def test_select_best_respects_constraints(self, mock_registry: MethodRegistry):
        """Test that select_best respects constraints."""
        selector = MethodSelector(mock_registry)
        constraints = SelectionConstraint(
            excluded_methods=frozenset(
                {"mathematical_reasoning", "chain_of_thought"}
            )
        )
        best = selector.select_best("Calculate 2 + 2", constraints=constraints)
        assert best not in {"mathematical_reasoning", "chain_of_thought"}

    def test_select_best_returns_none_no_matches(
        self, mock_registry: MethodRegistry
    ):
        """Test select_best returns None when no methods match constraints."""
        selector = MethodSelector(mock_registry)
        # Exclude all methods
        constraints = SelectionConstraint(
            allowed_methods=frozenset({"nonexistent_method"})
        )
        best = selector.select_best("Some problem", constraints=constraints)
        assert best is None

    def test_recommend_empty_registry(self):
        """Test recommendations with empty registry."""
        empty_registry = MethodRegistry()
        selector = MethodSelector(empty_registry)
        recommendations = selector.recommend("Some problem")
        assert recommendations == []

    def test_recommend_filters_low_scores(self, mock_registry: MethodRegistry):
        """Test that recommendations filter out low-scoring methods."""
        selector = MethodSelector(mock_registry)
        recommendations = selector.recommend("Hello world")
        # All recommendations should have score >= 0.15 (filter threshold)
        for rec in recommendations:
            assert rec.score >= 0.15
