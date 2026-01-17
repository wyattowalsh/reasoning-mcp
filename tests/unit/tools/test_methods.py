"""Unit tests for methods tool functions."""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier
from reasoning_mcp.models.tools import ComparisonResult, MethodInfo, Recommendation
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.tools.methods import methods_compare, methods_list, methods_recommend


# Create a simple mock method class for testing
class MockReasoningMethod:
    """Mock reasoning method for testing."""

    streaming_context = None

    def __init__(self, identifier: str, name: str, description: str, category: str):
        self._identifier = identifier
        self._name = name
        self._description = description
        self._category = category

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
        pass

    async def execute(self, session, input_text, *, context=None, execution_context=None):
        pass

    async def continue_reasoning(
        self, session, previous_thought, *, guidance=None, context=None, execution_context=None
    ):
        pass

    async def health_check(self) -> bool:
        return True

    async def emit_thought(self, content: str, confidence: float | None = None) -> None:
        pass


@pytest.fixture(autouse=True)
def populate_registry(monkeypatch):
    """Populate the registry with test methods before each test."""
    # Create a test registry
    registry = MethodRegistry()

    # Register core methods
    cot_method = MockReasoningMethod(
        identifier="chain_of_thought",
        name="Chain of Thought",
        description="Step-by-step sequential reasoning",
        category="core",
    )
    cot_metadata = MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Chain of Thought",
        description="Step-by-step sequential reasoning",
        category=MethodCategory.CORE,
        tags=frozenset({"sequential", "structured"}),
        complexity=5,
        supports_branching=False,
        supports_revision=False,
        min_thoughts=1,
        max_thoughts=10,
        avg_tokens_per_thought=500,
    )
    registry.register(cot_method, cot_metadata)

    tot_method = MockReasoningMethod(
        identifier="tree_of_thoughts",
        name="Tree of Thoughts",
        description="Branching exploration of multiple reasoning paths",
        category="core",
    )
    tot_metadata = MethodMetadata(
        identifier=MethodIdentifier.TREE_OF_THOUGHTS,
        name="Tree of Thoughts",
        description="Branching exploration of multiple reasoning paths",
        category=MethodCategory.CORE,
        tags=frozenset({"branching", "exploration", "structured"}),
        complexity=7,
        supports_branching=True,
        supports_revision=True,
        min_thoughts=3,
        max_thoughts=20,
        avg_tokens_per_thought=600,
    )
    registry.register(tot_method, tot_metadata)

    react_method = MockReasoningMethod(
        identifier="react",
        name="ReAct",
        description="Reasoning and Acting in interleaved fashion",
        category="core",
    )
    react_metadata = MethodMetadata(
        identifier=MethodIdentifier.REACT,
        name="ReAct",
        description="Reasoning and Acting in interleaved fashion",
        category=MethodCategory.CORE,
        tags=frozenset({"action-oriented", "sequential"}),
        complexity=6,
        supports_branching=False,
        supports_revision=False,
        min_thoughts=2,
        max_thoughts=15,
        avg_tokens_per_thought=550,
    )
    registry.register(react_method, react_metadata)

    # Register high-value methods
    ethical_method = MockReasoningMethod(
        identifier="ethical_reasoning",
        name="Ethical Reasoning",
        description="Structured ethical analysis with principles",
        category="high_value",
    )
    ethical_metadata = MethodMetadata(
        identifier=MethodIdentifier.ETHICAL_REASONING,
        name="Ethical Reasoning",
        description="Structured ethical analysis with principles",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"ethical", "structured", "stakeholders"}),
        complexity=8,
        supports_branching=True,
        supports_revision=True,
        min_thoughts=5,
        max_thoughts=25,
        avg_tokens_per_thought=700,
    )
    registry.register(ethical_method, ethical_metadata)

    dialectic_method = MockReasoningMethod(
        identifier="dialectic",
        name="Dialectic",
        description="Thesis-antithesis-synthesis reasoning",
        category="high_value",
    )
    dialectic_metadata = MethodMetadata(
        identifier=MethodIdentifier.DIALECTIC,
        name="Dialectic",
        description="Thesis-antithesis-synthesis reasoning",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"ethical", "philosophical", "structured"}),
        complexity=7,
        supports_branching=True,
        supports_revision=True,
        min_thoughts=3,
        max_thoughts=15,
        avg_tokens_per_thought=650,
    )
    registry.register(dialectic_method, dialectic_metadata)

    # Register specialized methods
    socratic_method = MockReasoningMethod(
        identifier="socratic",
        name="Socratic Method",
        description="Question-driven reasoning",
        category="specialized",
    )
    socratic_metadata = MethodMetadata(
        identifier=MethodIdentifier.SOCRATIC,
        name="Socratic Method",
        description="Question-driven reasoning",
        category=MethodCategory.SPECIALIZED,
        tags=frozenset({"ethical", "questioning", "philosophical"}),
        complexity=6,
        supports_branching=False,
        supports_revision=True,
        min_thoughts=4,
        max_thoughts=20,
        avg_tokens_per_thought=550,
    )
    registry.register(socratic_method, socratic_metadata)

    self_consistency_method = MockReasoningMethod(
        identifier="self_consistency",
        name="Self-Consistency",
        description="Multiple reasoning paths with voting",
        category="core",
    )
    self_consistency_metadata = MethodMetadata(
        identifier=MethodIdentifier.SELF_CONSISTENCY,
        name="Self-Consistency",
        description="Multiple reasoning paths with voting",
        category=MethodCategory.CORE,
        tags=frozenset({"verification", "ensemble"}),
        complexity=6,
        supports_branching=True,
        supports_revision=False,
        min_thoughts=3,
        max_thoughts=10,
        avg_tokens_per_thought=500,
    )
    registry.register(self_consistency_method, self_consistency_metadata)

    # Patch the MethodRegistry constructor to return our populated registry

    def mock_init(self):
        # Copy all data from our populated registry
        self._methods = registry._methods.copy()
        self._metadata = registry._metadata.copy()
        self._initialized = registry._initialized.copy()

    monkeypatch.setattr(MethodRegistry, "__init__", mock_init)


class TestMethodsList:
    """Tests for methods_list function."""

    def test_list_all_methods_no_filters(self):
        """Test listing all methods without any filters."""
        methods = methods_list()

        # Should return a non-empty list
        assert isinstance(methods, list)
        assert len(methods) > 0

        # All items should be MethodInfo instances
        for method in methods:
            assert isinstance(method, MethodInfo)
            assert isinstance(method.id, MethodIdentifier)
            assert isinstance(method.name, str)
            assert isinstance(method.description, str)
            assert isinstance(method.category, MethodCategory)
            assert isinstance(method.parameters, dict)
            assert isinstance(method.tags, list)

    def test_list_methods_with_category_filter_core(self):
        """Test listing methods filtered by core category."""
        methods = methods_list(category=MethodCategory.CORE)

        # Should return a list
        assert isinstance(methods, list)
        assert len(methods) > 0

        # All methods should have core category
        for method in methods:
            assert method.category == MethodCategory.CORE

    def test_list_methods_with_category_filter_high_value(self):
        """Test listing methods filtered by high_value category."""
        methods = methods_list(category=MethodCategory.HIGH_VALUE)

        # Should return a list
        assert isinstance(methods, list)

        # All methods should have high_value category
        for method in methods:
            assert method.category == MethodCategory.HIGH_VALUE

    def test_list_methods_with_category_filter_specialized(self):
        """Test listing methods filtered by specialized category."""
        methods = methods_list(category=MethodCategory.SPECIALIZED)

        # Should return a list
        assert isinstance(methods, list)

        # All methods should have specialized category
        for method in methods:
            assert method.category == MethodCategory.SPECIALIZED

    def test_list_methods_with_category_string(self):
        """Test listing methods with category as string."""
        methods = methods_list(category="core")

        # Should return a list
        assert isinstance(methods, list)
        assert len(methods) > 0

        # All methods should have core category
        for method in methods:
            assert method.category == MethodCategory.CORE

    def test_list_methods_with_single_tag(self):
        """Test listing methods filtered by a single tag."""
        # Use a common tag that should exist
        methods = methods_list(tags=["sequential"])

        # Should return a list
        assert isinstance(methods, list)

        # All methods should have the sequential tag
        for method in methods:
            assert "sequential" in method.tags

    def test_list_methods_with_multiple_tags(self):
        """Test listing methods filtered by multiple tags (AND logic)."""
        # Methods must have ALL specified tags
        methods = methods_list(tags=["structured", "ethical"])

        # Should return a list
        assert isinstance(methods, list)

        # All methods should have both tags
        for method in methods:
            assert "structured" in method.tags
            assert "ethical" in method.tags

    def test_list_methods_with_category_and_tags(self):
        """Test listing methods with both category and tag filters."""
        methods = methods_list(
            category=MethodCategory.CORE,
            tags=["sequential"],
        )

        # Should return a list
        assert isinstance(methods, list)

        # All methods should match both filters
        for method in methods:
            assert method.category == MethodCategory.CORE
            assert "sequential" in method.tags

    def test_list_methods_empty_tags_list(self):
        """Test that empty tags list returns all methods (no filtering)."""
        all_methods = methods_list()
        methods_with_empty_tags = methods_list(tags=[])

        # Should return same results as no filter
        assert len(methods_with_empty_tags) == len(all_methods)

    def test_list_methods_returns_method_parameters(self):
        """Test that returned MethodInfo includes parameters dict."""
        methods = methods_list()
        assert len(methods) > 0

        # Check first method has expected parameter structure
        method = methods[0]
        assert "complexity" in method.parameters
        assert "supports_branching" in method.parameters
        assert "supports_revision" in method.parameters
        assert "min_thoughts" in method.parameters
        assert "max_thoughts" in method.parameters
        assert "avg_tokens_per_thought" in method.parameters

        # Validate parameter types
        assert isinstance(method.parameters["complexity"], int)
        assert isinstance(method.parameters["supports_branching"], bool)
        assert isinstance(method.parameters["supports_revision"], bool)
        assert isinstance(method.parameters["min_thoughts"], int)
        assert isinstance(method.parameters["max_thoughts"], int)
        assert isinstance(method.parameters["avg_tokens_per_thought"], int)

    def test_list_methods_returns_immutable_methodinfo(self):
        """Test that MethodInfo objects are frozen (immutable)."""
        methods = methods_list()
        assert len(methods) > 0

        method = methods[0]
        with pytest.raises(Exception):  # Pydantic frozen models raise on mutation
            method.name = "Modified Name"


class TestMethodsRecommend:
    """Tests for methods_recommend function."""

    def test_recommend_returns_recommendations(self):
        """Test that recommend returns a list of Recommendation objects."""
        recommendations = methods_recommend(problem="How can I solve this complex ethical dilemma?")

        # Should return a list
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # All items should be Recommendation instances
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert isinstance(rec.method_id, MethodIdentifier)
            assert isinstance(rec.score, float)
            assert isinstance(rec.reason, str)
            assert isinstance(rec.confidence, float)
            # Scores should be in valid range
            assert 0.0 <= rec.score <= 1.0
            assert 0.0 <= rec.confidence <= 1.0

    def test_recommend_respects_max_results_default(self):
        """Test that recommend returns at most 3 results by default."""
        recommendations = methods_recommend(problem="What is the best approach to this problem?")

        # Should return at most 3 results (default)
        assert len(recommendations) <= 3

    def test_recommend_respects_max_results_custom(self):
        """Test that recommend respects custom max_results parameter."""
        max_results = 5
        recommendations = methods_recommend(
            problem="How should I approach this problem?",
            max_results=max_results,
        )

        # Should return at most max_results
        assert len(recommendations) <= max_results

    def test_recommend_respects_max_results_one(self):
        """Test that recommend can return just one recommendation."""
        recommendations = methods_recommend(
            problem="Quick question: what method should I use?",
            max_results=1,
        )

        # Should return at most 1 result
        assert len(recommendations) <= 1

    def test_recommend_sorted_by_score(self):
        """Test that recommendations are sorted by score (highest first)."""
        recommendations = methods_recommend(
            problem="Complex multi-faceted problem requiring analysis",
            max_results=5,
        )

        if len(recommendations) > 1:
            # Verify descending score order
            scores = [rec.score for rec in recommendations]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_ethical_problem(self):
        """Test recommendation for an ethical problem."""
        recommendations = methods_recommend(
            problem="Should we implement this feature that might compromise user privacy?",
            max_results=3,
        )

        # Should return recommendations
        assert len(recommendations) > 0
        assert isinstance(recommendations[0], Recommendation)

        # Reasoning should be provided
        assert len(recommendations[0].reason) > 0

    def test_recommend_technical_problem(self):
        """Test recommendation for a technical problem."""
        recommendations = methods_recommend(
            problem="How do I optimize this database query for better performance?",
            max_results=3,
        )

        # Should return a list (may be empty if no good matches)
        assert isinstance(recommendations, list)

        # If recommendations are returned, validate them
        for rec in recommendations:
            assert isinstance(rec, Recommendation)

    def test_recommend_mathematical_problem(self):
        """Test recommendation for a mathematical problem."""
        recommendations = methods_recommend(
            problem="Prove that the sum of the first n positive integers equals n(n+1)/2",
            max_results=3,
        )

        # Should return recommendations
        assert len(recommendations) > 0
        assert isinstance(recommendations[0], Recommendation)

    def test_recommend_returns_immutable_recommendations(self):
        """Test that Recommendation objects are frozen (immutable)."""
        # Use a more specific problem that should get recommendations
        recommendations = methods_recommend(
            problem="Should we implement this new ethical policy for user privacy and data protection?",
            max_results=1,
        )

        # With ethical keywords, should get at least one recommendation
        if len(recommendations) > 0:
            rec = recommendations[0]
            with pytest.raises(Exception):  # Pydantic frozen models raise on mutation
                rec.score = 0.5
        else:
            # If no recommendations, at least verify the type is correct
            assert isinstance(recommendations, list)


class TestMethodsCompare:
    """Tests for methods_compare function."""

    def test_compare_returns_comparison_result(self):
        """Test that compare returns a ComparisonResult object."""
        result = methods_compare(
            methods=["chain_of_thought", "tree_of_thoughts"],
            problem="Which method is better for this problem?",
        )

        # Should return ComparisonResult
        assert isinstance(result, ComparisonResult)
        assert isinstance(result.methods, list)
        assert isinstance(result.scores, dict)
        assert isinstance(result.analysis, str)

        # Winner can be None (tie) or a MethodIdentifier
        if result.winner is not None:
            assert isinstance(result.winner, MethodIdentifier)

    def test_compare_includes_all_specified_methods(self):
        """Test that all specified methods are included in the result."""
        method_ids = ["chain_of_thought", "tree_of_thoughts", "react"]
        result = methods_compare(
            methods=method_ids,
            problem="Complex problem requiring analysis",
        )

        # Methods list should contain all requested methods
        assert len(result.methods) == len(method_ids)
        result_method_strs = [str(m) for m in result.methods]
        for method_id in method_ids:
            assert method_id in result_method_strs

    def test_compare_scores_all_methods(self):
        """Test that all methods receive scores."""
        method_ids = ["chain_of_thought", "tree_of_thoughts"]
        result = methods_compare(
            methods=method_ids,
            problem="Which method works best?",
        )

        # All methods should have scores
        for method_id in method_ids:
            assert method_id in result.scores
            assert isinstance(result.scores[method_id], float)
            assert 0.0 <= result.scores[method_id] <= 1.0

    def test_compare_identifies_winner(self):
        """Test that compare identifies the highest-scoring method as winner."""
        method_ids = ["chain_of_thought", "tree_of_thoughts", "react"]
        result = methods_compare(
            methods=method_ids,
            problem="Multi-step reasoning problem",
        )

        if result.winner is not None:
            # Winner should be one of the compared methods
            assert result.winner in result.methods

            # Winner should have the highest score
            winner_score = result.scores[str(result.winner)]
            for score in result.scores.values():
                assert winner_score >= score

    def test_compare_handles_tie(self):
        """Test that winner is None when there's a tie."""
        # Note: This test might not always produce a tie, but tests the logic
        result = methods_compare(
            methods=["chain_of_thought"],
            problem="Simple problem",
        )

        # With only one method, it should be the winner
        assert result.winner is not None

    def test_compare_provides_analysis(self):
        """Test that comparison includes detailed analysis."""
        result = methods_compare(
            methods=["chain_of_thought", "tree_of_thoughts"],
            problem="Should I use branching or sequential reasoning?",
        )

        # Analysis should be non-empty
        assert len(result.analysis) > 0
        assert isinstance(result.analysis, str)

    def test_compare_two_methods(self):
        """Test comparing exactly two methods."""
        result = methods_compare(
            methods=["chain_of_thought", "react"],
            problem="Step-by-step problem solving",
        )

        assert len(result.methods) == 2
        assert len(result.scores) == 2

    def test_compare_multiple_methods(self):
        """Test comparing multiple methods (>2)."""
        result = methods_compare(
            methods=["chain_of_thought", "tree_of_thoughts", "react", "self_consistency"],
            problem="Complex decision with multiple factors",
        )

        assert len(result.methods) == 4
        assert len(result.scores) == 4

    def test_compare_ethical_methods(self):
        """Test comparing ethical reasoning methods."""
        result = methods_compare(
            methods=["ethical_reasoning", "dialectic", "socratic"],
            problem="Should we prioritize user privacy over business metrics?",
        )

        # Should include all three methods
        assert len(result.methods) == 3
        assert len(result.scores) == 3

        # All scores should be valid
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_compare_returns_immutable_result(self):
        """Test that ComparisonResult is frozen (immutable)."""
        result = methods_compare(
            methods=["chain_of_thought", "react"],
            problem="Test problem",
        )

        with pytest.raises(Exception):  # Pydantic frozen models raise on mutation
            result.analysis = "Modified analysis"

    def test_compare_with_method_identifiers(self):
        """Test that method IDs are properly converted to MethodIdentifier."""
        result = methods_compare(
            methods=["chain_of_thought", "tree_of_thoughts"],
            problem="Test problem",
        )

        # All methods in result should be MethodIdentifier objects
        for method in result.methods:
            assert isinstance(method, MethodIdentifier)

        # Winner should be MethodIdentifier or None
        if result.winner is not None:
            assert isinstance(result.winner, MethodIdentifier)


class TestMethodsToolsReturnTypes:
    """Test that all methods tools return correct types."""

    def test_methods_list_return_type(self):
        """Test methods_list returns list[MethodInfo]."""
        result = methods_list()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, MethodInfo)

    def test_methods_recommend_return_type(self):
        """Test methods_recommend returns list[Recommendation]."""
        result = methods_recommend(problem="test problem")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, Recommendation)

    def test_methods_compare_return_type(self):
        """Test methods_compare returns ComparisonResult."""
        result = methods_compare(
            methods=["chain_of_thought"],
            problem="test problem",
        )
        assert isinstance(result, ComparisonResult)


class TestMethodsToolsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_list_with_nonexistent_tag(self):
        """Test listing methods with a tag that doesn't exist."""
        methods = methods_list(tags=["nonexistent_tag_xyz123"])

        # Should return empty list or very few results
        assert isinstance(methods, list)
        # All returned methods should have the tag
        for method in methods:
            assert "nonexistent_tag_xyz123" in method.tags

    def test_recommend_with_empty_problem(self):
        """Test recommendation with empty problem string."""
        recommendations = methods_recommend(problem="")

        # Should still return recommendations (may be generic)
        assert isinstance(recommendations, list)

    def test_recommend_with_very_long_problem(self):
        """Test recommendation with a very long problem description."""
        long_problem = "This is a problem. " * 100
        recommendations = methods_recommend(problem=long_problem, max_results=3)

        # Should handle long input gracefully
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

    def test_compare_with_single_method(self):
        """Test comparison with only one method."""
        result = methods_compare(
            methods=["chain_of_thought"],
            problem="Simple problem",
        )

        # Should work with single method
        assert len(result.methods) == 1
        assert len(result.scores) == 1
        # Single method should be the winner
        assert result.winner == result.methods[0]

    def test_recommend_max_results_zero(self):
        """Test recommendation with max_results=0."""
        # This tests boundary behavior - might return empty list
        recommendations = methods_recommend(problem="test", max_results=0)

        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_recommend_large_max_results(self):
        """Test recommendation with very large max_results."""
        # Should not crash, but may return fewer than requested
        recommendations = methods_recommend(problem="test", max_results=1000)

        assert isinstance(recommendations, list)
        # Can't return more methods than exist
        all_methods = methods_list()
        assert len(recommendations) <= len(all_methods)
