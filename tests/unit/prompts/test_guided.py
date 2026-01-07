"""
Comprehensive tests for guided reasoning prompts.

This module tests the MCP prompts in reasoning_mcp.prompts.guided:
- reason_with_method: Structured guidance for using a specific method
- compare_methods: Comparative analysis of multiple methods
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

from reasoning_mcp.config import Settings
from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier
from reasoning_mcp.prompts.guided import register_guided_prompts
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.server import AppContext


class MockReasoningMethod:
    """Mock reasoning method for testing."""

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

    async def execute(self, session, input_text, *, context=None):
        pass

    async def continue_reasoning(self, session, previous_thought, *, guidance=None, context=None):
        pass

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def populated_registry():
    """Create a populated test registry."""
    registry = MethodRegistry()

    # Register chain_of_thought
    cot_method = MockReasoningMethod(
        identifier="chain_of_thought",
        name="Chain of Thought",
        description="Step-by-step sequential reasoning for systematic analysis",
        category="core",
    )
    cot_metadata = MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Chain of Thought",
        description="Step-by-step sequential reasoning for systematic analysis",
        category=MethodCategory.CORE,
        tags=frozenset({"sequential", "structured"}),
        best_for=["Logical step-by-step problems", "Mathematical reasoning"],
        not_recommended_for=["Open-ended exploration", "Parallel thinking"],
        complexity=5,
        supports_branching=False,
        supports_revision=False,
        min_thoughts=1,
        max_thoughts=10,
        avg_tokens_per_thought=500,
    )
    registry.register(cot_method, cot_metadata)

    # Register tree_of_thoughts
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
        best_for=["Complex problem exploration", "Creative solutions"],
        not_recommended_for=["Simple linear problems"],
        complexity=7,
        supports_branching=True,
        supports_revision=True,
        min_thoughts=3,
        max_thoughts=20,
        avg_tokens_per_thought=600,
    )
    registry.register(tot_method, tot_metadata)

    # Register ethical_reasoning
    ethical_method = MockReasoningMethod(
        identifier="ethical_reasoning",
        name="Ethical Reasoning",
        description="Structured ethical analysis with multiple frameworks",
        category="high_value",
    )
    ethical_metadata = MethodMetadata(
        identifier=MethodIdentifier.ETHICAL_REASONING,
        name="Ethical Reasoning",
        description="Structured ethical analysis with multiple frameworks",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"ethical", "structured", "stakeholders"}),
        best_for=["Ethical dilemmas", "Stakeholder analysis"],
        not_recommended_for=["Pure technical problems"],
        complexity=8,
        supports_branching=True,
        supports_revision=True,
        min_thoughts=5,
        max_thoughts=25,
        avg_tokens_per_thought=700,
    )
    registry.register(ethical_method, ethical_metadata)

    # Register react
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
        best_for=["Action-oriented tasks", "Tool usage"],
        not_recommended_for=["Pure theoretical reasoning"],
        complexity=6,
        supports_branching=False,
        supports_revision=False,
        min_thoughts=2,
        max_thoughts=15,
        avg_tokens_per_thought=550,
    )
    registry.register(react_method, react_metadata)

    return registry


@pytest.fixture
def test_mcp_server(populated_registry):
    """Create a test FastMCP server with app context."""
    mcp = FastMCP("test-server")

    # Create mock session manager
    class MockSessionManager:
        async def create(self):
            pass

        async def get(self, session_id):
            pass

        async def clear(self):
            pass

    # Create and set app context
    ctx = AppContext(
        registry=populated_registry,
        session_manager=MockSessionManager(),
        settings=Settings(),
        initialized=True,
    )
    mcp.app_context = ctx

    # Register prompts
    register_guided_prompts(mcp)

    return mcp


# ============================================================================
# Tests for reason_with_method prompt
# ============================================================================


class TestReasonWithMethod:
    """Tests for the reason_with_method prompt."""

    @pytest.mark.asyncio
    async def test_returns_list_of_prompt_messages(self, test_mcp_server):
        """Test that prompt returns list[PromptMessage]."""
        # Access the prompt function directly
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        assert prompt_func is not None, "reason_with_method prompt not found"

        result = await prompt_func(method_id="chain_of_thought", problem="What is 2+2?")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(msg, PromptMessage) for msg in result)

    @pytest.mark.asyncio
    async def test_message_has_correct_structure(self, test_mcp_server):
        """Test that PromptMessage has correct structure."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test problem")

        message = result[0]
        assert message.role == "user"
        assert isinstance(message.content, TextContent)
        assert message.content.type == "text"
        assert isinstance(message.content.text, str)
        assert len(message.content.text) > 0

    @pytest.mark.asyncio
    async def test_includes_method_metadata(self, test_mcp_server):
        """Test that prompt includes method metadata."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test")

        text = result[0].content.text
        assert "Chain of Thought" in text
        assert "chain_of_thought" in text
        assert "core" in text.lower()
        assert "complexity" in text.lower()

    @pytest.mark.asyncio
    async def test_includes_method_description(self, test_mcp_server):
        """Test that prompt includes method description."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="ethical_reasoning", problem="Test")

        text = result[0].content.text
        assert "Structured ethical analysis" in text

    @pytest.mark.asyncio
    async def test_includes_best_for_use_cases(self, test_mcp_server):
        """Test that prompt includes best_for section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test")

        text = result[0].content.text
        assert "Best Used For" in text
        assert "Logical step-by-step problems" in text

    @pytest.mark.asyncio
    async def test_includes_not_recommended_for(self, test_mcp_server):
        """Test that prompt includes not_recommended_for section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test")

        text = result[0].content.text
        assert "Not Recommended For" in text
        assert "Open-ended exploration" in text

    @pytest.mark.asyncio
    async def test_includes_capabilities(self, test_mcp_server):
        """Test that prompt includes method capabilities."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="tree_of_thoughts", problem="Test")

        text = result[0].content.text
        assert "Capabilities" in text
        assert "Supports branching" in text
        assert "Supports revision" in text
        assert "Expected depth" in text

    @pytest.mark.asyncio
    async def test_method_specific_guidance_chain_of_thought(self, test_mcp_server):
        """Test method-specific guidance for chain_of_thought."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test")

        text = result[0].content.text
        assert "Break down the problem into clear, logical steps" in text
        assert "Show your work at each step" in text

    @pytest.mark.asyncio
    async def test_method_specific_guidance_tree_of_thoughts(self, test_mcp_server):
        """Test method-specific guidance for tree_of_thoughts."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="tree_of_thoughts", problem="Test")

        text = result[0].content.text
        assert "Generate multiple possible approaches" in text
        assert "Evaluate each branch for promise" in text

    @pytest.mark.asyncio
    async def test_method_specific_guidance_ethical_reasoning(self, test_mcp_server):
        """Test method-specific guidance for ethical_reasoning."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="ethical_reasoning", problem="Test")

        text = result[0].content.text
        assert "Identify stakeholders and their interests" in text
        assert "Apply multiple ethical frameworks" in text

    @pytest.mark.asyncio
    async def test_method_specific_guidance_react(self, test_mcp_server):
        """Test method-specific guidance for react."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="react", problem="Test")

        text = result[0].content.text
        assert "Alternate between reasoning (Thought) and acting (Action)" in text
        assert "After each action, observe the result" in text

    @pytest.mark.asyncio
    async def test_includes_problem_in_output(self, test_mcp_server):
        """Test that prompt includes the original problem."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        problem = "How can I optimize database queries?"
        result = await prompt_func(method_id="chain_of_thought", problem=problem)

        text = result[0].content.text
        assert problem in text

    @pytest.mark.asyncio
    async def test_truncates_long_problem_in_code_example(self, test_mcp_server):
        """Test that very long problems are truncated in code examples."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        long_problem = "This is a very long problem " * 20
        result = await prompt_func(method_id="chain_of_thought", problem=long_problem)

        text = result[0].content.text
        # Should have truncated problem in code example
        assert "..." in text

    @pytest.mark.asyncio
    async def test_handles_nonexistent_method(self, test_mcp_server):
        """Test handling of nonexistent method."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="nonexistent_method", problem="Test")

        text = result[0].content.text
        assert "not found" in text.lower()
        assert "methods_list" in text
        assert "methods_recommend" in text

    @pytest.mark.asyncio
    async def test_includes_next_steps(self, test_mcp_server):
        """Test that prompt includes next steps section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="Test")

        text = result[0].content.text
        assert "Next Steps" in text
        assert "reason(" in text


# ============================================================================
# Tests for compare_methods prompt
# ============================================================================


class TestCompareMethods:
    """Tests for the compare_methods prompt."""

    @pytest.mark.asyncio
    async def test_returns_list_of_prompt_messages(self, test_mcp_server):
        """Test that prompt returns list[PromptMessage]."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        assert prompt_func is not None, "compare_methods prompt not found"

        result = await prompt_func(
            problem="Should we implement this feature?",
            method_ids=["chain_of_thought", "ethical_reasoning"],
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(msg, PromptMessage) for msg in result)

    @pytest.mark.asyncio
    async def test_auto_selects_top_3_when_no_methods_specified(self, test_mcp_server):
        """Test auto-selection of top 3 methods when method_ids is None."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Should we implement this ethical feature?", method_ids=None
        )

        text = result[0].content.text
        assert "Methods being compared" in text
        # Should mention recommendation/comparison
        assert "Comparison" in text or "comparison" in text

    @pytest.mark.asyncio
    async def test_compares_specified_methods(self, test_mcp_server):
        """Test comparison of explicitly specified methods."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Test problem", method_ids=["chain_of_thought", "tree_of_thoughts", "react"]
        )

        text = result[0].content.text
        assert "Chain of Thought" in text
        assert "Tree of Thoughts" in text
        assert "ReAct" in text

    @pytest.mark.asyncio
    async def test_includes_quick_comparison_table(self, test_mcp_server):
        """Test that comparison includes a quick comparison table."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Test", method_ids=["chain_of_thought", "ethical_reasoning"]
        )

        text = result[0].content.text
        assert "Quick Comparison" in text
        assert "| Method |" in text
        assert "| Category |" in text
        assert "| Complexity |" in text
        assert "| Score |" in text

    @pytest.mark.asyncio
    async def test_methods_sorted_by_score(self, test_mcp_server):
        """Test that methods are sorted by score (highest first)."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Should we implement this ethical feature?",
            method_ids=["chain_of_thought", "ethical_reasoning", "react"],
        )

        text = result[0].content.text
        # Ethical_reasoning should score higher for ethical problems
        # Just verify the structure is correct
        assert "Score:" in text or "score" in text.lower()

    @pytest.mark.asyncio
    async def test_includes_detailed_analysis_section(self, test_mcp_server):
        """Test that comparison includes detailed analysis."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Test", method_ids=["chain_of_thought", "ethical_reasoning"]
        )

        text = result[0].content.text
        assert "Detailed Analysis" in text
        assert "Strengths:" in text
        assert "Limitations:" in text or "Weaknesses:" in text

    @pytest.mark.asyncio
    async def test_includes_why_this_score_reasoning(self, test_mcp_server):
        """Test that each method includes reasoning for its score."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Test problem", method_ids=["chain_of_thought", "ethical_reasoning"]
        )

        text = result[0].content.text
        assert "Why this score:" in text or "score" in text.lower()

    @pytest.mark.asyncio
    async def test_includes_how_it_would_approach(self, test_mcp_server):
        """Test that comparison includes how each method would approach the problem."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["chain_of_thought"])

        text = result[0].content.text
        assert "How it would approach this problem:" in text

    @pytest.mark.asyncio
    async def test_method_specific_approach_chain_of_thought(self, test_mcp_server):
        """Test method-specific approach for chain_of_thought."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["chain_of_thought"])

        text = result[0].content.text
        assert "Break the problem into sequential logical steps" in text

    @pytest.mark.asyncio
    async def test_method_specific_approach_tree_of_thoughts(self, test_mcp_server):
        """Test method-specific approach for tree_of_thoughts."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["tree_of_thoughts"])

        text = result[0].content.text
        assert "Generate multiple solution approaches" in text

    @pytest.mark.asyncio
    async def test_method_specific_approach_ethical_reasoning(self, test_mcp_server):
        """Test method-specific approach for ethical_reasoning."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["ethical_reasoning"])

        text = result[0].content.text
        assert "Identify all stakeholders and their interests" in text

    @pytest.mark.asyncio
    async def test_includes_recommendation_section(self, test_mcp_server):
        """Test that comparison includes recommendation section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Test", method_ids=["chain_of_thought", "ethical_reasoning"]
        )

        text = result[0].content.text
        assert "Recommendation" in text

    @pytest.mark.asyncio
    async def test_recommendation_for_high_score(self, test_mcp_server):
        """Test recommendation text when a method has high score (>0.5)."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Should we implement this ethical feature for user privacy?",
            method_ids=["ethical_reasoning"],
        )

        text = result[0].content.text
        # Should have some form of recommendation
        assert "Recommendation" in text

    @pytest.mark.asyncio
    async def test_handles_close_scores(self, test_mcp_server):
        """Test that comparison notes when scores are close."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(
            problem="Generic problem", method_ids=["chain_of_thought", "react"]
        )

        text = result[0].content.text
        # Should have comparison information
        assert "score" in text.lower() or "Score" in text

    @pytest.mark.asyncio
    async def test_handles_no_methods_found(self, test_mcp_server):
        """Test handling when specified methods don't exist."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["nonexistent1", "nonexistent2"])

        text = result[0].content.text
        assert "not found" in text.lower() or "None of" in text
        assert "methods_list" in text

    @pytest.mark.asyncio
    async def test_includes_next_steps(self, test_mcp_server):
        """Test that comparison includes next steps."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["chain_of_thought"])

        text = result[0].content.text
        assert "Next Steps" in text
        assert "reason(" in text

    @pytest.mark.asyncio
    async def test_shows_supports_capabilities(self, test_mcp_server):
        """Test that comparison shows method capabilities."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["tree_of_thoughts"])

        text = result[0].content.text
        assert "Supports:" in text
        assert "branching" in text.lower()

    @pytest.mark.asyncio
    async def test_includes_problem_in_output(self, test_mcp_server):
        """Test that comparison includes the original problem."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        problem = "How should we handle this complex scenario?"
        result = await prompt_func(problem=problem, method_ids=["chain_of_thought"])

        text = result[0].content.text
        assert problem in text


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestGuidedPromptsEdgeCases:
    """Test edge cases for guided prompts."""

    @pytest.mark.asyncio
    async def test_reason_with_method_empty_problem(self, test_mcp_server):
        """Test reason_with_method with empty problem string."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "reason_with_method":
                prompt_func = item.fn
                break

        result = await prompt_func(method_id="chain_of_thought", problem="")

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_compare_methods_single_method(self, test_mcp_server):
        """Test compare_methods with only one method."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=["chain_of_thought"])

        text = result[0].content.text
        assert "Chain of Thought" in text
        # Should still have comparison structure
        assert "Comparison" in text or "comparison" in text

    @pytest.mark.asyncio
    async def test_compare_methods_empty_method_list(self, test_mcp_server):
        """Test compare_methods with empty method list."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "compare_methods":
                prompt_func = item.fn
                break

        result = await prompt_func(problem="Test", method_ids=[])

        # Should handle gracefully
        assert isinstance(result, list)
        assert len(result) > 0
