"""
Comprehensive tests for pipeline prompts.

This module tests the MCP prompts in reasoning_mcp.prompts.pipelines:
- deep_analysis: Multi-layer deep analysis prompt
- debug_code: Systematic code debugging prompt
- ethical_decision: Multi-framework ethical analysis prompt
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP

from reasoning_mcp.prompts.pipelines import register_pipeline_prompts


@pytest.fixture
def test_mcp_server():
    """Create a test FastMCP server with pipeline prompts registered."""
    mcp = FastMCP("test-pipeline-server")
    register_pipeline_prompts(mcp)
    return mcp


# ============================================================================
# Tests for deep_analysis prompt
# ============================================================================


class TestDeepAnalysisPrompt:
    """Tests for the deep_analysis prompt."""

    def test_returns_string(self, test_mcp_server):
        """Test that deep_analysis returns a string."""
        # Access the prompt function directly
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        assert prompt_func is not None, "deep_analysis prompt not found"

        result = prompt_func(topic="climate change", depth=3)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_topic_in_output(self, test_mcp_server):
        """Test that output includes the topic."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        topic = "artificial intelligence ethics"
        result = prompt_func(topic=topic, depth=3)

        assert topic in result

    def test_default_depth_is_3(self, test_mcp_server):
        """Test that default depth is 3 layers."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test topic")

        # Should have 3 layers
        assert "Layer 1:" in result
        assert "Layer 2:" in result
        assert "Layer 3:" in result
        assert "Layer 4:" not in result

    def test_depth_parameter_controls_layers(self, test_mcp_server):
        """Test that depth parameter controls number of layers."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        # Test depth=1
        result = prompt_func(topic="test", depth=1)
        assert "Layer 1:" in result
        assert "Layer 2:" not in result

        # Test depth=2
        result = prompt_func(topic="test", depth=2)
        assert "Layer 1:" in result
        assert "Layer 2:" in result
        assert "Layer 3:" not in result

        # Test depth=4
        result = prompt_func(topic="test", depth=4)
        assert "Layer 1:" in result
        assert "Layer 2:" in result
        assert "Layer 3:" in result
        assert "Layer 4:" in result
        assert "Layer 5:" not in result

    def test_depth_clamped_to_minimum_1(self, test_mcp_server):
        """Test that depth is clamped to minimum of 1."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=0)
        assert "Layer 1:" in result

        result = prompt_func(topic="test", depth=-5)
        assert "Layer 1:" in result

    def test_depth_clamped_to_maximum_5(self, test_mcp_server):
        """Test that depth is clamped to maximum of 5."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=10)
        assert "Layer 5:" in result
        assert "Layer 6:" not in result

        result = prompt_func(topic="test", depth=100)
        assert "Layer 5:" in result
        assert "Layer 6:" not in result

    def test_layer_1_is_surface_understanding(self, test_mcp_server):
        """Test that Layer 1 is Surface Understanding."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=3)

        assert "Layer 1: Surface Understanding" in result
        assert "Initial comprehension and key points" in result
        assert "primary concepts and definitions" in result

    def test_layer_2_is_structural_analysis(self, test_mcp_server):
        """Test that Layer 2 is Structural Analysis."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=3)

        assert "Layer 2: Structural Analysis" in result
        assert "Patterns, relationships, and underlying principles" in result
        assert "patterns or structures" in result

    def test_layer_3_is_implications(self, test_mcp_server):
        """Test that Layer 3 is Implications."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=3)

        assert "Layer 3: Implications" in result
        assert "Consequences, applications, and broader context" in result
        assert "immediate and long-term consequences" in result

    def test_layer_4_is_synthesis(self, test_mcp_server):
        """Test that Layer 4 is Synthesis."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=4)

        assert "Layer 4: Synthesis" in result
        assert "Integration of insights across all layers" in result
        assert "insights from different layers connect" in result

    def test_layer_5_is_meta_analysis(self, test_mcp_server):
        """Test that Layer 5 is Meta-Analysis."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=5)

        assert "Layer 5: Meta-Analysis" in result
        assert "Analysis of the analysis itself" in result
        assert "assumptions were made" in result

    def test_includes_instructions(self, test_mcp_server):
        """Test that output includes instructions section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=3)

        assert "Instructions:" in result
        assert "Progress through all 3 layers sequentially" in result
        assert "Build upon insights from previous layers" in result

    def test_instructions_reflect_actual_depth(self, test_mcp_server):
        """Test that instructions reference the actual depth used."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test", depth=2)
        assert "all 2 layers" in result

        result = prompt_func(topic="test", depth=5)
        assert "all 5 layers" in result

    def test_has_header_and_structured_format(self, test_mcp_server):
        """Test that output has proper markdown formatting."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="test topic", depth=3)

        assert "# Deep Analysis: test topic" in result
        assert "##" in result  # Layer headers
        assert "**Focus:**" in result
        assert "**Key Questions:**" in result


# ============================================================================
# Tests for debug_code prompt
# ============================================================================


class TestDebugCodePrompt:
    """Tests for the debug_code prompt."""

    def test_returns_string(self, test_mcp_server):
        """Test that debug_code returns a string."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        assert prompt_func is not None, "debug_code prompt not found"

        result = prompt_func(code="def test(): pass", error=None, language=None)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_code_in_output(self, test_mcp_server):
        """Test that output includes the provided code."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        code = "def divide(a, b):\n    return a / b"
        result = prompt_func(code=code)

        assert code in result
        assert "```" in result  # Code should be in code block

    def test_includes_error_when_provided(self, test_mcp_server):
        """Test that output includes error message when provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        error = "ZeroDivisionError: division by zero"
        result = prompt_func(code="def test(): pass", error=error)

        assert error in result
        assert "Error/Issue:" in result

    def test_handles_no_error_provided(self, test_mcp_server):
        """Test that output handles case when no error is provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="def test(): pass", error=None)

        assert "Issue:" in result
        assert "not working as expected" in result

    def test_includes_language_in_header_when_provided(self, test_mcp_server):
        """Test that language is shown in header when provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="def test(): pass", language="python")

        assert "(python)" in result

    def test_language_used_in_code_block(self, test_mcp_server):
        """Test that language is used in code block formatting."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="const x = 5;", language="javascript")

        assert "```javascript" in result

    def test_handles_no_language_provided(self, test_mcp_server):
        """Test that output handles case when no language is provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="some code", language=None)

        # Should not have language in header
        assert "# Code Debugging Workflow" in result
        # Code block should be generic
        assert "```\nsome code\n```" in result

    def test_has_all_six_debugging_steps(self, test_mcp_server):
        """Test that output includes all 6 debugging steps."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Step 1: Reproduce" in result
        assert "Step 2: Isolate" in result
        assert "Step 3: Hypothesize" in result
        assert "Step 4: Test" in result
        assert "Step 5: Fix" in result
        assert "Step 6: Verify" in result

    def test_step_1_reproduce_has_correct_objective(self, test_mcp_server):
        """Test that Step 1 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Verify and document the error" in result
        assert "Confirm the error occurs consistently" in result

    def test_step_2_isolate_has_correct_objective(self, test_mcp_server):
        """Test that Step 2 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Narrow down the problematic code section" in result
        assert "minimal reproducible example" in result

    def test_step_3_hypothesize_has_correct_objective(self, test_mcp_server):
        """Test that Step 3 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Form theories about the root cause" in result
        assert "what could be wrong" in result

    def test_step_4_test_has_correct_objective(self, test_mcp_server):
        """Test that Step 4 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Validate hypotheses through experimentation" in result
        assert "how would you test it" in result

    def test_step_5_fix_has_correct_objective(self, test_mcp_server):
        """Test that Step 5 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Implement the solution" in result
        assert "Implement the fix" in result
        assert "addresses the root cause" in result

    def test_step_6_verify_has_correct_objective(self, test_mcp_server):
        """Test that Step 6 has correct objective and guidance."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Objective:** Ensure the fix works completely" in result
        assert "Test with the original failing case" in result
        assert "no new issues were introduced" in result

    def test_includes_instructions(self, test_mcp_server):
        """Test that output includes instructions section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="test")

        assert "Instructions:" in result
        assert "Work through each step systematically" in result
        assert "Show the corrected code in the Fix step" in result


# ============================================================================
# Tests for ethical_decision prompt
# ============================================================================


class TestEthicalDecisionPrompt:
    """Tests for the ethical_decision prompt."""

    def test_returns_string(self, test_mcp_server):
        """Test that ethical_decision returns a string."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        assert prompt_func is not None, "ethical_decision prompt not found"

        result = prompt_func(scenario="Should we implement this feature?", stakeholders=None)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_scenario_in_output(self, test_mcp_server):
        """Test that output includes the scenario."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        scenario = "Should we deploy AI for resume screening?"
        result = prompt_func(scenario=scenario)

        assert scenario in result

    def test_includes_stakeholders_when_provided(self, test_mcp_server):
        """Test that stakeholders are included when provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        stakeholders = ["job applicants", "hiring managers", "company"]
        result = prompt_func(scenario="Test scenario", stakeholders=stakeholders)

        assert "Key Stakeholders:" in result
        for stakeholder in stakeholders:
            assert stakeholder in result

    def test_handles_no_stakeholders_provided(self, test_mcp_server):
        """Test that output handles case when no stakeholders provided."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="Test scenario", stakeholders=None)

        # Should not have stakeholders section
        assert "Key Stakeholders:" not in result

    def test_handles_empty_stakeholders_list(self, test_mcp_server):
        """Test that output handles empty stakeholders list."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="Test scenario", stakeholders=[])

        # Should not have stakeholders section for empty list
        assert "Key Stakeholders:" not in result

    def test_has_all_four_ethical_frameworks(self, test_mcp_server):
        """Test that output includes all 4 ethical frameworks."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "1. Consequentialist Analysis" in result
        assert "2. Deontological Analysis" in result
        assert "3. Virtue Ethics Analysis" in result
        assert "4. Care Ethics Analysis" in result

    def test_consequentialist_framework_has_correct_description(self, test_mcp_server):
        """Test Consequentialist framework description and questions."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Consequentialist Analysis" in result
        assert "Focus on outcomes and consequences" in result
        assert "likely consequences of each possible action" in result
        assert "Who benefits and who is harmed" in result
        assert "net utility" in result

    def test_deontological_framework_has_correct_description(self, test_mcp_server):
        """Test Deontological framework description and questions."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Deontological Analysis" in result
        assert "Focus on duties, rules, and principles" in result
        assert "moral duties or obligations" in result
        assert "universal principles" in result
        assert "categorical imperatives" in result

    def test_virtue_ethics_framework_has_correct_description(self, test_mcp_server):
        """Test Virtue Ethics framework description and questions."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Virtue Ethics Analysis" in result
        assert "Focus on character and virtues" in result
        assert "virtuous person do" in result
        assert "courage, honesty, compassion" in result
        assert "practical wisdom (phronesis)" in result

    def test_care_ethics_framework_has_correct_description(self, test_mcp_server):
        """Test Care Ethics framework description and questions."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Care Ethics Analysis" in result
        assert "Focus on relationships and compassion" in result
        assert "relationships affected" in result
        assert "care and compassion demand" in result
        assert "power dynamics and vulnerabilities" in result

    def test_includes_synthesis_section(self, test_mcp_server):
        """Test that output includes synthesis and recommendation section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Synthesis and Recommendation" in result
        assert "Key Tensions:" in result
        assert "Weighing Considerations:" in result
        assert "Recommended Action:" in result
        assert "Justification:" in result
        assert "Limitations:" in result

    def test_synthesis_mentions_framework_agreement(self, test_mcp_server):
        """Test that synthesis section asks about framework agreement."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "frameworks agree or conflict" in result

    def test_includes_instructions(self, test_mcp_server):
        """Test that output includes instructions section."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test")

        assert "Instructions:" in result
        assert "Provide substantive analysis for each framework" in result
        assert "Acknowledge complexity and competing values" in result
        assert "Consider how stakeholders are affected differently" in result

    def test_has_proper_markdown_formatting(self, test_mcp_server):
        """Test that output has proper markdown formatting."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        result = prompt_func(scenario="test scenario")

        assert "# Ethical Decision Analysis" in result
        assert "##" in result  # Section headers
        assert "###" in result  # Framework headers
        assert "**Scenario:**" in result
        assert "**Perspective:**" in result


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestPipelinePromptsEdgeCases:
    """Test edge cases for pipeline prompts."""

    def test_deep_analysis_with_empty_topic(self, test_mcp_server):
        """Test deep_analysis with empty topic string."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "deep_analysis":
                prompt_func = item.fn
                break

        result = prompt_func(topic="", depth=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_debug_code_with_empty_code(self, test_mcp_server):
        """Test debug_code with empty code string."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        result = prompt_func(code="")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_debug_code_with_multiline_code(self, test_mcp_server):
        """Test debug_code with multiline code."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "debug_code":
                prompt_func = item.fn
                break

        code = """def complex_function(x, y):
    if x > 0:
        return y / x
    else:
        return 0"""

        result = prompt_func(code=code, language="python")
        assert code in result
        assert "```python" in result

    def test_ethical_decision_with_long_scenario(self, test_mcp_server):
        """Test ethical_decision with long scenario text."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        scenario = "This is a very long ethical scenario. " * 50
        result = prompt_func(scenario=scenario)
        assert scenario in result

    def test_ethical_decision_with_many_stakeholders(self, test_mcp_server):
        """Test ethical_decision with many stakeholders."""
        prompt_func = None
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn") and item.fn.__name__ == "ethical_decision":
                prompt_func = item.fn
                break

        stakeholders = [
            "users",
            "developers",
            "management",
            "investors",
            "regulators",
            "society",
            "environment",
            "future generations",
        ]
        result = prompt_func(scenario="test", stakeholders=stakeholders)

        for stakeholder in stakeholders:
            assert stakeholder in result

    def test_all_prompts_registered(self, test_mcp_server):
        """Test that all three prompts are registered."""
        prompt_names = set()
        for item in test_mcp_server._prompt_manager._prompts.values():
            if hasattr(item, "fn"):
                prompt_names.add(item.fn.__name__)

        assert "deep_analysis" in prompt_names
        assert "debug_code" in prompt_names
        assert "ethical_decision" in prompt_names
