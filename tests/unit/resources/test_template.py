"""
Comprehensive tests for template resource in reasoning_mcp.resources.template.

This module provides complete test coverage for template resources:
- template://{template_id} - Returns predefined pipeline templates

Each resource is tested for:
1. All 5 built-in templates (deep_analysis, debate, verify, debug, brainstorm)
2. Template structure validation
3. Not-found handling with helpful error messages
4. Pipeline stage types (method, sequence, parallel, loop)
5. Metadata and use case information
6. Available templates discovery
"""

import json
from typing import Any

import pytest

from reasoning_mcp.models.core import MethodIdentifier

# ============================================================================
# Mock AppContext (Templates don't need session manager)
# ============================================================================


class MockFastMCP:
    """Mock FastMCP server for testing."""

    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}

    def resource(self, uri_pattern: str):
        """Decorator to register a resource."""

        def decorator(func):
            self.resources[uri_pattern] = func
            return func

        return decorator


# ============================================================================
# Test template://{template_id} Resource - Built-in Templates
# ============================================================================


class TestTemplateResourceBuiltins:
    """Test suite for built-in template resources."""

    @pytest.mark.asyncio
    async def test_template_deep_analysis(self):
        """Test deep_analysis template structure and content."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]
        result = await resource_func(template_id="deep_analysis")

        # Parse JSON
        template = json.loads(result)
        assert isinstance(template, dict)

        # Verify basic structure
        assert template["stage_type"] == "sequence"
        assert template["name"] == "deep_analysis"
        assert "stages" in template
        assert len(template["stages"]) == 3

        # Verify stages
        stages = template["stages"]
        assert stages[0]["stage_type"] == "method"
        assert stages[0]["method_id"] == "chain_of_thought"
        assert stages[0]["name"] == "analyze"
        assert stages[0]["max_thoughts"] == 15

        assert stages[1]["stage_type"] == "method"
        assert stages[1]["method_id"] == "self_reflection"
        assert stages[1]["name"] == "synthesize"
        assert stages[1]["max_thoughts"] == 10

        assert stages[2]["stage_type"] == "method"
        assert stages[2]["method_id"] == "sequential_thinking"
        assert stages[2]["name"] == "conclude"
        assert stages[2]["max_thoughts"] == 5

        # Verify metadata
        assert template["metadata"]["template_id"] == "deep_analysis"
        assert "use_case" in template["metadata"]
        assert template["metadata"]["estimated_thoughts"] == 30

        # Verify available templates list is included
        assert "_available_templates" in template
        assert isinstance(template["_available_templates"], dict)

    @pytest.mark.asyncio
    async def test_template_debate(self):
        """Test debate template structure and content."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]
        result = await resource_func(template_id="debate")

        template = json.loads(result)

        # Verify basic structure
        assert template["stage_type"] == "sequence"
        assert template["name"] == "debate"
        assert len(template["stages"]) == 3

        # Verify dialectic stages
        stages = template["stages"]
        assert stages[0]["method_id"] == "chain_of_thought"
        assert stages[0]["name"] == "thesis"

        assert stages[1]["method_id"] == "dialectic"
        assert stages[1]["name"] == "antithesis"

        assert stages[2]["method_id"] == "self_reflection"
        assert stages[2]["name"] == "synthesis"

        # Verify metadata
        assert template["metadata"]["template_id"] == "debate"
        assert "perspectives" in template["metadata"]["use_case"].lower()

    @pytest.mark.asyncio
    async def test_template_verify(self):
        """Test verify template structure with parallel execution."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]
        result = await resource_func(template_id="verify")

        template = json.loads(result)

        # Verify parallel structure
        assert template["stage_type"] == "parallel"
        assert template["name"] == "verify"
        assert "branches" in template
        assert len(template["branches"]) == 3

        # Verify all branches use chain_of_thought
        for i, branch in enumerate(template["branches"], start=1):
            assert branch["stage_type"] == "method"
            assert branch["method_id"] == "chain_of_thought"
            assert branch["name"] == f"path_{i}"
            assert branch["max_thoughts"] == 10

        # Verify merge strategy
        assert "merge_strategy" in template
        merge = template["merge_strategy"]
        assert merge["name"] == "vote"
        assert merge["selection_criteria"] == "most_common_conclusion"
        assert "min_agreement" in merge["metadata"]

        # Verify parallel execution settings
        assert template["max_concurrency"] == 3
        assert template["timeout_seconds"] == 180.0

        # Verify metadata
        assert template["metadata"]["template_id"] == "verify"
        assert "self-consistency" in template["metadata"]["use_case"].lower()

    @pytest.mark.asyncio
    async def test_template_debug(self):
        """Test debug template with loop structure."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]
        result = await resource_func(template_id="debug")

        template = json.loads(result)

        # Verify sequence structure
        assert template["stage_type"] == "sequence"
        assert template["name"] == "debug"
        assert len(template["stages"]) == 3

        stages = template["stages"]

        # Verify initial analysis
        assert stages[0]["stage_type"] == "method"
        assert stages[0]["method_id"] == "code_reasoning"
        assert stages[0]["name"] == "initial_analysis"

        # Verify loop structure
        assert stages[1]["stage_type"] == "loop"
        assert stages[1]["name"] == "iterative_debugging"
        assert stages[1]["max_iterations"] == 5

        # Verify loop body
        loop_body = stages[1]["body"]
        assert loop_body["stage_type"] == "sequence"
        assert len(loop_body["stages"]) == 2
        assert loop_body["stages"][0]["method_id"] == "react"
        assert loop_body["stages"][1]["method_id"] == "code_reasoning"

        # Verify condition
        condition = stages[1]["condition"]
        assert condition["name"] == "solution_found"
        assert condition["expression"] == "is_solved == False"
        assert condition["field"] == "is_solved"

        # Verify accumulator
        accumulator = stages[1]["accumulator"]
        assert accumulator["name"] == "debug_history"
        assert accumulator["operation"] == "append"
        assert accumulator["field"] == "content"

        # Verify final verification
        assert stages[2]["stage_type"] == "method"
        assert stages[2]["method_id"] == "self_reflection"
        assert stages[2]["name"] == "verify_solution"

        # Verify metadata
        assert template["metadata"]["template_id"] == "debug"
        assert "code" in template["metadata"]["use_case"].lower()

    @pytest.mark.asyncio
    async def test_template_brainstorm(self):
        """Test brainstorm template with parallel creative methods."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]
        result = await resource_func(template_id="brainstorm")

        template = json.loads(result)

        # Verify parallel structure
        assert template["stage_type"] == "parallel"
        assert template["name"] == "brainstorm"
        assert len(template["branches"]) == 4

        # Verify creative methods
        branches = template["branches"]
        assert branches[0]["method_id"] == "lateral_thinking"
        assert branches[0]["name"] == "creative_path"

        assert branches[1]["method_id"] == "analogical"
        assert branches[1]["name"] == "analogy_path"

        assert branches[2]["method_id"] == "tree_of_thoughts"
        assert branches[2]["name"] == "exploration_path"

        assert branches[3]["method_id"] == "step_back"
        assert branches[3]["name"] == "abstraction_path"

        # Verify merge strategy combines all
        merge = template["merge_strategy"]
        assert merge["name"] == "combine"
        assert merge["selection_criteria"] == "all_results"
        assert merge["aggregation"] == "synthesize"

        # Verify parallel execution settings
        assert template["max_concurrency"] == 4
        assert template["timeout_seconds"] == 240.0

        # Verify metadata
        assert template["metadata"]["template_id"] == "brainstorm"
        assert "creative" in template["metadata"]["use_case"].lower()
        assert template["metadata"]["estimated_thoughts"] == 45


# ============================================================================
# Test template://{template_id} Resource - Error Handling
# ============================================================================


class TestTemplateResourceErrors:
    """Test suite for template resource error handling."""

    @pytest.mark.asyncio
    async def test_template_not_found(self):
        """Test that non-existent template raises ValueError."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]

        with pytest.raises(ValueError) as exc_info:
            await resource_func(template_id="non_existent_template")

        error_msg = str(exc_info.value)
        assert "Template 'non_existent_template' not found" in error_msg
        assert "Available templates:" in error_msg

        # Verify error message lists available templates
        assert "deep_analysis" in error_msg
        assert "debate" in error_msg
        assert "verify" in error_msg
        assert "debug" in error_msg
        assert "brainstorm" in error_msg

    @pytest.mark.asyncio
    async def test_template_invalid_id_format(self):
        """Test template with invalid ID format."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]

        with pytest.raises(ValueError) as exc_info:
            await resource_func(template_id="")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_template_case_sensitive(self):
        """Test that template IDs are case-sensitive."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)

        resource_func = mcp.resources["template://{template_id}"]

        # Uppercase should fail
        with pytest.raises(ValueError):
            await resource_func(template_id="DEEP_ANALYSIS")

        # Mixed case should fail
        with pytest.raises(ValueError):
            await resource_func(template_id="Deep_Analysis")

        # Correct lowercase should succeed
        result = await resource_func(template_id="deep_analysis")
        template = json.loads(result)
        assert template["name"] == "deep_analysis"


# ============================================================================
# Test Template Helper Functions
# ============================================================================


class TestTemplateHelpers:
    """Test suite for template helper functions."""

    def test_get_available_templates(self):
        """Test get_available_templates returns all template IDs."""
        from reasoning_mcp.resources.template import get_available_templates

        templates = get_available_templates()

        assert isinstance(templates, list)
        assert len(templates) == 5

        # Verify all expected templates are present
        assert "brainstorm" in templates
        assert "debate" in templates
        assert "debug" in templates
        assert "deep_analysis" in templates
        assert "verify" in templates

        # Verify sorted order
        assert templates == sorted(templates)

    def test_get_template_valid(self):
        """Test get_template returns template for valid ID."""
        from reasoning_mcp.resources.template import get_template

        template = get_template("deep_analysis")

        assert template is not None
        assert isinstance(template, dict)
        assert template["name"] == "deep_analysis"
        assert "stages" in template

    def test_get_template_invalid(self):
        """Test get_template returns None for invalid ID."""
        from reasoning_mcp.resources.template import get_template

        template = get_template("non_existent")

        assert template is None

    def test_get_template_each_builtin(self):
        """Test get_template works for each built-in template."""
        from reasoning_mcp.resources.template import get_template

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]

        for template_id in template_ids:
            template = get_template(template_id)
            assert template is not None
            assert template["metadata"]["template_id"] == template_id


# ============================================================================
# Test Template Structure Validation
# ============================================================================


class TestTemplateStructure:
    """Test suite for template structure validation."""

    @pytest.mark.asyncio
    async def test_all_templates_have_metadata(self):
        """Test that all templates include complete metadata."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]

        for template_id in template_ids:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            # Verify metadata fields
            assert "metadata" in template
            metadata = template["metadata"]
            assert "template_id" in metadata
            assert metadata["template_id"] == template_id
            assert "use_case" in metadata
            assert "estimated_thoughts" in metadata

    @pytest.mark.asyncio
    async def test_all_templates_have_stage_type(self):
        """Test that all templates have valid stage_type."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]
        valid_stage_types = ["sequence", "parallel", "loop", "method"]

        for template_id in template_ids:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            assert "stage_type" in template
            assert template["stage_type"] in valid_stage_types

    @pytest.mark.asyncio
    async def test_sequence_templates_have_stages(self):
        """Test that sequence templates have stages list."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        # Sequence templates
        sequence_templates = ["deep_analysis", "debate", "debug"]

        for template_id in sequence_templates:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            assert template["stage_type"] == "sequence"
            assert "stages" in template
            assert isinstance(template["stages"], list)
            assert len(template["stages"]) > 0

    @pytest.mark.asyncio
    async def test_parallel_templates_have_branches(self):
        """Test that parallel templates have branches list."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        # Parallel templates
        parallel_templates = ["verify", "brainstorm"]

        for template_id in parallel_templates:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            assert template["stage_type"] == "parallel"
            assert "branches" in template
            assert isinstance(template["branches"], list)
            assert len(template["branches"]) > 0

            # Verify merge strategy exists
            assert "merge_strategy" in template
            assert "max_concurrency" in template

    @pytest.mark.asyncio
    async def test_method_stages_have_required_fields(self):
        """Test that method stages have all required fields."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        result = await resource_func(template_id="deep_analysis")
        template = json.loads(result)

        for stage in template["stages"]:
            assert stage["stage_type"] == "method"
            assert "method_id" in stage
            assert "name" in stage
            assert "description" in stage
            assert "max_thoughts" in stage

    @pytest.mark.asyncio
    async def test_available_templates_in_response(self):
        """Test that _available_templates is included in all responses."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]

        for template_id in template_ids:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            # Verify _available_templates exists
            assert "_available_templates" in template
            available = template["_available_templates"]

            # Verify it's a dictionary with all templates
            assert isinstance(available, dict)
            assert len(available) == 5

            # Verify each template has a use case description
            for tid in template_ids:
                assert tid in available
                assert isinstance(available[tid], str)

    @pytest.mark.asyncio
    async def test_json_formatting(self):
        """Test that template JSON is properly formatted."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        result = await resource_func(template_id="deep_analysis")

        # Verify it's valid JSON
        template = json.loads(result)
        assert isinstance(template, dict)

        # Verify it's pretty-printed (has indentation)
        assert "\n" in result
        assert "  " in result


# ============================================================================
# Integration Tests
# ============================================================================


class TestTemplateResourceIntegration:
    """Integration tests for template resource."""

    @pytest.mark.asyncio
    async def test_template_resource_registration(self):
        """Test that template resource is properly registered."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()

        # Before registration
        assert "template://{template_id}" not in mcp.resources

        # Register
        register_template_resources(mcp)

        # After registration
        assert "template://{template_id}" in mcp.resources
        assert callable(mcp.resources["template://{template_id}"])

    @pytest.mark.asyncio
    async def test_all_templates_accessible(self):
        """Test that all built-in templates are accessible via resource."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]

        for template_id in template_ids:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)
            assert template["metadata"]["template_id"] == template_id

    @pytest.mark.asyncio
    async def test_template_consistency(self):
        """Test that templates are consistent across multiple calls."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        # Call twice and compare
        result1 = await resource_func(template_id="deep_analysis")
        result2 = await resource_func(template_id="deep_analysis")

        template1 = json.loads(result1)
        template2 = json.loads(result2)

        # Should be identical (excluding UUIDs which are generated fresh each time)
        assert template1["name"] == template2["name"]
        assert template1["stage_type"] == template2["stage_type"]
        assert template1["metadata"]["template_id"] == template2["metadata"]["template_id"]
        assert len(template1["stages"]) == len(template2["stages"])

        # Verify stages have same structure (excluding IDs)
        for stage1, stage2 in zip(template1["stages"], template2["stages"], strict=True):
            assert stage1["method_id"] == stage2["method_id"]
            assert stage1["name"] == stage2["name"]
            assert stage1["max_thoughts"] == stage2["max_thoughts"]

    @pytest.mark.asyncio
    async def test_template_method_identifiers_valid(self):
        """Test that all method_id values in templates are valid MethodIdentifiers."""
        from reasoning_mcp.resources.template import register_template_resources

        mcp = MockFastMCP()
        register_template_resources(mcp)
        resource_func = mcp.resources["template://{template_id}"]

        # Get all valid method identifiers
        valid_methods = {member.value for member in MethodIdentifier}

        template_ids = ["deep_analysis", "debate", "verify", "debug", "brainstorm"]

        for template_id in template_ids:
            result = await resource_func(template_id=template_id)
            template = json.loads(result)

            # Recursively check all method_id fields
            def check_method_ids(obj):
                if isinstance(obj, dict):
                    if "method_id" in obj:
                        assert obj["method_id"] in valid_methods, (
                            f"Invalid method_id '{obj['method_id']}' in template '{template_id}'"
                        )
                    for value in obj.values():
                        check_method_ids(value)
                elif isinstance(obj, list):
                    for item in obj:
                        check_method_ids(item)

            check_method_ids(template)
