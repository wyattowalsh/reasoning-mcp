"""
Unit tests for router pipeline templates.

Tests cover:
- Template availability
- Template structure validation
- Template metadata
- Domain-based template lookup
"""

import pytest

from reasoning_mcp.router.templates import (
    get_available_templates,
    get_template,
    get_template_metadata,
    get_templates_for_domain,
)

# ============================================================================
# Template Availability Tests
# ============================================================================


@pytest.mark.unit
class TestTemplateAvailability:
    """Tests for template availability."""

    def test_get_available_templates_returns_list(self):
        """Test get_available_templates returns a list."""
        templates = get_available_templates()
        assert isinstance(templates, list)

    def test_minimum_templates_available(self):
        """Test minimum number of templates are available."""
        templates = get_available_templates()
        assert len(templates) >= 10, "Expected at least 10 templates"

    def test_expected_templates_exist(self):
        """Test expected core templates exist."""
        templates = get_available_templates()

        expected = [
            "verified_reasoning",
            "iterative_improve",
            "analyze_refine",
            "ethical_multi_view",
            "math_proof",
            "debug_code",
            "creative_explore",
            "scientific_method",
            "decompose_solve",
            "multi_agent_debate",
            "deep_research",
            "decision_matrix",
        ]

        for template_id in expected:
            assert template_id in templates, f"Missing template: {template_id}"


# ============================================================================
# Template Structure Tests
# ============================================================================


@pytest.mark.unit
class TestTemplateStructure:
    """Tests for template structure validation."""

    def test_get_template_returns_dict(self):
        """Test get_template returns a dictionary."""
        template = get_template("verified_reasoning")
        assert isinstance(template, dict)

    def test_template_has_name(self):
        """Test templates have a name field."""
        template = get_template("verified_reasoning")
        assert "name" in template
        assert template["name"] == "verified_reasoning"

    def test_template_has_stages(self):
        """Test templates have stages field."""
        template = get_template("verified_reasoning")
        assert "stages" in template
        assert isinstance(template["stages"], list)
        assert len(template["stages"]) > 0

    def test_template_has_metadata(self):
        """Test templates have metadata field."""
        template = get_template("verified_reasoning")
        assert "metadata" in template
        assert isinstance(template["metadata"], dict)

    def test_stage_has_required_fields(self):
        """Test stages have required fields."""
        template = get_template("verified_reasoning")

        for stage in template["stages"]:
            assert "name" in stage
            # Method stages have method_id
            if stage.get("stage_type") == "method":
                assert "method_id" in stage

    def test_unknown_template_returns_none(self):
        """Test unknown template returns None."""
        template = get_template("nonexistent_template")
        assert template is None

    def test_all_templates_are_valid(self):
        """Test all available templates are valid."""
        templates = get_available_templates()

        for template_id in templates:
            template = get_template(template_id)
            assert template is not None, f"Template {template_id} returned None"
            assert isinstance(template, dict), f"Template {template_id} is not a dict"
            assert "name" in template, f"Template {template_id} missing name"
            # Templates can have "stages" (sequence/parallel) or "body" (loop)
            has_stages = "stages" in template
            has_body = "body" in template
            assert has_stages or has_body, f"Template {template_id} missing stages or body"


# ============================================================================
# Template Metadata Tests
# ============================================================================


@pytest.mark.unit
class TestTemplateMetadata:
    """Tests for template metadata."""

    def test_get_template_metadata_returns_dict(self):
        """Test get_template_metadata returns a dictionary."""
        metadata = get_template_metadata("verified_reasoning")
        assert isinstance(metadata, dict)

    def test_metadata_has_template_id(self):
        """Test metadata includes template_id."""
        metadata = get_template_metadata("verified_reasoning")
        assert "template_id" in metadata
        assert metadata["template_id"] == "verified_reasoning"

    def test_metadata_has_best_for(self):
        """Test metadata includes best_for use cases."""
        metadata = get_template_metadata("verified_reasoning")
        assert "best_for" in metadata
        assert isinstance(metadata["best_for"], list)

    def test_metadata_has_description(self):
        """Test metadata includes description."""
        metadata = get_template_metadata("verified_reasoning")
        assert "description" in metadata
        assert len(metadata["description"]) > 0

    def test_unknown_template_metadata_returns_none(self):
        """Test unknown template metadata returns None."""
        metadata = get_template_metadata("nonexistent_template")
        assert metadata is None

    def test_all_templates_have_metadata(self):
        """Test all templates have complete metadata."""
        templates = get_available_templates()

        for template_id in templates:
            metadata = get_template_metadata(template_id)
            assert metadata is not None, f"Template {template_id} missing metadata"
            assert "template_id" in metadata
            assert "best_for" in metadata
            assert "description" in metadata


# ============================================================================
# Domain Template Lookup Tests
# ============================================================================


@pytest.mark.unit
class TestDomainTemplateLookup:
    """Tests for domain-based template lookup."""

    def test_get_templates_for_domain_returns_list(self):
        """Test get_templates_for_domain returns a list."""
        templates = get_templates_for_domain("ethical")
        assert isinstance(templates, list)

    def test_ethical_domain_finds_templates(self):
        """Test ethical domain finds relevant templates."""
        templates = get_templates_for_domain("ethical")
        assert len(templates) > 0
        assert "ethical_multi_view" in templates

    def test_math_domain_finds_templates(self):
        """Test math domain finds relevant templates."""
        templates = get_templates_for_domain("math")
        assert len(templates) > 0

    def test_debug_domain_finds_templates(self):
        """Test debug domain finds relevant templates."""
        templates = get_templates_for_domain("debug")
        assert len(templates) > 0
        assert "debug_code" in templates

    def test_creative_domain_finds_templates(self):
        """Test creative domain finds relevant templates."""
        templates = get_templates_for_domain("creative")
        assert len(templates) > 0
        assert "creative_explore" in templates

    def test_research_domain_finds_templates(self):
        """Test research domain finds relevant templates."""
        templates = get_templates_for_domain("research")
        assert len(templates) > 0

    def test_decision_domain_finds_templates(self):
        """Test decision domain finds relevant templates."""
        templates = get_templates_for_domain("decision")
        assert len(templates) > 0

    def test_case_insensitive_lookup(self):
        """Test domain lookup is case-insensitive."""
        templates_lower = get_templates_for_domain("ethical")
        templates_upper = get_templates_for_domain("ETHICAL")
        templates_mixed = get_templates_for_domain("Ethical")

        assert templates_lower == templates_upper
        assert templates_lower == templates_mixed

    def test_unknown_domain_returns_empty_list(self):
        """Test unknown domain returns empty list."""
        templates = get_templates_for_domain("nonexistent_domain_xyz")
        assert isinstance(templates, list)
        assert len(templates) == 0


# ============================================================================
# Specific Template Tests
# ============================================================================


@pytest.mark.unit
class TestSpecificTemplates:
    """Tests for specific template configurations."""

    def test_verified_reasoning_has_two_stages(self):
        """Test verified_reasoning has two stages."""
        template = get_template("verified_reasoning")
        assert len(template["stages"]) == 2

    def test_ethical_multi_view_has_parallel(self):
        """Test ethical_multi_view includes parallel execution."""
        template = get_template("ethical_multi_view")
        # First stage should be parallel
        assert template["stages"][0]["stage_type"] == "parallel"

    def test_iterative_improve_is_loop(self):
        """Test iterative_improve is a loop pipeline."""
        template = get_template("iterative_improve")
        assert template["stage_type"] == "loop"

    def test_decompose_solve_has_three_stages(self):
        """Test decompose_solve has three stages."""
        template = get_template("decompose_solve")
        assert len(template["stages"]) == 3

    def test_math_proof_has_four_stages(self):
        """Test math_proof has four stages."""
        template = get_template("math_proof")
        assert len(template["stages"]) == 4

    def test_debug_code_has_code_reasoning(self):
        """Test debug_code includes code_reasoning method."""
        template = get_template("debug_code")
        method_ids = [
            stage.get("method_id")
            for stage in template["stages"]
            if stage.get("stage_type") == "method"
        ]
        assert "code_reasoning" in method_ids


# ============================================================================
# Template Caching Tests
# ============================================================================


@pytest.mark.unit
class TestTemplateCaching:
    """Tests for template caching behavior."""

    def test_same_template_returns_same_object(self):
        """Test repeated calls return the same cached template."""
        template1 = get_template("verified_reasoning")
        template2 = get_template("verified_reasoning")

        # Should be the same object (cached)
        assert template1 is template2

    def test_different_templates_return_different_objects(self):
        """Test different templates return different objects."""
        template1 = get_template("verified_reasoning")
        template2 = get_template("analyze_refine")

        assert template1 is not template2
        assert template1["name"] != template2["name"]
