"""
Comprehensive tests for ThoughtNode and ThoughtEdge models in reasoning_mcp.models.thought.

This module provides complete test coverage for:
- ThoughtNode: Immutable nodes representing individual thoughts in the reasoning graph
- ThoughtEdge: Immutable edges representing relationships between thoughts

Each model is tested for:
1. Creation with required and optional fields
2. Default values
3. Immutability (frozen=True)
4. Field validation (types, ranges, constraints)
5. Helper methods (with_child, with_update)
6. JSON serialization/deserialization
7. Enum compatibility
8. Schema generation
"""

import uuid
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtEdge, ThoughtGraph, ThoughtNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_thought_id() -> str:
    """Provide a sample thought ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_edge_id() -> str:
    """Provide a sample edge ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_branch_id() -> str:
    """Provide a sample branch ID for testing."""
    return "branch-123"


@pytest.fixture
def minimal_thought_data() -> dict[str, Any]:
    """
    Provide minimal required data for creating a ThoughtNode.

    Returns:
        Dictionary with only required fields for ThoughtNode creation.
    """
    return {
        "type": ThoughtType.INITIAL,
        "method_id": MethodIdentifier.SEQUENTIAL_THINKING,
        "content": "This is a test thought",
    }


@pytest.fixture
def complete_thought_data(
    sample_thought_id: str,
    sample_branch_id: str,
) -> dict[str, Any]:
    """
    Provide complete data for creating a ThoughtNode with all fields.

    Returns:
        Dictionary with all possible fields for ThoughtNode creation.
    """
    parent_id = str(uuid.uuid4())
    child_id_1 = str(uuid.uuid4())
    child_id_2 = str(uuid.uuid4())

    return {
        "id": sample_thought_id,
        "type": ThoughtType.CONTINUATION,
        "method_id": MethodIdentifier.CHAIN_OF_THOUGHT,
        "content": "This is a complete thought with all fields",
        "summary": "A summary of the thought",
        "evidence": ["Evidence point 1", "Evidence point 2", "Evidence point 3"],
        "parent_id": parent_id,
        "children_ids": [child_id_1, child_id_2],
        "branch_id": sample_branch_id,
        "confidence": 0.85,
        "quality_score": 0.92,
        "is_valid": True,
        "metadata": {"key1": "value1", "key2": 42, "nested": {"inner": "data"}},
        "step_number": 5,
        "depth": 2,
    }


@pytest.fixture
def minimal_edge_data() -> dict[str, Any]:
    """
    Provide minimal required data for creating a ThoughtEdge.

    Returns:
        Dictionary with only required fields for ThoughtEdge creation.
    """
    source_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())

    return {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": "parent_child",
    }


@pytest.fixture
def complete_edge_data(sample_edge_id: str) -> dict[str, Any]:
    """
    Provide complete data for creating a ThoughtEdge with all fields.

    Returns:
        Dictionary with all possible fields for ThoughtEdge creation.
    """
    source_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())

    return {
        "id": sample_edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": "parent_child",
        "weight": 0.75,
        "metadata": {"relation_type": "strong", "confidence": 0.9},
    }


# ============================================================================
# ThoughtNode Tests - Basic Creation
# ============================================================================


class TestThoughtNodeCreation:
    """Test suite for ThoughtNode creation and basic properties."""

    def test_creation_with_minimal_fields(self, minimal_thought_data: dict[str, Any]):
        """Test creating a ThoughtNode with only required fields."""
        thought = ThoughtNode(**minimal_thought_data)

        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SEQUENTIAL_THINKING
        assert thought.content == minimal_thought_data["content"]
        # id is auto-generated
        assert thought.id is not None
        assert len(thought.id) > 0

    def test_creation_with_all_fields(self, complete_thought_data: dict[str, Any]):
        """Test creating a ThoughtNode with all possible fields."""
        thought = ThoughtNode(**complete_thought_data)

        assert thought.id == complete_thought_data["id"]
        assert thought.type == ThoughtType.CONTINUATION
        assert thought.method_id == MethodIdentifier.CHAIN_OF_THOUGHT
        assert thought.content == complete_thought_data["content"]
        assert thought.summary == complete_thought_data["summary"]
        assert thought.evidence == complete_thought_data["evidence"]
        assert thought.parent_id == complete_thought_data["parent_id"]
        assert thought.children_ids == complete_thought_data["children_ids"]
        assert thought.branch_id == complete_thought_data["branch_id"]
        assert thought.confidence == complete_thought_data["confidence"]
        assert thought.quality_score == complete_thought_data["quality_score"]
        assert thought.is_valid == complete_thought_data["is_valid"]
        assert thought.metadata == complete_thought_data["metadata"]
        assert thought.step_number == complete_thought_data["step_number"]
        assert thought.depth == complete_thought_data["depth"]

    def test_default_values(self, minimal_thought_data: dict[str, Any]):
        """Test that default values are set correctly for optional fields."""
        thought = ThoughtNode(**minimal_thought_data)

        # Optional string fields should be None
        assert thought.summary is None
        assert thought.parent_id is None
        assert thought.branch_id is None

        # Empty lists
        assert thought.evidence == []
        assert thought.children_ids == []

        # Default numeric values
        assert thought.confidence == 0.0  # default is 0.0, not None
        assert thought.quality_score is None

        # Default boolean
        assert thought.is_valid is True

        # Empty metadata
        assert thought.metadata == {}

        # Default integers
        assert thought.step_number == 0
        assert thought.depth == 0

        # created_at should be set automatically
        assert isinstance(thought.created_at, datetime)

    def test_is_pydantic_basemodel(self):
        """Test that ThoughtNode is a Pydantic BaseModel."""
        assert issubclass(ThoughtNode, BaseModel)

    def test_created_at_auto_generated(self, minimal_thought_data: dict[str, Any]):
        """Test that created_at is automatically generated if not provided."""
        before = datetime.now()
        thought = ThoughtNode(**minimal_thought_data)
        after = datetime.now()

        # Model uses naive datetime.now()
        assert before <= thought.created_at <= after


# ============================================================================
# ThoughtNode Tests - Immutability
# ============================================================================


class TestThoughtNodeImmutability:
    """Test suite for ThoughtNode immutability (frozen=True)."""

    def test_cannot_modify_id(self, minimal_thought_data: dict[str, Any]):
        """Test that id cannot be modified after creation."""
        thought = ThoughtNode(**minimal_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.id = str(uuid.uuid4())

    def test_cannot_modify_type(self, minimal_thought_data: dict[str, Any]):
        """Test that type cannot be modified after creation."""
        thought = ThoughtNode(**minimal_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.type = ThoughtType.CONCLUSION

    def test_cannot_modify_content(self, minimal_thought_data: dict[str, Any]):
        """Test that content cannot be modified after creation."""
        thought = ThoughtNode(**minimal_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.content = "Modified content"

    def test_cannot_modify_confidence(self, complete_thought_data: dict[str, Any]):
        """Test that confidence cannot be modified after creation."""
        thought = ThoughtNode(**complete_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.confidence = 0.5

    def test_cannot_modify_children_ids(self, complete_thought_data: dict[str, Any]):
        """Test that children_ids cannot be modified after creation."""
        thought = ThoughtNode(**complete_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.children_ids = [str(uuid.uuid4())]

    def test_cannot_modify_metadata(self, complete_thought_data: dict[str, Any]):
        """Test that metadata cannot be modified after creation."""
        thought = ThoughtNode(**complete_thought_data)

        with pytest.raises((ValidationError, AttributeError)):
            thought.metadata = {"new": "data"}


# ============================================================================
# ThoughtNode Tests - Field Validation
# ============================================================================


class TestThoughtNodeValidation:
    """Test suite for ThoughtNode field validation."""

    def test_confidence_validation_valid_range(self, minimal_thought_data: dict[str, Any]):
        """Test that confidence accepts values in valid range [0.0, 1.0]."""
        # Test boundary values
        thought_0 = ThoughtNode(**{**minimal_thought_data, "confidence": 0.0})
        assert thought_0.confidence == 0.0

        thought_1 = ThoughtNode(**{**minimal_thought_data, "confidence": 1.0})
        assert thought_1.confidence == 1.0

        # Test mid-range value
        thought_mid = ThoughtNode(**{**minimal_thought_data, "confidence": 0.5})
        assert thought_mid.confidence == 0.5

    def test_confidence_validation_below_zero(self, minimal_thought_data: dict[str, Any]):
        """Test that confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(**{**minimal_thought_data, "confidence": -0.1})

        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_validation_above_one(self, minimal_thought_data: dict[str, Any]):
        """Test that confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(**{**minimal_thought_data, "confidence": 1.1})

        assert "confidence" in str(exc_info.value).lower()

    def test_quality_score_validation_valid_range(self, minimal_thought_data: dict[str, Any]):
        """Test that quality_score accepts values in valid range [0.0, 1.0]."""
        # Test boundary values
        thought_0 = ThoughtNode(**{**minimal_thought_data, "quality_score": 0.0})
        assert thought_0.quality_score == 0.0

        thought_1 = ThoughtNode(**{**minimal_thought_data, "quality_score": 1.0})
        assert thought_1.quality_score == 1.0

        # Test mid-range value
        thought_mid = ThoughtNode(**{**minimal_thought_data, "quality_score": 0.75})
        assert thought_mid.quality_score == 0.75

    def test_quality_score_validation_below_zero(self, minimal_thought_data: dict[str, Any]):
        """Test that quality_score below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(**{**minimal_thought_data, "quality_score": -0.1})

        assert "quality_score" in str(exc_info.value).lower()

    def test_quality_score_validation_above_one(self, minimal_thought_data: dict[str, Any]):
        """Test that quality_score above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(**{**minimal_thought_data, "quality_score": 1.5})

        assert "quality_score" in str(exc_info.value).lower()

    def test_thought_type_enum_validation(self, minimal_thought_data: dict[str, Any]):
        """Test that ThoughtType enum values work correctly."""
        for thought_type in ThoughtType:
            thought = ThoughtNode(**{**minimal_thought_data, "type": thought_type})
            assert thought.type == thought_type

    def test_method_identifier_enum_validation(self, minimal_thought_data: dict[str, Any]):
        """Test that MethodIdentifier enum values work correctly."""
        # Test a few representative methods
        methods = [
            MethodIdentifier.SEQUENTIAL_THINKING,
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.REACT,
        ]

        for method in methods:
            thought = ThoughtNode(**{**minimal_thought_data, "method_id": method})
            assert thought.method_id == method

    def test_invalid_thought_type_raises_error(self, minimal_thought_data: dict[str, Any]):
        """Test that invalid thought type raises ValidationError."""
        with pytest.raises(ValidationError):
            ThoughtNode(**{**minimal_thought_data, "type": "invalid_type"})

    def test_invalid_method_id_raises_error(self, minimal_thought_data: dict[str, Any]):
        """Test that invalid method_id raises ValidationError."""
        with pytest.raises(ValidationError):
            ThoughtNode(**{**minimal_thought_data, "method_id": "invalid_method"})

    def test_step_number_validation_below_zero(self, minimal_thought_data: dict[str, Any]):
        """Test that step_number below 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ThoughtNode(**{**minimal_thought_data, "step_number": -1})

    def test_depth_validation_below_zero(self, minimal_thought_data: dict[str, Any]):
        """Test that depth below 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ThoughtNode(**{**minimal_thought_data, "depth": -1})

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise ValidationError."""
        # Missing type
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(
                method_id=MethodIdentifier.SEQUENTIAL_THINKING,
                content="Test",
            )
        assert "type" in str(exc_info.value).lower()

        # Missing method_id
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(
                type=ThoughtType.INITIAL,
                content="Test",
            )
        assert "method_id" in str(exc_info.value).lower()

        # Missing content
        with pytest.raises(ValidationError) as exc_info:
            ThoughtNode(
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            )
        assert "content" in str(exc_info.value).lower()


# ============================================================================
# ThoughtNode Tests - Helper Methods
# ============================================================================


class TestThoughtNodeHelperMethods:
    """Test suite for ThoughtNode helper methods (with_child, with_update)."""

    def test_with_child_returns_new_instance(self, minimal_thought_data: dict[str, Any]):
        """Test that with_child() returns a new ThoughtNode instance."""
        original = ThoughtNode(**minimal_thought_data)
        child_id = str(uuid.uuid4())

        updated = original.with_child(child_id)

        # Should be different instances
        assert original is not updated
        assert id(original) != id(updated)

    def test_with_child_adds_child_id(self, minimal_thought_data: dict[str, Any]):
        """Test that with_child() correctly adds child ID to children_ids."""
        original = ThoughtNode(**minimal_thought_data)
        child_id = str(uuid.uuid4())

        updated = original.with_child(child_id)

        # Original should be unchanged
        assert original.children_ids == []

        # Updated should have the child
        assert child_id in updated.children_ids
        assert len(updated.children_ids) == 1

    def test_with_child_preserves_existing_children(self, complete_thought_data: dict[str, Any]):
        """Test that with_child() preserves existing children_ids."""
        original = ThoughtNode(**complete_thought_data)
        original_children_count = len(original.children_ids)
        new_child_id = str(uuid.uuid4())

        updated = original.with_child(new_child_id)

        # Should have original children plus new one
        assert len(updated.children_ids) == original_children_count + 1
        assert new_child_id in updated.children_ids
        # All original children should still be present
        for child_id in original.children_ids:
            assert child_id in updated.children_ids

    def test_with_child_preserves_other_fields(self, complete_thought_data: dict[str, Any]):
        """Test that with_child() preserves all other fields."""
        original = ThoughtNode(**complete_thought_data)
        child_id = str(uuid.uuid4())

        updated = original.with_child(child_id)

        # All fields except children_ids should be identical
        assert updated.id == original.id
        assert updated.type == original.type
        assert updated.method_id == original.method_id
        assert updated.content == original.content
        assert updated.summary == original.summary
        assert updated.evidence == original.evidence
        assert updated.parent_id == original.parent_id
        assert updated.branch_id == original.branch_id
        assert updated.confidence == original.confidence
        assert updated.quality_score == original.quality_score
        assert updated.is_valid == original.is_valid
        assert updated.metadata == original.metadata
        assert updated.step_number == original.step_number
        assert updated.depth == original.depth
        assert updated.created_at == original.created_at

    def test_with_update_returns_new_instance(self, minimal_thought_data: dict[str, Any]):
        """Test that with_update() returns a new ThoughtNode instance."""
        original = ThoughtNode(**minimal_thought_data)

        updated = original.with_update(confidence=0.75)

        # Should be different instances
        assert original is not updated
        assert id(original) != id(updated)

    def test_with_update_modifies_specified_fields(self, minimal_thought_data: dict[str, Any]):
        """Test that with_update() correctly modifies specified fields."""
        original = ThoughtNode(**minimal_thought_data)

        updated = original.with_update(
            confidence=0.85,
            quality_score=0.90,
            summary="Updated summary",
        )

        # Original should be unchanged
        assert original.confidence == 0.0  # default is 0.0
        assert original.quality_score is None
        assert original.summary is None

        # Updated should have new values
        assert updated.confidence == 0.85
        assert updated.quality_score == 0.90
        assert updated.summary == "Updated summary"

    def test_with_update_preserves_other_fields(self, complete_thought_data: dict[str, Any]):
        """Test that with_update() preserves fields not being updated."""
        original = ThoughtNode(**complete_thought_data)

        updated = original.with_update(confidence=0.95)

        # Only confidence should change
        assert updated.confidence == 0.95

        # All other fields should be unchanged
        assert updated.id == original.id
        assert updated.type == original.type
        assert updated.method_id == original.method_id
        assert updated.content == original.content
        assert updated.summary == original.summary
        assert updated.evidence == original.evidence
        assert updated.parent_id == original.parent_id
        assert updated.children_ids == original.children_ids
        assert updated.branch_id == original.branch_id
        assert updated.quality_score == original.quality_score
        assert updated.is_valid == original.is_valid
        assert updated.metadata == original.metadata
        assert updated.step_number == original.step_number
        assert updated.depth == original.depth
        assert updated.created_at == original.created_at

    def test_with_update_multiple_fields(self, minimal_thought_data: dict[str, Any]):
        """Test that with_update() can update multiple fields at once."""
        original = ThoughtNode(**minimal_thought_data)

        updated = original.with_update(
            summary="New summary",
            confidence=0.75,
            quality_score=0.88,
            is_valid=False,
            evidence=["Evidence 1", "Evidence 2"],
            metadata={"key": "value"},
        )

        assert updated.summary == "New summary"
        assert updated.confidence == 0.75
        assert updated.quality_score == 0.88
        assert updated.is_valid is False
        assert updated.evidence == ["Evidence 1", "Evidence 2"]
        assert updated.metadata == {"key": "value"}


# ============================================================================
# ThoughtNode Tests - Serialization
# ============================================================================


class TestThoughtNodeSerialization:
    """Test suite for ThoughtNode JSON serialization/deserialization."""

    def test_json_serialization_minimal(self, minimal_thought_data: dict[str, Any]):
        """Test JSON serialization with minimal fields."""
        thought = ThoughtNode(**minimal_thought_data)

        json_str = thought.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_serialization_complete(self, complete_thought_data: dict[str, Any]):
        """Test JSON serialization with all fields."""
        thought = ThoughtNode(**complete_thought_data)

        json_str = thought.model_dump_json()
        assert isinstance(json_str, str)

        # Should contain key fields
        assert str(complete_thought_data["id"]) in json_str
        assert complete_thought_data["content"] in json_str
        assert complete_thought_data["summary"] in json_str

    def test_json_deserialization_minimal(self, minimal_thought_data: dict[str, Any]):
        """Test JSON deserialization with minimal fields."""
        original = ThoughtNode(**minimal_thought_data)
        json_str = original.model_dump_json()

        # Deserialize back to object
        deserialized = ThoughtNode.model_validate_json(json_str)

        assert deserialized.id == original.id
        assert deserialized.type == original.type
        assert deserialized.method_id == original.method_id
        assert deserialized.content == original.content

    def test_json_deserialization_complete(self, complete_thought_data: dict[str, Any]):
        """Test JSON deserialization with all fields."""
        original = ThoughtNode(**complete_thought_data)
        json_str = original.model_dump_json()

        # Deserialize back to object
        deserialized = ThoughtNode.model_validate_json(json_str)

        assert deserialized.id == original.id
        assert deserialized.type == original.type
        assert deserialized.method_id == original.method_id
        assert deserialized.content == original.content
        assert deserialized.summary == original.summary
        assert deserialized.evidence == original.evidence
        assert deserialized.parent_id == original.parent_id
        assert deserialized.children_ids == original.children_ids
        assert deserialized.branch_id == original.branch_id
        assert deserialized.confidence == original.confidence
        assert deserialized.quality_score == original.quality_score
        assert deserialized.is_valid == original.is_valid
        assert deserialized.metadata == original.metadata
        assert deserialized.step_number == original.step_number
        assert deserialized.depth == original.depth

    def test_model_dump_excludes_none(self, minimal_thought_data: dict[str, Any]):
        """Test that model_dump can exclude None values."""
        thought = ThoughtNode(**minimal_thought_data)

        dumped = thought.model_dump(exclude_none=True)

        # Optional None fields should not be present
        assert "summary" not in dumped
        assert "parent_id" not in dumped
        assert "branch_id" not in dumped
        assert "quality_score" not in dumped
        # confidence has default 0.0, not None, so it will be present
        assert "confidence" in dumped

    def test_roundtrip_serialization(self, complete_thought_data: dict[str, Any]):
        """Test that serialization and deserialization preserves all data."""
        original = ThoughtNode(**complete_thought_data)

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        deserialized = ThoughtNode.model_validate_json(json_str)

        # Serialize again
        json_str_2 = deserialized.model_dump_json()

        # Both JSON strings should be identical
        assert json_str == json_str_2


# ============================================================================
# ThoughtNode Tests - Schema
# ============================================================================


class TestThoughtNodeSchema:
    """Test suite for ThoughtNode JSON schema generation."""

    def test_model_json_schema_generates_valid_schema(self):
        """Test that model_json_schema() produces valid JSON Schema."""
        schema = ThoughtNode.model_json_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_schema_has_all_fields(self):
        """Test that schema contains all expected fields."""
        schema = ThoughtNode.model_json_schema()
        properties = schema["properties"]

        expected_fields = [
            "id",
            "type",
            "method_id",
            "content",
            "summary",
            "evidence",
            "parent_id",
            "children_ids",
            "branch_id",
            "confidence",
            "quality_score",
            "is_valid",
            "created_at",
            "metadata",
            "step_number",
            "depth",
        ]

        for field in expected_fields:
            assert field in properties, f"Field '{field}' missing from schema"

    def test_schema_required_fields(self):
        """Test that schema correctly marks required fields."""
        schema = ThoughtNode.model_json_schema()
        required = schema["required"]

        # Only type, method_id, and content are required (id has default_factory)
        expected_required = ["type", "method_id", "content"]

        for field in expected_required:
            assert field in required, f"Field '{field}' should be required"

    def test_schema_has_field_descriptions(self):
        """Test that schema includes field descriptions."""
        schema = ThoughtNode.model_json_schema()
        properties = schema["properties"]

        # At least some fields should have descriptions
        # (This depends on Field(..., description=...) in the model)
        # We check that descriptions are possible
        for field_name, field_schema in properties.items():
            # If description exists, it should be a string
            if "description" in field_schema:
                assert isinstance(field_schema["description"], str)


# ============================================================================
# ThoughtEdge Tests - Basic Creation
# ============================================================================


class TestThoughtEdgeCreation:
    """Test suite for ThoughtEdge creation and basic properties."""

    def test_creation_with_required_fields(self, minimal_edge_data: dict[str, Any]):
        """Test creating a ThoughtEdge with only required fields."""
        edge = ThoughtEdge(**minimal_edge_data)

        assert edge.source_id == minimal_edge_data["source_id"]
        assert edge.target_id == minimal_edge_data["target_id"]
        assert edge.edge_type == minimal_edge_data["edge_type"]
        # id is auto-generated
        assert edge.id is not None

    def test_creation_with_all_fields(self, complete_edge_data: dict[str, Any]):
        """Test creating a ThoughtEdge with all possible fields."""
        edge = ThoughtEdge(**complete_edge_data)

        assert edge.id == complete_edge_data["id"]
        assert edge.source_id == complete_edge_data["source_id"]
        assert edge.target_id == complete_edge_data["target_id"]
        assert edge.edge_type == complete_edge_data["edge_type"]
        assert edge.weight == complete_edge_data["weight"]
        assert edge.metadata == complete_edge_data["metadata"]

    def test_default_values(self, minimal_edge_data: dict[str, Any]):
        """Test that default values are set correctly for optional fields."""
        edge = ThoughtEdge(**minimal_edge_data)

        # Default weight
        assert edge.weight == 1.0

        # Empty metadata
        assert edge.metadata == {}

        # created_at should be set automatically (naive datetime)
        assert isinstance(edge.created_at, datetime)

    def test_is_pydantic_basemodel(self):
        """Test that ThoughtEdge is a Pydantic BaseModel."""
        assert issubclass(ThoughtEdge, BaseModel)

    def test_created_at_auto_generated(self, minimal_edge_data: dict[str, Any]):
        """Test that created_at is automatically generated if not provided."""
        before = datetime.now()
        edge = ThoughtEdge(**minimal_edge_data)
        after = datetime.now()

        # Model uses naive datetime.now()
        assert before <= edge.created_at <= after


# ============================================================================
# ThoughtEdge Tests - Immutability
# ============================================================================


class TestThoughtEdgeImmutability:
    """Test suite for ThoughtEdge immutability (frozen=True)."""

    def test_cannot_modify_id(self, minimal_edge_data: dict[str, Any]):
        """Test that id cannot be modified after creation."""
        edge = ThoughtEdge(**minimal_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.id = str(uuid.uuid4())

    def test_cannot_modify_source_id(self, minimal_edge_data: dict[str, Any]):
        """Test that source_id cannot be modified after creation."""
        edge = ThoughtEdge(**minimal_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.source_id = str(uuid.uuid4())

    def test_cannot_modify_target_id(self, minimal_edge_data: dict[str, Any]):
        """Test that target_id cannot be modified after creation."""
        edge = ThoughtEdge(**minimal_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.target_id = str(uuid.uuid4())

    def test_cannot_modify_edge_type(self, minimal_edge_data: dict[str, Any]):
        """Test that edge_type cannot be modified after creation."""
        edge = ThoughtEdge(**minimal_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.edge_type = "different_type"

    def test_cannot_modify_weight(self, complete_edge_data: dict[str, Any]):
        """Test that weight cannot be modified after creation."""
        edge = ThoughtEdge(**complete_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.weight = 0.5

    def test_cannot_modify_metadata(self, complete_edge_data: dict[str, Any]):
        """Test that metadata cannot be modified after creation."""
        edge = ThoughtEdge(**complete_edge_data)

        with pytest.raises((ValidationError, AttributeError)):
            edge.metadata = {"new": "data"}


# ============================================================================
# ThoughtEdge Tests - Field Validation
# ============================================================================


class TestThoughtEdgeValidation:
    """Test suite for ThoughtEdge field validation."""

    def test_edge_type_validation_valid_values(self, minimal_edge_data: dict[str, Any]):
        """Test that various edge_type values are accepted."""
        edge_types = [
            "parent_child",
            "sibling",
            "revision",
            "reference",
            "dependency",
        ]

        for edge_type in edge_types:
            edge = ThoughtEdge(**{**minimal_edge_data, "edge_type": edge_type})
            assert edge.edge_type == edge_type

    def test_edge_type_allows_empty_string(self, minimal_edge_data: dict[str, Any]):
        """Test that empty edge_type string is allowed (no validation)."""
        # The model doesn't validate against empty strings
        edge = ThoughtEdge(**{**minimal_edge_data, "edge_type": ""})
        assert edge.edge_type == ""

    def test_weight_validation_valid_range(self, minimal_edge_data: dict[str, Any]):
        """Test that weight accepts positive values."""
        weights = [0.0, 0.5, 1.0, 1.5, 2.0]

        for weight in weights:
            edge = ThoughtEdge(**{**minimal_edge_data, "weight": weight})
            assert edge.weight == weight

    def test_weight_validation_negative_raises_error(self, minimal_edge_data: dict[str, Any]):
        """Test that negative weight raises ValidationError."""
        with pytest.raises(ValidationError):
            ThoughtEdge(**{**minimal_edge_data, "weight": -0.1})

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise ValidationError."""
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        # Missing source_id
        with pytest.raises(ValidationError) as exc_info:
            ThoughtEdge(
                target_id=target_id,
                edge_type="parent_child",
            )
        assert "source_id" in str(exc_info.value).lower()

        # Missing target_id
        with pytest.raises(ValidationError) as exc_info:
            ThoughtEdge(
                source_id=source_id,
                edge_type="parent_child",
            )
        assert "target_id" in str(exc_info.value).lower()

        # Missing edge_type
        with pytest.raises(ValidationError) as exc_info:
            ThoughtEdge(
                source_id=source_id,
                target_id=target_id,
            )
        assert "edge_type" in str(exc_info.value).lower()


# ============================================================================
# ThoughtEdge Tests - Serialization
# ============================================================================


class TestThoughtEdgeSerialization:
    """Test suite for ThoughtEdge JSON serialization/deserialization."""

    def test_json_serialization_minimal(self, minimal_edge_data: dict[str, Any]):
        """Test JSON serialization with minimal fields."""
        edge = ThoughtEdge(**minimal_edge_data)

        json_str = edge.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_serialization_complete(self, complete_edge_data: dict[str, Any]):
        """Test JSON serialization with all fields."""
        edge = ThoughtEdge(**complete_edge_data)

        json_str = edge.model_dump_json()
        assert isinstance(json_str, str)

        # Should contain key fields
        assert str(complete_edge_data["id"]) in json_str
        assert str(complete_edge_data["source_id"]) in json_str
        assert str(complete_edge_data["target_id"]) in json_str

    def test_json_deserialization_minimal(self, minimal_edge_data: dict[str, Any]):
        """Test JSON deserialization with minimal fields."""
        original = ThoughtEdge(**minimal_edge_data)
        json_str = original.model_dump_json()

        # Deserialize back to object
        deserialized = ThoughtEdge.model_validate_json(json_str)

        assert deserialized.id == original.id
        assert deserialized.source_id == original.source_id
        assert deserialized.target_id == original.target_id
        assert deserialized.edge_type == original.edge_type
        assert deserialized.weight == original.weight

    def test_json_deserialization_complete(self, complete_edge_data: dict[str, Any]):
        """Test JSON deserialization with all fields."""
        original = ThoughtEdge(**complete_edge_data)
        json_str = original.model_dump_json()

        # Deserialize back to object
        deserialized = ThoughtEdge.model_validate_json(json_str)

        assert deserialized.id == original.id
        assert deserialized.source_id == original.source_id
        assert deserialized.target_id == original.target_id
        assert deserialized.edge_type == original.edge_type
        assert deserialized.weight == original.weight
        assert deserialized.metadata == original.metadata

    def test_roundtrip_serialization(self, complete_edge_data: dict[str, Any]):
        """Test that serialization and deserialization preserves all data."""
        original = ThoughtEdge(**complete_edge_data)

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        deserialized = ThoughtEdge.model_validate_json(json_str)

        # Serialize again
        json_str_2 = deserialized.model_dump_json()

        # Both JSON strings should be identical
        assert json_str == json_str_2


# ============================================================================
# ThoughtEdge Tests - Schema
# ============================================================================


class TestThoughtEdgeSchema:
    """Test suite for ThoughtEdge JSON schema generation."""

    def test_model_json_schema_generates_valid_schema(self):
        """Test that model_json_schema() produces valid JSON Schema."""
        schema = ThoughtEdge.model_json_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_schema_has_all_fields(self):
        """Test that schema contains all expected fields."""
        schema = ThoughtEdge.model_json_schema()
        properties = schema["properties"]

        expected_fields = [
            "id",
            "source_id",
            "target_id",
            "edge_type",
            "weight",
            "metadata",
            "created_at",
        ]

        for field in expected_fields:
            assert field in properties, f"Field '{field}' missing from schema"

    def test_schema_required_fields(self):
        """Test that schema correctly marks required fields."""
        schema = ThoughtEdge.model_json_schema()
        required = schema["required"]

        # Only source_id, target_id, edge_type are required (id has default_factory)
        expected_required = ["source_id", "target_id", "edge_type"]

        for field in expected_required:
            assert field in required, f"Field '{field}' should be required"


# ============================================================================
# Integration Tests - ThoughtNode and ThoughtEdge
# ============================================================================


class TestThoughtNodeEdgeIntegration:
    """Integration tests between ThoughtNode and ThoughtEdge."""

    def test_thought_node_with_matching_edge(
        self,
        minimal_thought_data: dict[str, Any],
    ):
        """Test creating ThoughtNode and ThoughtEdge with matching IDs."""
        # Create parent and child thoughts
        parent_thought = ThoughtNode(**minimal_thought_data)

        child_id = str(uuid.uuid4())
        child_data = {
            **minimal_thought_data,
            "id": child_id,
            "parent_id": parent_thought.id,
        }
        child_thought = ThoughtNode(**child_data)

        # Update parent with child
        parent_with_child = parent_thought.with_child(child_id)

        # Create edge connecting them
        edge = ThoughtEdge(
            source_id=parent_with_child.id,
            target_id=child_thought.id,
            edge_type="parent_child",
        )

        # Verify relationships
        assert child_thought.id in parent_with_child.children_ids
        assert child_thought.parent_id == parent_with_child.id
        assert edge.source_id == parent_with_child.id
        assert edge.target_id == child_thought.id

    def test_multiple_thought_nodes_with_edges(
        self,
        minimal_thought_data: dict[str, Any],
    ):
        """Test creating multiple ThoughtNodes connected by ThoughtEdges."""
        # Create initial thought
        thought_1 = ThoughtNode(**minimal_thought_data)

        # Create second thought
        thought_2_id = str(uuid.uuid4())
        thought_2 = ThoughtNode(
            **{
                **minimal_thought_data,
                "id": thought_2_id,
                "type": ThoughtType.CONTINUATION,
                "parent_id": thought_1.id,
            }
        )

        # Create third thought
        thought_3_id = str(uuid.uuid4())
        thought_3 = ThoughtNode(
            **{
                **minimal_thought_data,
                "id": thought_3_id,
                "type": ThoughtType.CONCLUSION,
                "parent_id": thought_2.id,
            }
        )

        # Update thoughts with children
        thought_1_updated = thought_1.with_child(thought_2_id)
        thought_2_updated = thought_2.with_child(thought_3_id)

        # Create edges
        edge_1_2 = ThoughtEdge(
            source_id=thought_1_updated.id,
            target_id=thought_2_updated.id,
            edge_type="parent_child",
        )

        edge_2_3 = ThoughtEdge(
            source_id=thought_2_updated.id,
            target_id=thought_3.id,
            edge_type="parent_child",
        )

        # Verify chain
        assert thought_2_id in thought_1_updated.children_ids
        assert thought_3_id in thought_2_updated.children_ids
        assert edge_1_2.source_id == thought_1_updated.id
        assert edge_1_2.target_id == thought_2_updated.id
        assert edge_2_3.source_id == thought_2_updated.id
        assert edge_2_3.target_id == thought_3.id

    def test_branching_thoughts_with_edges(
        self,
        minimal_thought_data: dict[str, Any],
    ):
        """Test creating branching ThoughtNodes with multiple children."""
        # Create parent thought
        parent = ThoughtNode(**minimal_thought_data)

        # Create two child thoughts (branches)
        child_1_id = str(uuid.uuid4())
        child_1 = ThoughtNode(
            **{
                **minimal_thought_data,
                "id": child_1_id,
                "type": ThoughtType.BRANCH,
                "parent_id": parent.id,
                "branch_id": "branch_1",
            }
        )

        child_2_id = str(uuid.uuid4())
        child_2 = ThoughtNode(
            **{
                **minimal_thought_data,
                "id": child_2_id,
                "type": ThoughtType.BRANCH,
                "parent_id": parent.id,
                "branch_id": "branch_2",
            }
        )

        # Update parent with both children
        parent_updated = parent.with_child(child_1_id).with_child(child_2_id)

        # Create edges for both branches
        edge_1 = ThoughtEdge(
            source_id=parent_updated.id,
            target_id=child_1.id,
            edge_type="parent_child",
        )

        edge_2 = ThoughtEdge(
            source_id=parent_updated.id,
            target_id=child_2.id,
            edge_type="parent_child",
        )

        # Verify branching structure
        assert len(parent_updated.children_ids) == 2
        assert child_1_id in parent_updated.children_ids
        assert child_2_id in parent_updated.children_ids
        assert edge_1.source_id == parent_updated.id
        assert edge_2.source_id == parent_updated.id


# ============================================================================
# ThoughtGraph Tests - Creation
# ============================================================================


class TestThoughtGraphCreation:
    """Test suite for ThoughtGraph creation and initialization."""

    def test_create_empty_graph(self):
        """Test creating an empty ThoughtGraph."""
        graph = ThoughtGraph()

        assert graph.id is not None
        assert len(graph.id) > 0
        assert graph.nodes == {}
        assert graph.edges == {}
        assert graph.root_id is None
        assert graph.metadata == {}
        assert isinstance(graph.created_at, datetime)

    def test_create_graph_with_custom_id(self):
        """Test creating a ThoughtGraph with a custom ID."""
        custom_id = str(uuid.uuid4())
        graph = ThoughtGraph(id=custom_id)

        assert graph.id == custom_id

    def test_create_graph_default_values(self):
        """Test that default values are set correctly."""
        graph = ThoughtGraph()

        assert graph.nodes == {}
        assert graph.edges == {}
        assert graph.root_id is None
        assert graph.metadata == {}
        assert isinstance(graph.created_at, datetime)

    def test_create_graph_with_metadata(self):
        """Test creating a ThoughtGraph with custom metadata."""
        metadata = {"project": "test", "version": "1.0", "tags": ["reasoning", "mcp"]}
        graph = ThoughtGraph(metadata=metadata)

        assert graph.metadata == metadata


# ============================================================================
# ThoughtGraph Tests - Properties
# ============================================================================


class TestThoughtGraphProperties:
    """Test suite for ThoughtGraph computed properties."""

    def test_node_count_empty(self):
        """Test node_count property on empty graph."""
        graph = ThoughtGraph()
        assert graph.node_count == 0

    def test_node_count_with_nodes(self, minimal_thought_data: dict[str, Any]):
        """Test node_count property with nodes added."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought_1)
        assert graph.node_count == 1

        graph = graph.add_thought(thought_2)
        assert graph.node_count == 2

    def test_edge_count_empty(self):
        """Test edge_count property on empty graph."""
        graph = ThoughtGraph()
        assert graph.edge_count == 0

    def test_edge_count_with_edges(self, minimal_thought_data: dict[str, Any]):
        """Test edge_count property with edges added."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**{**minimal_thought_data, "parent_id": thought_1.id})

        graph = graph.add_thought(thought_1).add_thought(thought_2)

        # One edge should be auto-created for parent-child relationship
        assert graph.edge_count == 1

    def test_max_depth_empty(self):
        """Test max_depth property on empty graph."""
        graph = ThoughtGraph()
        assert graph.max_depth == 0

    def test_max_depth_single_node(self, minimal_thought_data: dict[str, Any]):
        """Test max_depth property with single node."""
        graph = ThoughtGraph()
        thought = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(thought)

        assert graph.max_depth == 0

    def test_max_depth_linear_chain(self, minimal_thought_data: dict[str, Any]):
        """Test max_depth property with linear chain."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**{**minimal_thought_data, "depth": 0})
        thought_2 = ThoughtNode(**{**minimal_thought_data, "depth": 1, "parent_id": thought_1.id})
        thought_3 = ThoughtNode(**{**minimal_thought_data, "depth": 2, "parent_id": thought_2.id})

        graph = graph.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        assert graph.max_depth == 2

    def test_max_depth_branching(self, minimal_thought_data: dict[str, Any]):
        """Test max_depth property with branching structure."""
        graph = ThoughtGraph()

        root = ThoughtNode(**{**minimal_thought_data, "depth": 0})
        branch_1 = ThoughtNode(**{**minimal_thought_data, "depth": 1, "parent_id": root.id})
        branch_2 = ThoughtNode(**{**minimal_thought_data, "depth": 2, "parent_id": root.id})

        graph = graph.add_thought(root).add_thought(branch_1).add_thought(branch_2)

        assert graph.max_depth == 2

    def test_branch_count_no_branches(self, minimal_thought_data: dict[str, Any]):
        """Test branch_count property with no branches."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought_1).add_thought(thought_2)

        assert graph.branch_count == 0

    def test_branch_count_with_branches(self, minimal_thought_data: dict[str, Any]):
        """Test branch_count property with branches."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_1"})
        thought_2 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_2"})
        thought_3 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_1"})

        graph = graph.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        # Two unique branch_ids
        assert graph.branch_count == 2

    def test_leaf_ids_empty(self):
        """Test leaf_ids property on empty graph."""
        graph = ThoughtGraph()
        assert graph.leaf_ids == []

    def test_leaf_ids_single_node(self, minimal_thought_data: dict[str, Any]):
        """Test leaf_ids property with single node."""
        graph = ThoughtGraph()
        thought = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(thought)

        assert graph.leaf_ids == [thought.id]

    def test_leaf_ids_with_children(self, minimal_thought_data: dict[str, Any]):
        """Test leaf_ids property excludes nodes with children."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child_id = str(uuid.uuid4())
        child = ThoughtNode(**{**minimal_thought_data, "id": child_id, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        # Only child should be a leaf
        assert graph.leaf_ids == [child_id]
        assert parent.id not in graph.leaf_ids


# ============================================================================
# ThoughtGraph Tests - Add Thought
# ============================================================================


class TestThoughtGraphAddThought:
    """Test suite for ThoughtGraph add_thought method."""

    def test_add_first_thought_sets_root(self, minimal_thought_data: dict[str, Any]):
        """Test that adding first thought sets root_id."""
        graph = ThoughtGraph()
        thought = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought)

        assert graph.root_id == thought.id

    def test_add_thought_with_parent(self, minimal_thought_data: dict[str, Any]):
        """Test adding a thought with parent_id."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child = ThoughtNode(**{**minimal_thought_data, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        assert child.id in graph.nodes
        assert child.parent_id == parent.id

    def test_add_thought_creates_edge_to_parent(self, minimal_thought_data: dict[str, Any]):
        """Test that adding thought with parent auto-creates edge."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child = ThoughtNode(**{**minimal_thought_data, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        # Should have one edge from parent to child
        assert graph.edge_count == 1
        edge = list(graph.edges.values())[0]
        assert edge.source_id == parent.id
        assert edge.target_id == child.id

    def test_add_multiple_thoughts(self, minimal_thought_data: dict[str, Any]):
        """Test adding multiple thoughts."""
        graph = ThoughtGraph()

        thoughts = [ThoughtNode(**minimal_thought_data) for _ in range(5)]

        for thought in thoughts:
            graph = graph.add_thought(thought)

        assert graph.node_count == 5
        for thought in thoughts:
            assert thought.id in graph.nodes

    def test_add_thought_returns_self_for_chaining(self, minimal_thought_data: dict[str, Any]):
        """Test that add_thought returns ThoughtGraph for method chaining."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)
        thought_3 = ThoughtNode(**minimal_thought_data)

        result = graph.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        assert isinstance(result, ThoughtGraph)
        assert result.node_count == 3


# ============================================================================
# ThoughtGraph Tests - Add Edge
# ============================================================================


class TestThoughtGraphAddEdge:
    """Test suite for ThoughtGraph add_edge method."""

    def test_add_edge_basic(self, minimal_thought_data: dict[str, Any]):
        """Test adding a basic edge."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought_1).add_thought(thought_2)

        edge = ThoughtEdge(
            source_id=thought_1.id,
            target_id=thought_2.id,
            edge_type="reference"
        )

        graph = graph.add_edge(edge)

        assert edge.id in graph.edges
        assert graph.edge_count == 1

    def test_add_multiple_edges(self, minimal_thought_data: dict[str, Any]):
        """Test adding multiple edges."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)
        thought_3 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        edge_1 = ThoughtEdge(source_id=thought_1.id, target_id=thought_2.id, edge_type="reference")
        edge_2 = ThoughtEdge(source_id=thought_2.id, target_id=thought_3.id, edge_type="reference")

        graph = graph.add_edge(edge_1).add_edge(edge_2)

        assert graph.edge_count == 2
        assert edge_1.id in graph.edges
        assert edge_2.id in graph.edges

    def test_add_edge_returns_self_for_chaining(self, minimal_thought_data: dict[str, Any]):
        """Test that add_edge returns ThoughtGraph for method chaining."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(thought_1).add_thought(thought_2)

        edge = ThoughtEdge(source_id=thought_1.id, target_id=thought_2.id, edge_type="reference")

        result = graph.add_edge(edge)

        assert isinstance(result, ThoughtGraph)


# ============================================================================
# ThoughtGraph Tests - Get Node
# ============================================================================


class TestThoughtGraphGetNode:
    """Test suite for ThoughtGraph get_node method."""

    def test_get_node_exists(self, minimal_thought_data: dict[str, Any]):
        """Test getting an existing node."""
        graph = ThoughtGraph()
        thought = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(thought)

        retrieved = graph.get_node(thought.id)

        assert retrieved is not None
        assert retrieved.id == thought.id

    def test_get_node_not_exists(self):
        """Test getting a non-existent node."""
        graph = ThoughtGraph()

        retrieved = graph.get_node(str(uuid.uuid4()))

        assert retrieved is None

    def test_get_node_empty_graph(self):
        """Test getting node from empty graph."""
        graph = ThoughtGraph()

        retrieved = graph.get_node(str(uuid.uuid4()))

        assert retrieved is None


# ============================================================================
# ThoughtGraph Tests - Get Path
# ============================================================================


class TestThoughtGraphGetPath:
    """Test suite for ThoughtGraph get_path method."""

    def test_get_path_direct_connection(self, minimal_thought_data: dict[str, Any]):
        """Test getting path between directly connected nodes."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child = ThoughtNode(**{**minimal_thought_data, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        path = graph.get_path(parent.id, child.id)

        assert path is not None
        assert path == [parent.id, child.id]

    def test_get_path_multi_hop(self, minimal_thought_data: dict[str, Any]):
        """Test getting path across multiple hops."""
        graph = ThoughtGraph()

        node_1 = ThoughtNode(**{**minimal_thought_data, "depth": 0})
        node_2 = ThoughtNode(**{**minimal_thought_data, "depth": 1, "parent_id": node_1.id})
        node_3 = ThoughtNode(**{**minimal_thought_data, "depth": 2, "parent_id": node_2.id})

        graph = graph.add_thought(node_1).add_thought(node_2).add_thought(node_3)

        path = graph.get_path(node_1.id, node_3.id)

        assert path is not None
        assert path == [node_1.id, node_2.id, node_3.id]

    def test_get_path_no_path_exists(self, minimal_thought_data: dict[str, Any]):
        """Test getting path when no path exists."""
        graph = ThoughtGraph()

        # Create two disconnected nodes
        node_1 = ThoughtNode(**minimal_thought_data)
        node_2 = ThoughtNode(**minimal_thought_data)

        graph = graph.add_thought(node_1).add_thought(node_2)

        path = graph.get_path(node_1.id, node_2.id)

        assert path is None

    def test_get_path_same_node(self, minimal_thought_data: dict[str, Any]):
        """Test getting path from node to itself."""
        graph = ThoughtGraph()

        node = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(node)

        path = graph.get_path(node.id, node.id)

        assert path is not None
        assert path == [node.id]

    def test_get_path_nonexistent_node(self, minimal_thought_data: dict[str, Any]):
        """Test getting path with non-existent node."""
        graph = ThoughtGraph()

        node = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(node)

        path = graph.get_path(node.id, str(uuid.uuid4()))

        assert path is None


# ============================================================================
# ThoughtGraph Tests - Get Ancestors
# ============================================================================


class TestThoughtGraphGetAncestors:
    """Test suite for ThoughtGraph get_ancestors method."""

    def test_get_ancestors_root_node(self, minimal_thought_data: dict[str, Any]):
        """Test getting ancestors of root node."""
        graph = ThoughtGraph()

        root = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(root)

        ancestors = graph.get_ancestors(root.id)

        assert ancestors == []

    def test_get_ancestors_single_parent(self, minimal_thought_data: dict[str, Any]):
        """Test getting ancestors with single parent."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child = ThoughtNode(**{**minimal_thought_data, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        ancestors = graph.get_ancestors(child.id)

        assert ancestors == [parent.id]

    def test_get_ancestors_multiple_generations(self, minimal_thought_data: dict[str, Any]):
        """Test getting ancestors across multiple generations."""
        graph = ThoughtGraph()

        grandparent = ThoughtNode(**{**minimal_thought_data, "depth": 0})
        parent = ThoughtNode(**{**minimal_thought_data, "depth": 1, "parent_id": grandparent.id})
        child = ThoughtNode(**{**minimal_thought_data, "depth": 2, "parent_id": parent.id})

        graph = graph.add_thought(grandparent).add_thought(parent).add_thought(child)

        ancestors = graph.get_ancestors(child.id)

        assert len(ancestors) == 2
        assert parent.id in ancestors
        assert grandparent.id in ancestors

    def test_get_ancestors_nonexistent_node(self):
        """Test getting ancestors of non-existent node."""
        graph = ThoughtGraph()

        ancestors = graph.get_ancestors(str(uuid.uuid4()))

        assert ancestors == []


# ============================================================================
# ThoughtGraph Tests - Get Descendants
# ============================================================================


class TestThoughtGraphGetDescendants:
    """Test suite for ThoughtGraph get_descendants method."""

    def test_get_descendants_leaf_node(self, minimal_thought_data: dict[str, Any]):
        """Test getting descendants of leaf node."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child = ThoughtNode(**{**minimal_thought_data, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        descendants = graph.get_descendants(child.id)

        assert descendants == []

    def test_get_descendants_single_child(self, minimal_thought_data: dict[str, Any]):
        """Test getting descendants with single child."""
        graph = ThoughtGraph()

        parent = ThoughtNode(**minimal_thought_data)
        child_id = str(uuid.uuid4())
        child = ThoughtNode(**{**minimal_thought_data, "id": child_id, "parent_id": parent.id})

        graph = graph.add_thought(parent).add_thought(child)

        descendants = graph.get_descendants(parent.id)

        assert descendants == [child_id]

    def test_get_descendants_multiple_generations(self, minimal_thought_data: dict[str, Any]):
        """Test getting descendants across multiple generations."""
        graph = ThoughtGraph()

        grandparent = ThoughtNode(**{**minimal_thought_data, "depth": 0})
        parent_id = str(uuid.uuid4())
        parent = ThoughtNode(**{**minimal_thought_data, "id": parent_id, "depth": 1, "parent_id": grandparent.id})
        child_id = str(uuid.uuid4())
        child = ThoughtNode(**{**minimal_thought_data, "id": child_id, "depth": 2, "parent_id": parent.id})

        graph = graph.add_thought(grandparent).add_thought(parent).add_thought(child)

        descendants = graph.get_descendants(grandparent.id)

        assert len(descendants) == 2
        assert parent_id in descendants
        assert child_id in descendants

    def test_get_descendants_nonexistent_node(self):
        """Test getting descendants of non-existent node."""
        graph = ThoughtGraph()

        descendants = graph.get_descendants(str(uuid.uuid4()))

        assert descendants == []


# ============================================================================
# ThoughtGraph Tests - Get Branch
# ============================================================================


class TestThoughtGraphGetBranch:
    """Test suite for ThoughtGraph get_branch method."""

    def test_get_branch_exists(self, minimal_thought_data: dict[str, Any]):
        """Test getting existing branch."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_1"})
        thought_2 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_1"})
        thought_3 = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_2"})

        graph = graph.add_thought(thought_1).add_thought(thought_2).add_thought(thought_3)

        branch_nodes = graph.get_branch("branch_1")

        assert len(branch_nodes) == 2
        assert thought_1 in branch_nodes
        assert thought_2 in branch_nodes
        assert thought_3 not in branch_nodes

    def test_get_branch_not_exists(self, minimal_thought_data: dict[str, Any]):
        """Test getting non-existent branch."""
        graph = ThoughtGraph()

        thought = ThoughtNode(**{**minimal_thought_data, "branch_id": "branch_1"})
        graph = graph.add_thought(thought)

        branch_nodes = graph.get_branch("branch_2")

        assert branch_nodes == []

    def test_get_branch_multiple_nodes(self, minimal_thought_data: dict[str, Any]):
        """Test getting branch with multiple nodes."""
        graph = ThoughtGraph()

        thoughts = [
            ThoughtNode(**{**minimal_thought_data, "branch_id": "main_branch"})
            for _ in range(5)
        ]

        for thought in thoughts:
            graph = graph.add_thought(thought)

        branch_nodes = graph.get_branch("main_branch")

        assert len(branch_nodes) == 5
        for thought in thoughts:
            assert thought in branch_nodes


# ============================================================================
# ThoughtGraph Tests - Get Main Path
# ============================================================================


class TestThoughtGraphGetMainPath:
    """Test suite for ThoughtGraph get_main_path method."""

    def test_get_main_path_empty_graph(self):
        """Test getting main path from empty graph."""
        graph = ThoughtGraph()

        path = graph.get_main_path()

        assert path == []

    def test_get_main_path_single_node(self, minimal_thought_data: dict[str, Any]):
        """Test getting main path with single node."""
        graph = ThoughtGraph()

        node = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(node)

        path = graph.get_main_path()

        assert path == [node.id]

    def test_get_main_path_linear_chain(self, minimal_thought_data: dict[str, Any]):
        """Test getting main path in linear chain."""
        graph = ThoughtGraph()

        node_1 = ThoughtNode(**{**minimal_thought_data, "depth": 0, "confidence": 0.8})
        node_2 = ThoughtNode(**{**minimal_thought_data, "depth": 1, "parent_id": node_1.id, "confidence": 0.9})
        node_3 = ThoughtNode(**{**minimal_thought_data, "depth": 2, "parent_id": node_2.id, "confidence": 0.85})

        graph = graph.add_thought(node_1).add_thought(node_2).add_thought(node_3)

        path = graph.get_main_path()

        assert path == [node_1.id, node_2.id, node_3.id]

    def test_get_main_path_branching_follows_confidence(self, minimal_thought_data: dict[str, Any]):
        """Test that main path follows highest confidence when branching."""
        graph = ThoughtGraph()

        root = ThoughtNode(**{**minimal_thought_data, "depth": 0, "confidence": 0.8})

        # Two branches with different confidence
        high_conf_child_id = str(uuid.uuid4())
        high_conf_child = ThoughtNode(**{
            **minimal_thought_data,
            "id": high_conf_child_id,
            "depth": 1,
            "parent_id": root.id,
            "confidence": 0.95
        })

        low_conf_child = ThoughtNode(**{
            **minimal_thought_data,
            "depth": 1,
            "parent_id": root.id,
            "confidence": 0.5
        })

        graph = graph.add_thought(root).add_thought(high_conf_child).add_thought(low_conf_child)

        path = graph.get_main_path()

        # Should follow high confidence path
        assert path == [root.id, high_conf_child_id]


# ============================================================================
# ThoughtGraph Tests - Serialization
# ============================================================================


class TestThoughtGraphSerialization:
    """Test suite for ThoughtGraph JSON serialization/deserialization."""

    def test_json_serialization_empty(self):
        """Test JSON serialization of empty graph."""
        graph = ThoughtGraph()

        json_str = graph.model_dump_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_json_serialization_with_nodes(self, minimal_thought_data: dict[str, Any]):
        """Test JSON serialization of graph with nodes."""
        graph = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**{**minimal_thought_data, "parent_id": thought_1.id})

        graph = graph.add_thought(thought_1).add_thought(thought_2)

        json_str = graph.model_dump_json()

        assert isinstance(json_str, str)
        assert str(thought_1.id) in json_str
        assert str(thought_2.id) in json_str

    def test_json_deserialization(self, minimal_thought_data: dict[str, Any]):
        """Test JSON deserialization."""
        graph = ThoughtGraph()

        thought = ThoughtNode(**minimal_thought_data)
        graph = graph.add_thought(thought)

        json_str = graph.model_dump_json()

        deserialized = ThoughtGraph.model_validate_json(json_str)

        assert deserialized.id == graph.id
        assert deserialized.node_count == graph.node_count
        assert deserialized.edge_count == graph.edge_count
        assert deserialized.root_id == graph.root_id

    def test_roundtrip_serialization(self, minimal_thought_data: dict[str, Any]):
        """Test that serialization and deserialization preserves all data."""
        original = ThoughtGraph()

        thought_1 = ThoughtNode(**minimal_thought_data)
        thought_2 = ThoughtNode(**{**minimal_thought_data, "parent_id": thought_1.id})

        original = original.add_thought(thought_1).add_thought(thought_2)

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        deserialized = ThoughtGraph.model_validate_json(json_str)

        # Serialize again
        json_str_2 = deserialized.model_dump_json()

        # Both JSON strings should be identical
        assert json_str == json_str_2
