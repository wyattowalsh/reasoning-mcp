"""Unit tests for method resources.

This module provides comprehensive test coverage for method resources:
- method://{method_id} - Returns method metadata as JSON
- method://{method_id}/schema - Returns JSON schema for method inputs

Tests cover:
- Basic functionality (correct data structure)
- Method ID validation
- Not-found handling
- All metadata fields present and correct types
- Schema structure (JSON Schema Draft 7 format)
- Different method types (with/without context requirement)
- Various method categories
- Error handling for uninitialized registry
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.resources.method import register_method_resources
from reasoning_mcp.server import AppContext

# ============================================================================
# Mock Classes and Fixtures
# ============================================================================


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


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Create a test registry populated with mock methods."""
    registry = MethodRegistry()

    # Register a basic method (no context required)
    cot_method = MockReasoningMethod(
        identifier="chain_of_thought",
        name="Chain of Thought",
        description="Step-by-step sequential reasoning through problems",
        category="core",
    )
    cot_metadata = MethodMetadata(
        identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
        name="Chain of Thought",
        description="Step-by-step sequential reasoning through problems",
        category=MethodCategory.CORE,
        tags=frozenset({"sequential", "structured", "reasoning"}),
        complexity=5,
        supports_branching=False,
        supports_revision=True,
        requires_context=False,
        min_thoughts=3,
        max_thoughts=10,
        avg_tokens_per_thought=500,
        best_for=("logical problems", "mathematics", "step-by-step analysis"),
        not_recommended_for=("creative brainstorming", "open-ended exploration"),
    )
    registry.register(cot_method, cot_metadata)

    # Register a method requiring context
    ethical_method = MockReasoningMethod(
        identifier="ethical_reasoning",
        name="Ethical Reasoning",
        description="Structured ethical analysis with stakeholder consideration",
        category="high_value",
    )
    ethical_metadata = MethodMetadata(
        identifier=MethodIdentifier.ETHICAL_REASONING,
        name="Ethical Reasoning",
        description="Structured ethical analysis with stakeholder consideration",
        category=MethodCategory.HIGH_VALUE,
        tags=frozenset({"ethical", "structured", "stakeholders"}),
        complexity=8,
        supports_branching=True,
        supports_revision=True,
        requires_context=True,  # This method requires context
        min_thoughts=5,
        max_thoughts=25,
        avg_tokens_per_thought=700,
        best_for=("ethical dilemmas", "stakeholder analysis", "moral reasoning"),
        not_recommended_for=("technical problems", "mathematical proofs"),
    )
    registry.register(ethical_method, ethical_metadata)

    # Register a specialized method
    socratic_method = MockReasoningMethod(
        identifier="socratic",
        name="Socratic Method",
        description="Question-driven reasoning and exploration",
        category="specialized",
    )
    socratic_metadata = MethodMetadata(
        identifier=MethodIdentifier.SOCRATIC,
        name="Socratic Method",
        description="Question-driven reasoning and exploration",
        category=MethodCategory.SPECIALIZED,
        tags=frozenset({"questioning", "philosophical", "exploration"}),
        complexity=6,
        supports_branching=False,
        supports_revision=True,
        requires_context=False,
        min_thoughts=4,
        max_thoughts=20,
        avg_tokens_per_thought=550,
        best_for=("understanding concepts", "philosophical questions", "learning"),
        not_recommended_for=("time-sensitive decisions", "numerical calculations"),
    )
    registry.register(socratic_method, socratic_metadata)

    return registry


@pytest.fixture
def mock_mcp_server(mock_registry: MethodRegistry) -> MagicMock:
    """Create a mock FastMCP server with context."""
    mcp = MagicMock()

    # Create mock context with initialized registry
    ctx = AppContext(
        registry=mock_registry,
        session_manager=MagicMock(),
        settings=MagicMock(),
        initialized=True,
    )

    # Make get_context return our context
    mcp.get_context.return_value = ctx

    # Store resource handlers
    mcp._resource_handlers = {}

    # Mock the resource decorator
    def mock_resource(uri_pattern: str):
        def decorator(func):
            mcp._resource_handlers[uri_pattern] = func
            return func

        return decorator

    mcp.resource = mock_resource

    return mcp


@pytest.fixture(autouse=True)
def patch_get_app_context(mock_mcp_server: MagicMock):
    """Autouse fixture to patch get_app_context for all tests.

    This ensures that resource handlers calling get_app_context() from
    reasoning_mcp.server receive the mock context set up by mock_mcp_server.
    """
    ctx = mock_mcp_server.get_context()
    with patch("reasoning_mcp.server.get_app_context", return_value=ctx):
        yield ctx


# ============================================================================
# Tests for get_method_metadata resource
# ============================================================================


class TestGetMethodMetadata:
    """Tests for method://{method_id} resource."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_basic_method(self, mock_mcp_server: MagicMock):
        """Test retrieving metadata for a basic method."""
        # Register resources
        register_method_resources(mock_mcp_server)

        # Get the handler
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        # Call with chain_of_thought
        result = await handler("chain_of_thought")

        # Parse JSON result
        metadata = json.loads(result)

        # Verify structure
        assert isinstance(metadata, dict)

        # Verify all required fields are present
        assert "identifier" in metadata
        assert "name" in metadata
        assert "description" in metadata
        assert "category" in metadata
        assert "complexity" in metadata
        assert "supports_branching" in metadata
        assert "supports_revision" in metadata
        assert "requires_context" in metadata
        assert "min_thoughts" in metadata
        assert "max_thoughts" in metadata
        assert "avg_tokens_per_thought" in metadata
        assert "tags" in metadata
        assert "best_for" in metadata
        assert "not_recommended_for" in metadata

        # Verify field values
        assert metadata["identifier"] == "chain_of_thought"
        assert metadata["name"] == "Chain of Thought"
        assert metadata["description"] == "Step-by-step sequential reasoning through problems"
        assert metadata["category"] == "core"
        assert metadata["complexity"] == 5
        assert metadata["supports_branching"] is False
        assert metadata["supports_revision"] is True
        assert metadata["requires_context"] is False
        assert metadata["min_thoughts"] == 3
        assert metadata["max_thoughts"] == 10
        assert metadata["avg_tokens_per_thought"] == 500

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_all_field_types(self, mock_mcp_server: MagicMock):
        """Test that all metadata fields have correct types."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")
        metadata = json.loads(result)

        # Verify types
        assert isinstance(metadata["identifier"], str)
        assert isinstance(metadata["name"], str)
        assert isinstance(metadata["description"], str)
        assert isinstance(metadata["category"], str)
        assert isinstance(metadata["complexity"], int)
        assert isinstance(metadata["supports_branching"], bool)
        assert isinstance(metadata["supports_revision"], bool)
        assert isinstance(metadata["requires_context"], bool)
        assert isinstance(metadata["min_thoughts"], int)
        assert isinstance(metadata["max_thoughts"], int)
        assert isinstance(metadata["avg_tokens_per_thought"], int)
        assert isinstance(metadata["tags"], list)
        assert isinstance(metadata["best_for"], list)
        assert isinstance(metadata["not_recommended_for"], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_tags_are_sorted(self, mock_mcp_server: MagicMock):
        """Test that tags are returned in sorted order."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")
        metadata = json.loads(result)

        tags = metadata["tags"]
        assert tags == sorted(tags)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_high_value_category(self, mock_mcp_server: MagicMock):
        """Test retrieving metadata for high_value category method."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("ethical_reasoning")
        metadata = json.loads(result)

        assert metadata["identifier"] == "ethical_reasoning"
        assert metadata["name"] == "Ethical Reasoning"
        assert metadata["category"] == "high_value"
        assert metadata["complexity"] == 8
        assert metadata["supports_branching"] is True
        assert metadata["requires_context"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_specialized_category(self, mock_mcp_server: MagicMock):
        """Test retrieving metadata for specialized category method."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("socratic")
        metadata = json.loads(result)

        assert metadata["identifier"] == "socratic"
        assert metadata["name"] == "Socratic Method"
        assert metadata["category"] == "specialized"
        assert metadata["complexity"] == 6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_method_not_found(self, mock_mcp_server: MagicMock):
        """Test that ValueError is raised for non-existent method."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        with pytest.raises(ValueError, match="not found in registry"):
            await handler("nonexistent_method")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_returns_valid_json(self, mock_mcp_server: MagicMock):
        """Test that result is valid JSON string."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")

        # Should be a string
        assert isinstance(result, str)

        # Should parse as JSON without error
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_json_is_formatted(self, mock_mcp_server: MagicMock):
        """Test that JSON is formatted with indentation."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")

        # Should contain newlines and spaces (formatted)
        assert "\n" in result
        assert "  " in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_with_context_required(self, mock_mcp_server: MagicMock):
        """Test metadata for method that requires context."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("ethical_reasoning")
        metadata = json.loads(result)

        assert metadata["requires_context"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_without_context_required(self, mock_mcp_server: MagicMock):
        """Test metadata for method that doesn't require context."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")
        metadata = json.loads(result)

        assert metadata["requires_context"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_with_branching_support(self, mock_mcp_server: MagicMock):
        """Test metadata for method that supports branching."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("ethical_reasoning")
        metadata = json.loads(result)

        assert metadata["supports_branching"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_metadata_without_branching_support(self, mock_mcp_server: MagicMock):
        """Test metadata for method that doesn't support branching."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        result = await handler("chain_of_thought")
        metadata = json.loads(result)

        assert metadata["supports_branching"] is False


# ============================================================================
# Tests for get_method_schema resource
# ============================================================================


class TestGetMethodSchema:
    """Tests for method://{method_id}/schema resource."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_basic_structure(self, mock_mcp_server: MagicMock):
        """Test basic schema structure."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        # Verify it's a valid JSON Schema
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["type"] == "object"
        assert "title" in schema
        assert "description" in schema
        assert "properties" in schema
        assert "required" in schema
        assert "additionalProperties" in schema

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_has_input_text_property(self, mock_mcp_server: MagicMock):
        """Test that schema includes input_text property."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        # Check input_text property
        assert "input_text" in schema["properties"]
        input_text_prop = schema["properties"]["input_text"]
        assert input_text_prop["type"] == "string"
        assert input_text_prop["minLength"] == 1
        assert "description" in input_text_prop

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_has_context_property(self, mock_mcp_server: MagicMock):
        """Test that schema includes context property."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        # Check context property
        assert "context" in schema["properties"]
        context_prop = schema["properties"]["context"]
        assert context_prop["type"] == "object"
        assert context_prop["additionalProperties"] is True
        assert "description" in context_prop

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_input_text_required_by_default(self, mock_mcp_server: MagicMock):
        """Test that input_text is always required."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        assert "input_text" in schema["required"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_context_not_required_by_default(self, mock_mcp_server: MagicMock):
        """Test that context is optional by default."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        # Context should not be in required for methods without requires_context
        assert "context" not in schema["required"] or schema["required"] == ["input_text"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_context_required_when_method_requires_it(
        self, mock_mcp_server: MagicMock
    ):
        """Test that context is required when method.requires_context is True."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("ethical_reasoning")
        schema = json.loads(result)

        # Context should be required for ethical_reasoning
        assert "context" in schema["required"]
        assert "input_text" in schema["required"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_context_description_changes_when_required(
        self, mock_mcp_server: MagicMock
    ):
        """Test that context description changes when it's required."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        # Method without required context
        result1 = await handler("chain_of_thought")
        schema1 = json.loads(result1)

        # Method with required context
        result2 = await handler("ethical_reasoning")
        schema2 = json.loads(result2)

        # Descriptions should be different
        desc1 = schema1["properties"]["context"]["description"]
        desc2 = schema2["properties"]["context"]["description"]

        assert "optional" in desc1.lower() or "Required" not in desc1
        assert "Required" in desc2 or "required" in desc2.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_title_includes_method_name(self, mock_mcp_server: MagicMock):
        """Test that schema title includes method name."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        assert "Chain of Thought" in schema["title"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_description_includes_method_name(self, mock_mcp_server: MagicMock):
        """Test that schema description includes method name."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("socratic")
        schema = json.loads(result)

        assert "Socratic Method" in schema["description"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_additional_properties_false(self, mock_mcp_server: MagicMock):
        """Test that additionalProperties is false at top level."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        assert schema["additionalProperties"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_method_not_found(self, mock_mcp_server: MagicMock):
        """Test that ValueError is raised for non-existent method."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        with pytest.raises(ValueError, match="not found in registry"):
            await handler("nonexistent_method")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_returns_valid_json(self, mock_mcp_server: MagicMock):
        """Test that result is valid JSON string."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")

        # Should be a string
        assert isinstance(result, str)

        # Should parse as JSON without error
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_json_is_formatted(self, mock_mcp_server: MagicMock):
        """Test that JSON is formatted with indentation."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")

        # Should contain newlines and spaces (formatted)
        assert "\n" in result
        assert "  " in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_schema_draft_7_compliant(self, mock_mcp_server: MagicMock):
        """Test that schema follows JSON Schema Draft 7 format."""
        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        result = await handler("chain_of_thought")
        schema = json.loads(result)

        # Draft 7 specific requirements
        assert "$schema" in schema
        assert "http://json-schema.org/draft-07/schema#" in schema["$schema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert isinstance(schema["required"], list)


# ============================================================================
# Tests for error handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in method resources."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metadata_uninitialized_registry(self, mock_mcp_server: MagicMock):
        """Test that RuntimeError is raised when registry not initialized."""
        # Create uninitialized context
        ctx = AppContext(
            registry=MethodRegistry(),
            session_manager=MagicMock(),
            settings=MagicMock(),
            initialized=False,  # Not initialized
        )

        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        # Patch get_app_context to return the uninitialized context
        with patch("reasoning_mcp.server.get_app_context", return_value=ctx):
            with pytest.raises(RuntimeError, match="not initialized"):
                await handler("chain_of_thought")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_schema_uninitialized_registry(self, mock_mcp_server: MagicMock):
        """Test that RuntimeError is raised when registry not initialized."""
        # Create uninitialized context
        ctx = AppContext(
            registry=MethodRegistry(),
            session_manager=MagicMock(),
            settings=MagicMock(),
            initialized=False,  # Not initialized
        )

        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        # Patch get_app_context to return the uninitialized context
        with patch("reasoning_mcp.server.get_app_context", return_value=ctx):
            with pytest.raises(RuntimeError, match="not initialized"):
                await handler("chain_of_thought")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metadata_empty_registry(self, mock_mcp_server: MagicMock):
        """Test getting metadata from empty registry raises ValueError."""
        # Create empty but initialized context
        ctx = AppContext(
            registry=MethodRegistry(),
            session_manager=MagicMock(),
            settings=MagicMock(),
            initialized=True,
        )

        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        # Patch get_app_context to return the empty registry context
        with patch("reasoning_mcp.server.get_app_context", return_value=ctx):
            with pytest.raises(ValueError, match="not found"):
                await handler("chain_of_thought")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_schema_empty_registry(self, mock_mcp_server: MagicMock):
        """Test getting schema from empty registry raises ValueError."""
        # Create empty but initialized context
        ctx = AppContext(
            registry=MethodRegistry(),
            session_manager=MagicMock(),
            settings=MagicMock(),
            initialized=True,
        )

        register_method_resources(mock_mcp_server)
        handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        # Patch get_app_context to return the empty registry context
        with patch("reasoning_mcp.server.get_app_context", return_value=ctx):
            with pytest.raises(ValueError, match="not found"):
                await handler("chain_of_thought")


# ============================================================================
# Tests for resource registration
# ============================================================================


class TestResourceRegistration:
    """Tests for resource registration."""

    @pytest.mark.unit
    def test_resources_are_registered(self, mock_mcp_server: MagicMock):
        """Test that both resources are registered with correct URI patterns."""
        register_method_resources(mock_mcp_server)

        # Check both resources were registered
        assert "method://{method_id}" in mock_mcp_server._resource_handlers
        assert "method://{method_id}/schema" in mock_mcp_server._resource_handlers

    @pytest.mark.unit
    def test_resource_handlers_are_async(self, mock_mcp_server: MagicMock):
        """Test that resource handlers are async functions."""
        register_method_resources(mock_mcp_server)

        metadata_handler = mock_mcp_server._resource_handlers["method://{method_id}"]
        schema_handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        # Should be coroutine functions
        import asyncio

        assert asyncio.iscoroutinefunction(metadata_handler)
        assert asyncio.iscoroutinefunction(schema_handler)


# ============================================================================
# Integration tests
# ============================================================================


class TestResourceIntegration:
    """Integration tests for method resources."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metadata_and_schema_consistency(self, mock_mcp_server: MagicMock):
        """Test that metadata and schema are consistent for same method."""
        register_method_resources(mock_mcp_server)
        metadata_handler = mock_mcp_server._resource_handlers["method://{method_id}"]
        schema_handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        method_id = "ethical_reasoning"

        # Get both
        metadata_json = await metadata_handler(method_id)
        schema_json = await schema_handler(method_id)

        metadata = json.loads(metadata_json)
        schema = json.loads(schema_json)

        # Check consistency
        assert metadata["name"] in schema["title"]
        assert metadata["name"] in schema["description"]

        # If metadata says context required, schema should require it
        if metadata["requires_context"]:
            assert "context" in schema["required"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_methods_in_registry_have_valid_schema(self, mock_mcp_server: MagicMock):
        """Test that all registered methods return valid schemas."""
        register_method_resources(mock_mcp_server)
        schema_handler = mock_mcp_server._resource_handlers["method://{method_id}/schema"]

        # Get context and registry
        ctx = mock_mcp_server.get_context()
        registry = ctx.registry

        # Test each registered method
        for method_id in registry.registered_identifiers:
            result = await schema_handler(method_id)
            schema = json.loads(result)

            # Verify basic schema structure
            assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
            assert "input_text" in schema["properties"]
            assert "context" in schema["properties"]
            assert "input_text" in schema["required"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_methods_in_registry_have_valid_metadata(self, mock_mcp_server: MagicMock):
        """Test that all registered methods return valid metadata."""
        register_method_resources(mock_mcp_server)
        metadata_handler = mock_mcp_server._resource_handlers["method://{method_id}"]

        # Get context and registry
        ctx = mock_mcp_server.get_context()
        registry = ctx.registry

        # Test each registered method
        for method_id in registry.registered_identifiers:
            result = await metadata_handler(method_id)
            metadata = json.loads(result)

            # Verify all required fields
            required_fields = [
                "identifier",
                "name",
                "description",
                "category",
                "complexity",
                "supports_branching",
                "supports_revision",
                "requires_context",
                "min_thoughts",
                "max_thoughts",
                "avg_tokens_per_thought",
                "tags",
                "best_for",
                "not_recommended_for",
            ]
            for field in required_fields:
                assert field in metadata, f"Missing field: {field}"
