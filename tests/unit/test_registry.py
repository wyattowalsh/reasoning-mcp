"""
Comprehensive tests for MethodRegistry in reasoning_mcp.registry.

This module provides complete test coverage for the MethodRegistry class:
- Registry initialization
- Method registration and unregistration
- Method retrieval (get, get_metadata)
- Method listing with filtering
- Method initialization
- Health checks

The tests use a MockMethod class that implements the ReasoningMethod protocol
to avoid dependencies on actual method implementations.
"""

import pytest
from typing import Any

from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier


# ============================================================================
# Mock Method Implementation
# ============================================================================


class MockMethod:
    """Mock reasoning method for testing.

    This class implements the ReasoningMethod protocol to be used in registry tests
    without requiring actual method implementations.
    """

    def __init__(
        self,
        identifier: str = "mock_method",
        name: str = "Mock Method",
        description: str = "A mock method",
        category: str = "core",
        healthy: bool = True,
        init_error: bool = False,
    ):
        self._identifier = identifier
        self._name = name
        self._description = description
        self._category = category
        self._healthy = healthy
        self._init_error = init_error
        self._initialized = False

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
        if self._init_error:
            raise RuntimeError("Init failed")
        self._initialized = True

    async def execute(self, session, input_text: str, *, context: dict[str, Any] | None = None):
        return None

    async def continue_reasoning(self, session, previous_thought, *, guidance: str | None = None, context: dict[str, Any] | None = None):
        return None

    async def health_check(self) -> bool:
        return self._healthy


# ============================================================================
# Registry Initialization Tests
# ============================================================================


class TestMethodRegistryInit:
    """Tests for MethodRegistry initialization."""

    def test_empty_registry(self):
        """Test creating an empty registry."""
        registry = MethodRegistry()
        assert registry.method_count == 0
        assert registry.registered_identifiers == frozenset()

    def test_no_methods_registered(self):
        """Test that no methods are registered initially."""
        registry = MethodRegistry()
        assert not registry.is_registered("any_method")

    def test_registry_internal_state(self):
        """Test internal state is properly initialized."""
        registry = MethodRegistry()
        # Access internal state to verify initialization
        assert hasattr(registry, "_methods")
        assert hasattr(registry, "_metadata")
        assert hasattr(registry, "_initialized")


# ============================================================================
# Method Registration Tests
# ============================================================================


class TestMethodRegistryRegister:
    """Tests for method registration."""

    def test_register_method(self):
        """Test registering a method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step reasoning",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)

        assert registry.method_count == 1
        assert registry.is_registered(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert registry.is_registered("chain_of_thought")

    def test_register_duplicate_raises(self):
        """Test that registering duplicate raises ValueError."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(method, metadata)

    def test_register_with_replace(self):
        """Test that replace=True allows overwriting."""
        registry = MethodRegistry()
        method1 = MockMethod(name="Method 1")
        method2 = MockMethod(name="Method 2")
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method1, metadata)
        registry.register(method2, metadata, replace=True)

        assert registry.method_count == 1
        retrieved = registry.get(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert retrieved.name == "Method 2"

    def test_register_multiple_methods(self):
        """Test registering multiple different methods."""
        registry = MethodRegistry()

        method1 = MockMethod(identifier="method1")
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        method2 = MockMethod(identifier="method2")
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method1, metadata1)
        registry.register(method2, metadata2)

        assert registry.method_count == 2
        assert registry.is_registered("chain_of_thought")
        assert registry.is_registered("tree_of_thoughts")

    def test_register_invalid_method_raises(self):
        """Test that registering invalid object raises TypeError."""
        registry = MethodRegistry()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        # Try to register an object that doesn't implement the protocol
        invalid_method = "not a method"

        with pytest.raises(TypeError, match="ReasoningMethod protocol"):
            registry.register(invalid_method, metadata)


# ============================================================================
# Method Unregistration Tests
# ============================================================================


class TestMethodRegistryUnregister:
    """Tests for method unregistration."""

    def test_unregister_existing(self):
        """Test unregistering an existing method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        result = registry.unregister(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert result is True
        assert registry.method_count == 0
        assert not registry.is_registered("chain_of_thought")

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent method returns False."""
        registry = MethodRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_unregister_removes_metadata(self):
        """Test that unregistering also removes metadata."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        registry.unregister(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert registry.get_metadata("chain_of_thought") is None

    def test_unregister_removes_initialization_status(self):
        """Test that unregistering removes initialization status."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        # Manually mark as initialized
        registry._initialized.add("chain_of_thought")

        registry.unregister(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert not registry.is_initialized("chain_of_thought")


# ============================================================================
# Method Retrieval Tests
# ============================================================================


class TestMethodRegistryGet:
    """Tests for getting methods."""

    def test_get_existing(self):
        """Test getting an existing method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        retrieved = registry.get(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert retrieved is method

    def test_get_nonexistent(self):
        """Test getting a non-existent method returns None."""
        registry = MethodRegistry()
        assert registry.get("nonexistent") is None

    def test_get_with_string_identifier(self):
        """Test getting a method with string identifier."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        retrieved = registry.get("chain_of_thought")

        assert retrieved is method

    def test_get_metadata(self):
        """Test getting metadata for a method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        registry.register(method, metadata)
        retrieved = registry.get_metadata(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert retrieved is metadata

    def test_get_metadata_nonexistent(self):
        """Test getting metadata for non-existent method returns None."""
        registry = MethodRegistry()
        assert registry.get_metadata("nonexistent") is None


# ============================================================================
# Method Listing Tests
# ============================================================================


class TestMethodRegistryList:
    """Tests for listing methods."""

    def test_list_all(self):
        """Test listing all methods."""
        registry = MethodRegistry()

        for identifier in [MethodIdentifier.CHAIN_OF_THOUGHT, MethodIdentifier.TREE_OF_THOUGHTS]:
            method = MockMethod(identifier=str(identifier))
            metadata = MethodMetadata(
                identifier=identifier,
                name=str(identifier),
                description="Test",
                category=MethodCategory.CORE,
            )
            registry.register(method, metadata)

        methods = registry.list_methods()
        assert len(methods) == 2

    def test_list_empty_registry(self):
        """Test listing methods in an empty registry."""
        registry = MethodRegistry()
        methods = registry.list_methods()
        assert methods == []

    def test_list_by_category(self):
        """Test filtering by category."""
        registry = MethodRegistry()

        # Register core method
        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method1, metadata1)

        # Register high_value method
        method2 = MockMethod()
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.ETHICAL_REASONING,
            name="Ethical",
            description="Test",
            category=MethodCategory.HIGH_VALUE,
        )
        registry.register(method2, metadata2)

        core_methods = registry.list_methods(category=MethodCategory.CORE)
        assert len(core_methods) == 1
        assert core_methods[0].identifier == MethodIdentifier.CHAIN_OF_THOUGHT

    def test_list_by_category_string(self):
        """Test filtering by category using string."""
        registry = MethodRegistry()

        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        methods = registry.list_methods(category="core")
        assert len(methods) == 1

    def test_list_by_tags(self):
        """Test filtering by tags."""
        registry = MethodRegistry()

        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            tags=frozenset({"step-by-step", "reasoning"}),
        )
        registry.register(method, metadata)

        # Matching tags
        matches = registry.list_methods(tags={"step-by-step"})
        assert len(matches) == 1

        # Non-matching tags
        no_matches = registry.list_methods(tags={"branching"})
        assert len(no_matches) == 0

    def test_list_by_multiple_tags(self):
        """Test filtering by multiple tags (must have ALL)."""
        registry = MethodRegistry()

        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            tags=frozenset({"step-by-step", "reasoning", "linear"}),
        )
        registry.register(method1, metadata1)

        method2 = MockMethod()
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
            tags=frozenset({"step-by-step", "branching"}),
        )
        registry.register(method2, metadata2)

        # Both have "step-by-step"
        matches = registry.list_methods(tags={"step-by-step"})
        assert len(matches) == 2

        # Only method1 has both "step-by-step" and "reasoning"
        matches = registry.list_methods(tags={"step-by-step", "reasoning"})
        assert len(matches) == 1
        assert matches[0].identifier == MethodIdentifier.CHAIN_OF_THOUGHT

    def test_list_by_initialized_only(self):
        """Test filtering by initialization status."""
        registry = MethodRegistry()

        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method1, metadata1)

        method2 = MockMethod()
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method2, metadata2)

        # Mark only method1 as initialized
        registry._initialized.add("chain_of_thought")

        initialized = registry.list_methods(initialized_only=True)
        assert len(initialized) == 1
        assert initialized[0].identifier == MethodIdentifier.CHAIN_OF_THOUGHT

    def test_list_with_combined_filters(self):
        """Test combining multiple filters."""
        registry = MethodRegistry()

        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            tags=frozenset({"step-by-step"}),
        )
        registry.register(method1, metadata1)

        method2 = MockMethod()
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.ETHICAL_REASONING,
            name="Ethical",
            description="Test",
            category=MethodCategory.HIGH_VALUE,
            tags=frozenset({"step-by-step"}),
        )
        registry.register(method2, metadata2)

        # Filter by both category and tags
        matches = registry.list_methods(
            category=MethodCategory.CORE,
            tags={"step-by-step"}
        )
        assert len(matches) == 1
        assert matches[0].identifier == MethodIdentifier.CHAIN_OF_THOUGHT


# ============================================================================
# Method Initialization Tests
# ============================================================================


class TestMethodRegistryInitialize:
    """Tests for method initialization."""

    async def test_initialize_all(self):
        """Test initializing all methods."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.initialize()

        assert results["chain_of_thought"] is True
        assert registry.is_initialized("chain_of_thought")

    async def test_initialize_specific(self):
        """Test initializing a specific method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.initialize(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert results["chain_of_thought"] is True

    async def test_initialize_multiple(self):
        """Test initializing multiple methods."""
        registry = MethodRegistry()

        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method1, metadata1)

        method2 = MockMethod()
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method2, metadata2)

        results = await registry.initialize()

        assert len(results) == 2
        assert results["chain_of_thought"] is True
        assert results["tree_of_thoughts"] is True

    async def test_initialize_failure(self):
        """Test handling initialization failure."""
        registry = MethodRegistry()
        method = MockMethod(init_error=True)
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.initialize()

        assert results["chain_of_thought"] is False
        assert not registry.is_initialized("chain_of_thought")

    async def test_initialize_nonexistent(self):
        """Test initializing a non-existent method."""
        registry = MethodRegistry()

        results = await registry.initialize("nonexistent")

        assert results["nonexistent"] is False

    async def test_initialize_already_initialized(self):
        """Test initializing an already-initialized method."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        # Initialize once
        await registry.initialize()

        # Initialize again - should succeed without re-initializing
        results = await registry.initialize(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert results["chain_of_thought"] is True

    async def test_initialize_partial_failure(self):
        """Test initialization when some methods fail."""
        registry = MethodRegistry()

        method1 = MockMethod()
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method1, metadata1)

        method2 = MockMethod(init_error=True)
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method2, metadata2)

        results = await registry.initialize()

        assert results["chain_of_thought"] is True
        assert results["tree_of_thoughts"] is False


# ============================================================================
# Health Check Tests
# ============================================================================


class TestMethodRegistryHealthCheck:
    """Tests for health checks."""

    async def test_health_check_healthy(self):
        """Test health check on healthy method."""
        registry = MethodRegistry()
        method = MockMethod(healthy=True)
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.health_check()

        assert results["chain_of_thought"] is True

    async def test_health_check_unhealthy(self):
        """Test health check on unhealthy method."""
        registry = MethodRegistry()
        method = MockMethod(healthy=False)
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.health_check()

        assert results["chain_of_thought"] is False

    async def test_health_check_nonexistent(self):
        """Test health check on non-existent method."""
        registry = MethodRegistry()
        results = await registry.health_check("nonexistent")
        assert results["nonexistent"] is False

    async def test_health_check_specific_method(self):
        """Test health check on a specific method."""
        registry = MethodRegistry()
        method = MockMethod(healthy=True)
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.health_check(MethodIdentifier.CHAIN_OF_THOUGHT)

        assert len(results) == 1
        assert results["chain_of_thought"] is True

    async def test_health_check_multiple_methods(self):
        """Test health check on multiple methods."""
        registry = MethodRegistry()

        method1 = MockMethod(healthy=True)
        metadata1 = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method1, metadata1)

        method2 = MockMethod(healthy=False)
        metadata2 = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method2, metadata2)

        results = await registry.health_check()

        assert len(results) == 2
        assert results["chain_of_thought"] is True
        assert results["tree_of_thoughts"] is False

    async def test_health_check_exception_handling(self):
        """Test health check handles exceptions gracefully."""
        registry = MethodRegistry()

        # Create a method that raises an exception during health check
        class BrokenMethod(MockMethod):
            async def health_check(self) -> bool:
                raise RuntimeError("Health check failed")

        method = BrokenMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )
        registry.register(method, metadata)

        results = await registry.health_check()

        assert results["chain_of_thought"] is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestMethodRegistryIntegration:
    """Integration tests for complete workflows."""

    async def test_full_lifecycle(self):
        """Test complete method lifecycle: register -> initialize -> health check -> unregister."""
        registry = MethodRegistry()
        method = MockMethod()
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        # Register
        registry.register(method, metadata)
        assert registry.is_registered("chain_of_thought")

        # Initialize
        init_results = await registry.initialize()
        assert init_results["chain_of_thought"] is True
        assert registry.is_initialized("chain_of_thought")

        # Health check
        health_results = await registry.health_check()
        assert health_results["chain_of_thought"] is True

        # Unregister
        result = registry.unregister(MethodIdentifier.CHAIN_OF_THOUGHT)
        assert result is True
        assert not registry.is_registered("chain_of_thought")

    async def test_replace_and_reinitialize(self):
        """Test replacing a method and reinitializing."""
        registry = MethodRegistry()

        method1 = MockMethod(name="Method 1", healthy=True)
        method2 = MockMethod(name="Method 2", healthy=False)
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
        )

        # Register and initialize first method
        registry.register(method1, metadata)
        await registry.initialize()

        health1 = await registry.health_check()
        assert health1["chain_of_thought"] is True

        # Replace with second method
        registry.register(method2, metadata, replace=True)

        # Should still be marked as initialized
        assert registry.is_initialized("chain_of_thought")

        # But health check should reflect new method
        health2 = await registry.health_check()
        assert health2["chain_of_thought"] is False

    def test_registered_identifiers_property(self):
        """Test the registered_identifiers property returns correct set."""
        registry = MethodRegistry()

        for identifier in [
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.REACT,
        ]:
            method = MockMethod()
            metadata = MethodMetadata(
                identifier=identifier,
                name=str(identifier),
                description="Test",
                category=MethodCategory.CORE,
            )
            registry.register(method, metadata)

        identifiers = registry.registered_identifiers
        assert isinstance(identifiers, frozenset)
        assert len(identifiers) == 3
        assert "chain_of_thought" in identifiers
        assert "tree_of_thoughts" in identifiers
        assert "react" in identifiers
