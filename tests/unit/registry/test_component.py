"""Comprehensive tests for ComponentRegistry in reasoning_mcp.registry.component.

This module provides complete test coverage for the ComponentRegistry class:
- Registry initialization
- Component registration and unregistration
- Component retrieval (get, get_components)
- Component existence checking (is_registered)
- Component type listing (list_component_types)
- Health checks
- Registry clearing

The tests use mock component objects to avoid dependencies on actual
component implementations.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.component_registry import ComponentRegistry
from reasoning_mcp.models.component import ComponentType

# ============================================================================
# Mock Component Classes
# ============================================================================


class MockMethod:
    """Mock reasoning method for testing."""

    def __init__(self, name: str = "Mock Method") -> None:
        self.name = name

    async def execute(self, query: str) -> str:
        return f"Result from {self.name}"


class MockExecutor:
    """Mock pipeline executor for testing."""

    def __init__(self, name: str = "Mock Executor") -> None:
        self.name = name

    async def execute_stage(self, stage: str) -> str:
        return f"Executed {stage} with {self.name}"


class MockMiddleware:
    """Mock middleware for testing."""

    def __init__(self, name: str = "Mock Middleware") -> None:
        self.name = name

    async def process(self, request: str) -> str:
        return f"Processed by {self.name}"


# ============================================================================
# Registry Initialization Tests
# ============================================================================


class TestComponentRegistryInit:
    """Tests for ComponentRegistry initialization."""

    def test_empty_registry(self) -> None:
        """Test creating an empty registry."""
        registry = ComponentRegistry()
        assert registry.list_component_types() == []

    def test_no_components_registered(self) -> None:
        """Test that no components are registered initially."""
        registry = ComponentRegistry()
        for component_type in ComponentType:
            assert not registry.is_registered(component_type, "any_component")

    def test_registry_internal_state(self) -> None:
        """Test internal state is properly initialized."""
        registry = ComponentRegistry()
        # Access internal state to verify initialization
        assert hasattr(registry, "_components")
        assert hasattr(registry, "_timeouts")
        assert hasattr(registry, "_max_retries")
        assert hasattr(registry, "_lock")

    def test_all_component_types_initialized(self) -> None:
        """Test that all component types have empty dictionaries."""
        registry = ComponentRegistry()
        for component_type in ComponentType:
            components = registry.get_components(component_type)
            assert components == {}


# ============================================================================
# Component Registration Tests
# ============================================================================


class TestComponentRegistryRegister:
    """Tests for component registration."""

    def test_register_method(self) -> None:
        """Test registering a method component."""
        registry = ComponentRegistry()
        method = MockMethod("Test Method")

        registry.register(ComponentType.METHOD, "test_method", method)

        assert registry.is_registered(ComponentType.METHOD, "test_method")
        retrieved = registry.get(ComponentType.METHOD, "test_method")
        assert retrieved is method

    def test_register_with_custom_timeout(self) -> None:
        """Test registering with custom timeout."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method, timeout=60.0)

        assert registry.is_registered(ComponentType.METHOD, "test_method")

    def test_register_with_custom_max_retries(self) -> None:
        """Test registering with custom max_retries."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method, max_retries=5)

        assert registry.is_registered(ComponentType.METHOD, "test_method")

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate raises ValueError."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ComponentType.METHOD, "test_method", method)

    def test_register_multiple_components_same_type(self) -> None:
        """Test registering multiple components of the same type."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)

        assert registry.is_registered(ComponentType.METHOD, "method1")
        assert registry.is_registered(ComponentType.METHOD, "method2")

    def test_register_components_different_types(self) -> None:
        """Test registering components of different types."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "test_method", method)
        registry.register(ComponentType.EXECUTOR, "test_executor", executor)

        assert registry.is_registered(ComponentType.METHOD, "test_method")
        assert registry.is_registered(ComponentType.EXECUTOR, "test_executor")

    def test_register_same_identifier_different_types(self) -> None:
        """Test that same identifier can be used for different types."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        # Same identifier "test" but different types - should both work
        registry.register(ComponentType.METHOD, "test", method)
        registry.register(ComponentType.EXECUTOR, "test", executor)

        assert registry.is_registered(ComponentType.METHOD, "test")
        assert registry.is_registered(ComponentType.EXECUTOR, "test")
        assert registry.get(ComponentType.METHOD, "test") is method
        assert registry.get(ComponentType.EXECUTOR, "test") is executor

    def test_register_all_component_types(self) -> None:
        """Test registering at least one component for each type."""
        registry = ComponentRegistry()

        # Create mock components for all types
        components = {
            ComponentType.METHOD: MockMethod(),
            ComponentType.EXECUTOR: MockExecutor(),
            ComponentType.MIDDLEWARE: MockMiddleware(),
            ComponentType.STORAGE: object(),
            ComponentType.EVALUATOR: object(),
            ComponentType.SELECTOR: object(),
            ComponentType.PIPELINE: object(),
            ComponentType.MCP_TOOL: object(),
            ComponentType.MCP_RESOURCE: object(),
        }

        for component_type, component in components.items():
            registry.register(component_type, "test", component)

        for component_type in ComponentType:
            assert registry.is_registered(component_type, "test")


# ============================================================================
# Component Unregistration Tests
# ============================================================================


class TestComponentRegistryUnregister:
    """Tests for component unregistration."""

    def test_unregister_existing(self) -> None:
        """Test unregistering an existing component."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        result = registry.unregister(ComponentType.METHOD, "test_method")

        assert result is True
        assert not registry.is_registered(ComponentType.METHOD, "test_method")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a non-existent component returns False."""
        registry = ComponentRegistry()
        result = registry.unregister(ComponentType.METHOD, "nonexistent")
        assert result is False

    def test_unregister_removes_from_get(self) -> None:
        """Test that unregistering removes component from get."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        registry.unregister(ComponentType.METHOD, "test_method")

        assert registry.get(ComponentType.METHOD, "test_method") is None

    def test_unregister_one_component_leaves_others(self) -> None:
        """Test that unregistering one component doesn't affect others."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)

        registry.unregister(ComponentType.METHOD, "method1")

        assert not registry.is_registered(ComponentType.METHOD, "method1")
        assert registry.is_registered(ComponentType.METHOD, "method2")

    def test_unregister_only_affects_specified_type(self) -> None:
        """Test that unregistering only affects the specified type."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "test", method)
        registry.register(ComponentType.EXECUTOR, "test", executor)

        registry.unregister(ComponentType.METHOD, "test")

        assert not registry.is_registered(ComponentType.METHOD, "test")
        assert registry.is_registered(ComponentType.EXECUTOR, "test")


# ============================================================================
# Component Retrieval Tests
# ============================================================================


class TestComponentRegistryGet:
    """Tests for getting components."""

    def test_get_existing(self) -> None:
        """Test getting an existing component."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        retrieved = registry.get(ComponentType.METHOD, "test_method")

        assert retrieved is method

    def test_get_nonexistent(self) -> None:
        """Test getting a non-existent component returns None."""
        registry = ComponentRegistry()
        assert registry.get(ComponentType.METHOD, "nonexistent") is None

    def test_get_from_wrong_type(self) -> None:
        """Test getting a component from wrong type returns None."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)

        # Try to get it as an executor
        assert registry.get(ComponentType.EXECUTOR, "test_method") is None

    def test_get_components_empty(self) -> None:
        """Test getting components from empty registry."""
        registry = ComponentRegistry()
        components = registry.get_components(ComponentType.METHOD)
        assert components == {}

    def test_get_components_single(self) -> None:
        """Test getting components with single registered component."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        components = registry.get_components(ComponentType.METHOD)

        assert len(components) == 1
        assert "test_method" in components
        assert components["test_method"] is method

    def test_get_components_multiple(self) -> None:
        """Test getting multiple components of same type."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")
        method3 = MockMethod("Method 3")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)
        registry.register(ComponentType.METHOD, "method3", method3)

        components = registry.get_components(ComponentType.METHOD)

        assert len(components) == 3
        assert components["method1"] is method1
        assert components["method2"] is method2
        assert components["method3"] is method3

    def test_get_components_returns_copy(self) -> None:
        """Test that get_components returns a copy, not original dict."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        components1 = registry.get_components(ComponentType.METHOD)
        components2 = registry.get_components(ComponentType.METHOD)

        # Should be equal but not the same object
        assert components1 == components2
        assert components1 is not components2

    def test_get_components_only_returns_specified_type(self) -> None:
        """Test that get_components only returns the specified type."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)

        method_components = registry.get_components(ComponentType.METHOD)
        executor_components = registry.get_components(ComponentType.EXECUTOR)

        assert len(method_components) == 1
        assert "method" in method_components
        assert "executor" not in method_components

        assert len(executor_components) == 1
        assert "executor" in executor_components
        assert "method" not in executor_components


# ============================================================================
# Component Existence Tests
# ============================================================================


class TestComponentRegistryIsRegistered:
    """Tests for checking component registration."""

    def test_is_registered_true(self) -> None:
        """Test is_registered returns True for registered component."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)

        assert registry.is_registered(ComponentType.METHOD, "test_method") is True

    def test_is_registered_false(self) -> None:
        """Test is_registered returns False for non-existent component."""
        registry = ComponentRegistry()
        assert registry.is_registered(ComponentType.METHOD, "nonexistent") is False

    def test_is_registered_wrong_type(self) -> None:
        """Test is_registered is type-specific."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)

        assert registry.is_registered(ComponentType.METHOD, "test_method") is True
        assert registry.is_registered(ComponentType.EXECUTOR, "test_method") is False

    def test_is_registered_after_unregister(self) -> None:
        """Test is_registered returns False after unregistration."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        assert registry.is_registered(ComponentType.METHOD, "test_method") is True

        registry.unregister(ComponentType.METHOD, "test_method")
        assert registry.is_registered(ComponentType.METHOD, "test_method") is False


# ============================================================================
# Component Type Listing Tests
# ============================================================================


class TestComponentRegistryListComponentTypes:
    """Tests for listing component types."""

    def test_list_component_types_empty(self) -> None:
        """Test listing component types in empty registry."""
        registry = ComponentRegistry()
        types = registry.list_component_types()
        assert types == []

    def test_list_component_types_single(self) -> None:
        """Test listing with single component type."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        types = registry.list_component_types()

        assert len(types) == 1
        assert ComponentType.METHOD in types

    def test_list_component_types_multiple(self) -> None:
        """Test listing with multiple component types."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()
        middleware = MockMiddleware()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)
        registry.register(ComponentType.MIDDLEWARE, "middleware", middleware)

        types = registry.list_component_types()

        assert len(types) == 3
        assert ComponentType.METHOD in types
        assert ComponentType.EXECUTOR in types
        assert ComponentType.MIDDLEWARE in types

    def test_list_component_types_after_unregister(self) -> None:
        """Test listing after unregistering all components of a type."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)

        types_before = registry.list_component_types()
        assert len(types_before) == 2

        # Unregister all methods
        registry.unregister(ComponentType.METHOD, "method")

        types_after = registry.list_component_types()
        assert len(types_after) == 1
        assert ComponentType.EXECUTOR in types_after
        assert ComponentType.METHOD not in types_after

    def test_list_component_types_multiple_per_type(self) -> None:
        """Test that multiple components per type still shows type once."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")
        method3 = MockMethod("Method 3")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)
        registry.register(ComponentType.METHOD, "method3", method3)

        types = registry.list_component_types()

        assert len(types) == 1
        assert ComponentType.METHOD in types


# ============================================================================
# Health Check Tests
# ============================================================================


class TestComponentRegistryHealthCheck:
    """Tests for health checks."""

    def test_health_check_empty_registry(self) -> None:
        """Test health check on empty registry."""
        registry = ComponentRegistry()
        health = registry.health_check()

        assert health["status"] == "degraded"
        assert health["total_components"] == 0
        assert health["component_types"] == {}

    def test_health_check_single_component(self) -> None:
        """Test health check with single component."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "test_method", method)
        health = registry.health_check()

        assert health["status"] == "healthy"
        assert health["total_components"] == 1
        assert "method" in health["component_types"]
        assert health["component_types"]["method"]["count"] == 1
        assert "test_method" in health["component_types"]["method"]["identifiers"]

    def test_health_check_multiple_types(self) -> None:
        """Test health check with multiple component types."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)

        health = registry.health_check()

        assert health["status"] == "healthy"
        assert health["total_components"] == 2
        assert len(health["component_types"]) == 2
        assert "method" in health["component_types"]
        assert "executor" in health["component_types"]

    def test_health_check_multiple_components_same_type(self) -> None:
        """Test health check with multiple components of same type."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")
        method3 = MockMethod("Method 3")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)
        registry.register(ComponentType.METHOD, "method3", method3)

        health = registry.health_check()

        assert health["status"] == "healthy"
        assert health["total_components"] == 3
        assert health["component_types"]["method"]["count"] == 3
        assert len(health["component_types"]["method"]["identifiers"]) == 3

    def test_health_check_format(self) -> None:
        """Test health check returns correct format."""
        registry = ComponentRegistry()
        method = MockMethod()

        registry.register(ComponentType.METHOD, "method", method)
        health = registry.health_check()

        # Verify structure
        assert "status" in health
        assert "component_types" in health
        assert "total_components" in health
        assert isinstance(health["status"], str)
        assert isinstance(health["component_types"], dict)
        assert isinstance(health["total_components"], int)

        # Verify component type structure
        assert "count" in health["component_types"]["method"]
        assert "identifiers" in health["component_types"]["method"]
        assert isinstance(health["component_types"]["method"]["count"], int)
        assert isinstance(health["component_types"]["method"]["identifiers"], list)


# ============================================================================
# Clear Tests
# ============================================================================


class TestComponentRegistryClear:
    """Tests for clearing components."""

    def test_clear_specific_type(self) -> None:
        """Test clearing a specific component type."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)

        registry.clear(ComponentType.METHOD)

        assert not registry.is_registered(ComponentType.METHOD, "method")
        assert registry.is_registered(ComponentType.EXECUTOR, "executor")

    def test_clear_all(self) -> None:
        """Test clearing all components."""
        registry = ComponentRegistry()

        method = MockMethod()
        executor = MockExecutor()
        middleware = MockMiddleware()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)
        registry.register(ComponentType.MIDDLEWARE, "middleware", middleware)

        registry.clear()

        assert not registry.is_registered(ComponentType.METHOD, "method")
        assert not registry.is_registered(ComponentType.EXECUTOR, "executor")
        assert not registry.is_registered(ComponentType.MIDDLEWARE, "middleware")
        assert registry.list_component_types() == []

    def test_clear_empty_type(self) -> None:
        """Test clearing an empty component type."""
        registry = ComponentRegistry()

        # Clear a type that has no components
        registry.clear(ComponentType.METHOD)

        # Should not raise error
        assert not registry.is_registered(ComponentType.METHOD, "anything")

    def test_clear_type_with_multiple_components(self) -> None:
        """Test clearing a type with multiple components."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")
        method3 = MockMethod("Method 3")

        registry.register(ComponentType.METHOD, "method1", method1)
        registry.register(ComponentType.METHOD, "method2", method2)
        registry.register(ComponentType.METHOD, "method3", method3)

        registry.clear(ComponentType.METHOD)

        assert len(registry.get_components(ComponentType.METHOD)) == 0

    def test_clear_all_leaves_empty_registry(self) -> None:
        """Test that clearing all leaves the registry in initial state."""
        registry = ComponentRegistry()

        # Register various components
        for component_type in [
            ComponentType.METHOD,
            ComponentType.EXECUTOR,
            ComponentType.MIDDLEWARE,
        ]:
            registry.register(component_type, "test", object())

        registry.clear()

        # Verify empty state
        assert registry.list_component_types() == []
        health = registry.health_check()
        assert health["status"] == "degraded"
        assert health["total_components"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestComponentRegistryIntegration:
    """Integration tests for complete workflows."""

    def test_full_lifecycle(self) -> None:
        """Test complete component lifecycle: register -> get -> unregister."""
        registry = ComponentRegistry()
        method = MockMethod("Test Method")

        # Register
        registry.register(ComponentType.METHOD, "test_method", method)
        assert registry.is_registered(ComponentType.METHOD, "test_method")

        # Get
        retrieved = registry.get(ComponentType.METHOD, "test_method")
        assert retrieved is method

        # Unregister
        result = registry.unregister(ComponentType.METHOD, "test_method")
        assert result is True
        assert not registry.is_registered(ComponentType.METHOD, "test_method")

    def test_multiple_types_workflow(self) -> None:
        """Test workflow with multiple component types."""
        registry = ComponentRegistry()

        # Register different types
        method = MockMethod()
        executor = MockExecutor()
        middleware = MockMiddleware()

        registry.register(ComponentType.METHOD, "method", method)
        registry.register(ComponentType.EXECUTOR, "executor", executor)
        registry.register(ComponentType.MIDDLEWARE, "middleware", middleware)

        # Health check
        health = registry.health_check()
        assert health["total_components"] == 3
        assert len(health["component_types"]) == 3

        # List types
        types = registry.list_component_types()
        assert len(types) == 3

        # Clear specific type
        registry.clear(ComponentType.METHOD)
        assert not registry.is_registered(ComponentType.METHOD, "method")
        assert registry.is_registered(ComponentType.EXECUTOR, "executor")

        # Final health check
        health = registry.health_check()
        assert health["total_components"] == 2

    def test_replace_pattern(self) -> None:
        """Test pattern of unregistering and re-registering."""
        registry = ComponentRegistry()

        method1 = MockMethod("Method 1")
        method2 = MockMethod("Method 2")

        # Register first
        registry.register(ComponentType.METHOD, "test", method1)
        assert registry.get(ComponentType.METHOD, "test") is method1

        # Unregister and register new
        registry.unregister(ComponentType.METHOD, "test")
        registry.register(ComponentType.METHOD, "test", method2)
        assert registry.get(ComponentType.METHOD, "test") is method2

    def test_bulk_operations(self) -> None:
        """Test bulk registration and operations."""
        registry = ComponentRegistry()

        # Register 10 methods
        methods = [MockMethod(f"Method {i}") for i in range(10)]
        for i, method in enumerate(methods):
            registry.register(ComponentType.METHOD, f"method_{i}", method)

        # Verify all registered
        assert len(registry.get_components(ComponentType.METHOD)) == 10

        # Health check
        health = registry.health_check()
        assert health["total_components"] == 10

        # Unregister half
        for i in range(5):
            registry.unregister(ComponentType.METHOD, f"method_{i}")

        # Verify count
        assert len(registry.get_components(ComponentType.METHOD)) == 5

        # Clear all
        registry.clear(ComponentType.METHOD)
        assert len(registry.get_components(ComponentType.METHOD)) == 0
