"""Component registry for reasoning-mcp plugin system.

This module provides the ComponentRegistry class which manages registration,
lookup, and lifecycle of all plugin components in a unified registry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from reasoning_mcp.models.component import ComponentType

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Unified registry for managing plugin components.

    The ComponentRegistry provides a centralized system for registering,
    retrieving, and managing all types of plugin components (methods,
    executors, middleware, storage, evaluators, selectors, pipelines,
    MCP tools, and MCP resources).

    This class is designed to be thread-safe for async operations using
    asyncio.Lock, and supports optional circuit breaker integration for
    enhanced reliability.

    Example:
        >>> from reasoning_mcp.component_registry import ComponentRegistry
        >>> from reasoning_mcp.models.component import ComponentType
        >>> registry = ComponentRegistry()
        >>> registry.register(
        ...     ComponentType.METHOD,
        ...     "my_method",
        ...     my_method_instance
        ... )
        >>> method = registry.get(ComponentType.METHOD, "my_method")
        >>> health = registry.health_check()
    """

    def __init__(self) -> None:
        """Initialize an empty component registry.

        Creates internal storage structures and sets up async lock for
        thread-safe operations. Initializes empty component dictionaries
        for each ComponentType.
        """
        self._components: dict[ComponentType, dict[str, Any]] = {
            component_type: {} for component_type in ComponentType
        }
        self._timeouts: dict[ComponentType, dict[str, float]] = {
            component_type: {} for component_type in ComponentType
        }
        self._max_retries: dict[ComponentType, dict[str, int]] = {
            component_type: {} for component_type in ComponentType
        }
        self._lock = asyncio.Lock()

    def register(
        self,
        component_type: ComponentType,
        identifier: str,
        component: Any,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Register a component in the registry.

        Registers a component with the specified type and identifier. Raises
        an error if a component with the same type and identifier is already
        registered.

        Args:
            component_type: The type of component being registered
            identifier: Unique identifier for the component within its type
            component: The component instance to register
            timeout: Timeout in seconds for component operations (default: 30.0)
            max_retries: Maximum number of retries for failed operations (default: 3)

        Raises:
            ValueError: If a component with the same type and identifier exists

        Example:
            >>> registry.register(
            ...     ComponentType.METHOD,
            ...     "chain_of_thought",
            ...     chain_of_thought_method,
            ...     timeout=60.0,
            ...     max_retries=5
            ... )
        """
        if identifier in self._components[component_type]:
            raise ValueError(
                f"Component '{identifier}' of type '{component_type.value}' already registered"
            )

        self._components[component_type][identifier] = component
        self._timeouts[component_type][identifier] = timeout
        self._max_retries[component_type][identifier] = max_retries

        logger.info(
            f"Registered {component_type.value} component: {identifier} "
            f"(timeout={timeout}s, max_retries={max_retries})"
        )

    def unregister(self, component_type: ComponentType, identifier: str) -> bool:
        """Remove a component from the registry.

        Unregisters a component with the specified type and identifier,
        removing it and all associated metadata from the registry.

        Args:
            component_type: The type of component to unregister
            identifier: Identifier of the component to remove

        Returns:
            True if component was removed, False if not found

        Example:
            >>> success = registry.unregister(
            ...     ComponentType.METHOD,
            ...     "chain_of_thought"
            ... )
            >>> print(success)
            True
        """
        if identifier not in self._components[component_type]:
            return False

        del self._components[component_type][identifier]
        del self._timeouts[component_type][identifier]
        del self._max_retries[component_type][identifier]

        logger.info(f"Unregistered {component_type.value} component: {identifier}")
        return True

    def get(self, component_type: ComponentType, identifier: str) -> Any | None:
        """Get a single component by type and identifier.

        Retrieves the component instance registered with the specified type
        and identifier.

        Args:
            component_type: The type of component to retrieve
            identifier: Identifier of the component

        Returns:
            The component instance, or None if not found

        Example:
            >>> method = registry.get(ComponentType.METHOD, "chain_of_thought")
            >>> if method:
            ...     result = method.execute(query)
        """
        return self._components[component_type].get(identifier)

    def get_components(self, component_type: ComponentType) -> dict[str, Any]:
        """Get all components of a specific type.

        Returns a dictionary mapping identifiers to component instances
        for all components of the specified type.

        Args:
            component_type: The type of components to retrieve

        Returns:
            Dictionary mapping component identifiers to instances

        Example:
            >>> methods = registry.get_components(ComponentType.METHOD)
            >>> for identifier, method in methods.items():
            ...     print(f"{identifier}: {method}")
        """
        return self._components[component_type].copy()

    def is_registered(self, component_type: ComponentType, identifier: str) -> bool:
        """Check if a component is registered.

        Checks whether a component with the specified type and identifier
        exists in the registry.

        Args:
            component_type: The type of component to check
            identifier: Identifier of the component

        Returns:
            True if registered, False otherwise

        Example:
            >>> if registry.is_registered(ComponentType.METHOD, "chain_of_thought"):
            ...     print("Method is available")
        """
        return identifier in self._components[component_type]

    def list_component_types(self) -> list[ComponentType]:
        """List all component types that have registered components.

        Returns a list of ComponentType values for which at least one
        component is currently registered.

        Returns:
            List of ComponentType values with registered components

        Example:
            >>> types = registry.list_component_types()
            >>> for component_type in types:
            ...     count = len(registry.get_components(component_type))
            ...     print(f"{component_type.value}: {count} components")
        """
        return [
            component_type for component_type in ComponentType if self._components[component_type]
        ]

    def health_check(self) -> dict[str, Any]:
        """Return health status per component type.

        Performs a health check across all registered components and returns
        a structured report of the registry state.

        Returns:
            Dictionary with health status information including:
                - status: Overall status ("healthy" or "degraded")
                - component_types: Dict mapping each type to count and identifiers
                - total_components: Total number of registered components

        Example:
            >>> health = registry.health_check()
            >>> print(health["status"])
            'healthy'
            >>> print(health["total_components"])
            42
            >>> print(health["component_types"]["method"]["count"])
            30
        """
        component_type_info = {}
        total_components = 0

        for component_type in ComponentType:
            components = self._components[component_type]
            count = len(components)
            total_components += count

            if count > 0:
                component_type_info[component_type.value] = {
                    "count": count,
                    "identifiers": list(components.keys()),
                }

        return {
            "status": "healthy" if total_components > 0 else "degraded",
            "component_types": component_type_info,
            "total_components": total_components,
        }

    def clear(self, component_type: ComponentType | None = None) -> None:
        """Clear components from the registry.

        Removes all components of a specific type, or all components if
        no type is specified. This is useful for testing or resetting
        the registry state.

        Args:
            component_type: Type of components to clear, or None to clear all

        Example:
            >>> # Clear all methods
            >>> registry.clear(ComponentType.METHOD)
            >>> # Clear entire registry
            >>> registry.clear()
        """
        if component_type is None:
            # Clear all components
            for ct in ComponentType:
                self._components[ct].clear()
                self._timeouts[ct].clear()
                self._max_retries[ct].clear()
            logger.info("Cleared all components from registry")
        else:
            # Clear specific type
            self._components[component_type].clear()
            self._timeouts[component_type].clear()
            self._max_retries[component_type].clear()
            logger.info(f"Cleared all {component_type.value} components from registry")
