"""Method registry for reasoning-mcp.

This module provides the MethodRegistry class which manages registration,
lookup, and lifecycle of reasoning methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# ReasoningMethod must be imported at runtime for isinstance() checks
# since it's a @runtime_checkable Protocol
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod

if TYPE_CHECKING:
    from reasoning_mcp.models.core import MethodCategory, MethodIdentifier

logger = logging.getLogger(__name__)


class MethodRegistry:
    """Registry for managing reasoning methods.

    The MethodRegistry is responsible for:
    - Registering and unregistering reasoning methods
    - Looking up methods by identifier
    - Listing methods by category or tags
    - Managing method lifecycle (initialization, health checks)

    This class is typically used as a singleton within the application context.

    Example:
        >>> registry = MethodRegistry()
        >>> registry.register(my_method, my_metadata)
        >>> method = registry.get("chain_of_thought")
        >>> await method.execute(session, "What is 2+2?")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._methods: dict[str, ReasoningMethod] = {}
        self._metadata: dict[str, MethodMetadata] = {}
        self._initialized: set[str] = set()

    @property
    def method_count(self) -> int:
        """Return the number of registered methods."""
        return len(self._methods)

    @property
    def registered_identifiers(self) -> frozenset[str]:
        """Return the set of registered method identifiers."""
        return frozenset(self._methods.keys())

    def is_registered(self, identifier: str | MethodIdentifier) -> bool:
        """Check if a method is registered.

        Args:
            identifier: Method identifier to check

        Returns:
            True if registered, False otherwise
        """
        key = str(identifier)
        return key in self._methods

    def is_initialized(self, identifier: str | MethodIdentifier) -> bool:
        """Check if a method has been initialized.

        Args:
            identifier: Method identifier to check

        Returns:
            True if initialized, False otherwise
        """
        key = str(identifier)
        return key in self._initialized

    def register(
        self,
        method: ReasoningMethod,
        metadata: MethodMetadata,
        *,
        replace: bool = False,
    ) -> None:
        """Register a reasoning method.

        Args:
            method: The reasoning method instance
            metadata: Metadata describing the method
            replace: If True, replace existing method; if False, raise on duplicate

        Raises:
            ValueError: If method already registered and replace=False
            TypeError: If method doesn't satisfy ReasoningMethod protocol
        """
        key = str(metadata.identifier)
        if key in self._methods and not replace:
            raise ValueError(f"Method '{key}' already registered")

        # Validate protocol compliance at runtime
        if not isinstance(method, ReasoningMethod):
            raise TypeError(f"Method must satisfy ReasoningMethod protocol")

        self._methods[key] = method
        self._metadata[key] = metadata
        logger.info(f"Registered method: {key}")

    def unregister(self, identifier: str | MethodIdentifier) -> bool:
        """Unregister a reasoning method.

        Args:
            identifier: Method identifier to remove

        Returns:
            True if method was removed, False if not found
        """
        key = str(identifier)
        if key not in self._methods:
            return False

        del self._methods[key]
        del self._metadata[key]
        self._initialized.discard(key)
        logger.info(f"Unregistered method: {key}")
        return True

    def get(self, identifier: str | MethodIdentifier) -> ReasoningMethod | None:
        """Get a reasoning method by identifier.

        Args:
            identifier: Method identifier

        Returns:
            The reasoning method, or None if not found
        """
        return self._methods.get(str(identifier))

    def get_metadata(self, identifier: str | MethodIdentifier) -> MethodMetadata | None:
        """Get metadata for a reasoning method.

        Args:
            identifier: Method identifier

        Returns:
            The method metadata, or None if not found
        """
        return self._metadata.get(str(identifier))

    def list_methods(
        self,
        *,
        category: MethodCategory | str | None = None,
        tags: set[str] | None = None,
        initialized_only: bool = False,
    ) -> list[MethodMetadata]:
        """List registered methods with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (methods must have ALL tags)
            initialized_only: Only include initialized methods

        Returns:
            List of matching method metadata
        """
        results = []
        for key, metadata in self._metadata.items():
            # Filter by category
            if category is not None and str(metadata.category) != str(category):
                continue
            # Filter by tags
            if tags is not None and not tags.issubset(metadata.tags):
                continue
            # Filter by initialization status
            if initialized_only and key not in self._initialized:
                continue
            results.append(metadata)
        return results

    async def initialize(
        self,
        identifier: str | MethodIdentifier | None = None,
    ) -> dict[str, bool]:
        """Initialize one or all methods.

        Args:
            identifier: Specific method to initialize, or None for all

        Returns:
            Dict mapping method identifiers to initialization success status
        """
        results: dict[str, bool] = {}

        if identifier is not None:
            keys = [str(identifier)]
        else:
            keys = list(self._methods.keys())

        for key in keys:
            if key not in self._methods:
                logger.warning(f"Cannot initialize unknown method: {key}")
                results[key] = False
                continue

            if key in self._initialized:
                logger.debug(f"Method already initialized: {key}")
                results[key] = True
                continue

            try:
                method = self._methods[key]
                await method.initialize()
                self._initialized.add(key)
                logger.info(f"Initialized method: {key}")
                results[key] = True
            except Exception as e:
                logger.error(f"Failed to initialize method {key}: {e}")
                results[key] = False

        return results

    async def health_check(
        self,
        identifier: str | MethodIdentifier | None = None,
    ) -> dict[str, bool]:
        """Check health of one or all methods.

        Args:
            identifier: Specific method to check, or None for all

        Returns:
            Dict mapping method identifiers to health status
        """
        results: dict[str, bool] = {}

        if identifier is not None:
            keys = [str(identifier)]
        else:
            keys = list(self._methods.keys())

        for key in keys:
            if key not in self._methods:
                results[key] = False
                continue

            try:
                method = self._methods[key]
                results[key] = await method.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {key}: {e}")
                results[key] = False

        return results
