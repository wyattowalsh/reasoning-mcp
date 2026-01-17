"""MCP resources for exposing reasoning method metadata and schemas.

This module provides MCP resources that allow clients to discover and inspect
reasoning methods available in the registry. Resources are exposed through
standard MCP resource URIs:
- method://{method_id} - Returns method metadata as JSON
- method://{method_id}/schema - Returns JSON schema for method input parameters

These resources enable clients to dynamically discover available methods and
their capabilities without hardcoding method information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reasoning_mcp.server import AppContext

logger = logging.getLogger(__name__)


def register_method_resources(mcp: FastMCP) -> None:
    """Register method-related MCP resources with the server.

    This function registers two resource endpoints:
    1. method://{method_id} - Returns method metadata
    2. method://{method_id}/schema - Returns method's JSON schema

    Args:
        mcp: The FastMCP server instance to register resources with

    Example:
        >>> from reasoning_mcp.server import mcp
        >>> register_method_resources(mcp)
    """

    @mcp.resource("method://{method_id}")
    async def get_method_metadata(method_id: str) -> str:
        """Get metadata for a specific reasoning method.

        This resource returns comprehensive metadata about a reasoning method
        including its description, category, complexity, capabilities, and
        usage recommendations.

        Args:
            method_id: The unique identifier of the method (e.g., "chain_of_thought")

        Returns:
            JSON string containing method metadata

        Resource URI:
            method://{method_id}

        Example Response:
            {
                "identifier": "chain_of_thought",
                "name": "Chain of Thought",
                "description": "Step-by-step reasoning...",
                "category": "core",
                "complexity": 3,
                "supports_branching": false,
                "supports_revision": true,
                "requires_context": false,
                "min_thoughts": 3,
                "max_thoughts": 0,
                "avg_tokens_per_thought": 500,
                "tags": ["reasoning", "step-by-step"],
                "best_for": ["logical problems", "math"],
                "not_recommended_for": ["creative tasks"]
            }

        Raises:
            ValueError: If method_id is not found in registry
        """
        import json

        # Get app context from server
        from reasoning_mcp.server import get_app_context

        ctx: AppContext = get_app_context()

        # Validate registry is initialized
        if not ctx.initialized:
            logger.error("Cannot retrieve metadata: registry not initialized")
            raise RuntimeError("Registry not initialized")

        # Get metadata from registry
        metadata = ctx.registry.get_metadata(method_id)
        if metadata is None:
            logger.warning(f"Method not found: {method_id}")
            raise ValueError(f"Method '{method_id}' not found in registry")

        # Convert metadata to dict
        metadata_dict: dict[str, Any] = {
            "identifier": str(metadata.identifier),
            "name": metadata.name,
            "description": metadata.description,
            "category": str(metadata.category),
            "complexity": metadata.complexity,
            "supports_branching": metadata.supports_branching,
            "supports_revision": metadata.supports_revision,
            "requires_context": metadata.requires_context,
            "min_thoughts": metadata.min_thoughts,
            "max_thoughts": metadata.max_thoughts,
            "avg_tokens_per_thought": metadata.avg_tokens_per_thought,
            "tags": sorted(metadata.tags),
            "best_for": list(metadata.best_for),
            "not_recommended_for": list(metadata.not_recommended_for),
        }

        logger.debug(f"Retrieved metadata for method: {method_id}")
        return json.dumps(metadata_dict, indent=2)

    @mcp.resource("method://{method_id}/schema")
    async def get_method_schema(method_id: str) -> str:
        """Get JSON schema for a reasoning method's input parameters.

        This resource returns a JSON Schema (Draft 7) describing the expected
        input parameters for the method's execute() function. This enables
        clients to validate inputs and generate forms/UIs dynamically.

        Args:
            method_id: The unique identifier of the method

        Returns:
            JSON string containing JSON Schema for method inputs

        Resource URI:
            method://{method_id}/schema

        Example Response:
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "title": "Chain of Thought Input",
                "description": "Parameters for Chain of Thought reasoning",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "The problem or question to reason about",
                        "minLength": 1
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional additional context",
                        "additionalProperties": true
                    }
                },
                "required": ["input_text"],
                "additionalProperties": false
            }

        Raises:
            ValueError: If method_id is not found in registry
        """
        import json

        # Get app context from server
        from reasoning_mcp.server import get_app_context

        ctx: AppContext = get_app_context()

        # Validate registry is initialized
        if not ctx.initialized:
            logger.error("Cannot retrieve schema: registry not initialized")
            raise RuntimeError("Registry not initialized")

        # Verify method exists
        metadata = ctx.registry.get_metadata(method_id)
        if metadata is None:
            logger.warning(f"Method not found: {method_id}")
            raise ValueError(f"Method '{method_id}' not found in registry")

        # Build JSON Schema for method input
        # All methods follow the ReasoningMethod protocol with execute() signature:
        # execute(session, input_text, *, context=None) -> ThoughtNode
        schema: dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": f"{metadata.name} Input",
            "description": f"Parameters for {metadata.name} reasoning method",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "The problem, question, or task to reason about",
                    "minLength": 1,
                },
                "context": {
                    "type": "object",
                    "description": "Optional additional context and parameters",
                    "additionalProperties": True,
                },
            },
            "required": ["input_text"],
            "additionalProperties": False,
        }

        # Add method-specific context requirements if needed
        if metadata.requires_context:
            schema["required"].append("context")
            schema["properties"]["context"]["description"] = (
                "Required additional context for this method"
            )

        logger.debug(f"Retrieved schema for method: {method_id}")
        return json.dumps(schema, indent=2)

    logger.info("Registered method resources: method://{method_id}, method://{method_id}/schema")


__all__ = ["register_method_resources"]
