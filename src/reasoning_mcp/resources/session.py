"""MCP resources for session access.

This module provides MCP resource endpoints for accessing session state
and thought graph visualizations. Resources follow the URI pattern:
- session://{session_id} - Returns session state as JSON
- session://{session_id}/graph - Returns thought graph in mermaid format
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reasoning_mcp.server import AppContext


def register_session_resources(mcp: FastMCP) -> None:
    """Register session-related MCP resources with the server.

    This function registers two resources:
    1. session://{session_id} - Returns complete session state as JSON
    2. session://{session_id}/graph - Returns thought graph in mermaid format

    Args:
        mcp: The FastMCP server instance to register resources with

    Examples:
        >>> from mcp.server.fastmcp import FastMCP
        >>> mcp = FastMCP("test-server")
        >>> register_session_resources(mcp)
    """

    @mcp.resource("session://{session_id}")
    async def get_session_state(session_id: str) -> str:
        """Get complete session state as JSON.

        This resource returns the full session state including:
        - Session configuration and metadata
        - Current status and lifecycle information
        - Runtime metrics (thought counts, depth, confidence, etc.)
        - Active method and branch information
        - Thought graph structure (nodes and edges)

        Args:
            session_id: The unique identifier of the session to retrieve

        Returns:
            JSON string containing the complete session state

        Raises:
            ValueError: If session_id is not found

        Examples:
            Access via MCP client:
            >>> # MCP client request
            >>> resource = await client.read_resource("session://abc-123-def")
            >>> session_data = json.loads(resource.contents[0].text)
            >>> print(f"Status: {session_data['status']}")
            >>> print(f"Thoughts: {session_data['metrics']['total_thoughts']}")
        """
        # Get app context from server
        from reasoning_mcp.server import get_app_context

        ctx: AppContext = get_app_context()

        # Retrieve session from manager
        session = await ctx.session_manager.get(session_id)

        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Convert session to dict and serialize to JSON
        session_dict = session.model_dump(mode="json")
        return json.dumps(session_dict, indent=2, default=str)

    @mcp.resource("session://{session_id}/graph")
    async def get_session_graph(session_id: str) -> str:
        """Get thought graph visualization in mermaid format.

        This resource returns a mermaid graph definition that visualizes
        the thought graph structure. The graph shows:
        - Nodes representing individual thoughts with their content
        - Edges showing thought relationships (derives, supports, etc.)
        - Branch information via node grouping
        - Confidence scores via node labels

        The mermaid format can be rendered by tools that support mermaid
        diagrams (GitHub, many markdown renderers, etc.).

        Args:
            session_id: The unique identifier of the session

        Returns:
            Mermaid graph definition as a string

        Raises:
            ValueError: If session_id is not found

        Examples:
            Access via MCP client:
            >>> # MCP client request
            >>> resource = await client.read_resource("session://abc-123-def/graph")
            >>> mermaid_graph = resource.contents[0].text
            >>> # Render in markdown:
            >>> print(f"```mermaid\\n{mermaid_graph}\\n```")

            Example mermaid output:
            ```mermaid
            graph TD
                root["Initial: Starting analysis (conf: 0.80)"]
                child1["Continuation: Breaking down problem (conf: 0.75)"]
                child2["Branch: Alternative approach (conf: 0.70)"]
                root --> child1
                root --> child2
            ```
        """
        # Get app context from server
        from reasoning_mcp.server import get_app_context

        ctx: AppContext = get_app_context()

        # Retrieve session from manager
        session = await ctx.session_manager.get(session_id)

        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Generate mermaid graph from thought graph
        graph = session.graph
        mermaid_lines = ["graph TD"]

        # Add nodes with labels
        for node_id, node in graph.nodes.items():
            # Create a safe node identifier (mermaid doesn't like certain chars)
            safe_id = node_id.replace("-", "_")

            # Truncate content for readability
            content_preview = node.content[:50] + "..." if len(node.content) > 50 else node.content

            # Format node label with type, content preview, and confidence
            label = f"{node.type.value}: {content_preview} (conf: {node.confidence:.2f})"

            # Escape quotes in label
            label = label.replace('"', '\\"')

            # Add branch info if present
            branch_info = f" [branch: {node.branch_id}]" if node.branch_id else ""

            mermaid_lines.append(f'    {safe_id}["{label}{branch_info}"]')

        # Add edges with relationship types
        for _edge_id, edge in graph.edges.items():
            source_safe = edge.source_id.replace("-", "_")
            target_safe = edge.target_id.replace("-", "_")

            # Use different arrow styles for different edge types
            if edge.edge_type == "derives":
                arrow = "-->"
            elif edge.edge_type == "supports":
                arrow = "-.->'"
            elif edge.edge_type == "contradicts":
                arrow = "-.x"
            elif edge.edge_type == "branches":
                arrow = "==>"
            else:
                arrow = "-->"

            # Add edge with optional label
            if edge.edge_type != "derives":
                # Show non-default edge types as labels
                mermaid_lines.append(f"    {source_safe} {arrow}|{edge.edge_type}| {target_safe}")
            else:
                mermaid_lines.append(f"    {source_safe} {arrow} {target_safe}")

        # Handle empty graph case
        if len(graph.nodes) == 0:
            mermaid_lines.append('    empty["No thoughts in this session yet"]')

        return "\n".join(mermaid_lines)


__all__ = ["register_session_resources"]
