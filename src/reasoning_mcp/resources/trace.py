"""MCP resources for session trace access.

This module provides MCP resource endpoints for accessing detailed execution
traces of reasoning sessions. Resources follow the URI pattern:
- trace://{session_id} - Returns detailed execution trace of a reasoning session
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from reasoning_mcp.server import AppContext


def register_trace_resources(mcp: FastMCP) -> None:
    """Register trace-related MCP resources with the server.

    This function registers a resource for accessing detailed session traces:
    1. trace://{session_id} - Returns execution trace with thought sequence,
       branching info, and final conclusions

    Args:
        mcp: The FastMCP server instance to register resources with

    Examples:
        >>> from mcp.server.fastmcp import FastMCP
        >>> mcp = FastMCP("test-server")
        >>> register_trace_resources(mcp)
    """

    @mcp.resource("trace://{session_id}")
    async def get_session_trace(session_id: str) -> str:
        """Get detailed execution trace of a reasoning session.

        This resource returns a comprehensive trace of the session's execution,
        including:
        - Session metadata (id, method, status, timestamps)
        - Thought sequence in chronological order with content and confidence
        - Branching information (branch points, active branches)
        - Final conclusions and outcomes
        - Timing information for performance analysis

        The trace is formatted as structured JSON suitable for analysis,
        debugging, and visualization tools.

        Args:
            session_id: The unique identifier of the session to trace

        Returns:
            JSON string containing the detailed execution trace

        Raises:
            ValueError: If session_id is not found

        Examples:
            Access via MCP client:
            >>> # MCP client request
            >>> resource = await client.read_resource("trace://abc-123-def")
            >>> trace_data = json.loads(resource.contents[0].text)
            >>> print(f"Session: {trace_data['session_id']}")
            >>> print(f"Total thoughts: {len(trace_data['thoughts'])}")
            >>> print(f"Status: {trace_data['metadata']['status']}")

            Example trace structure:
            {
                "session_id": "abc-123-def",
                "metadata": {
                    "method": "chain_of_thought",
                    "status": "completed",
                    "created_at": "2024-01-06T12:00:00",
                    "started_at": "2024-01-06T12:00:01",
                    "completed_at": "2024-01-06T12:05:30",
                    "duration_seconds": 329.5
                },
                "thoughts": [
                    {
                        "thought_number": 1,
                        "id": "thought-1",
                        "type": "initial",
                        "content": "Let's analyze the problem...",
                        "confidence": 0.8,
                        "depth": 0,
                        "created_at": "2024-01-06T12:00:01"
                    }
                ],
                "branches": {
                    "total_created": 2,
                    "total_merged": 1,
                    "total_pruned": 0,
                    "branch_points": [
                        {
                            "branch_id": "branch-1",
                            "parent_thought_id": "thought-3",
                            "created_at": "2024-01-06T12:02:15"
                        }
                    ]
                },
                "conclusions": [
                    {
                        "thought_id": "thought-15",
                        "content": "Final conclusion...",
                        "confidence": 0.95
                    }
                ]
            }
        """
        # Get app context from server
        from reasoning_mcp.server import get_app_context

        ctx: AppContext = get_app_context()

        # Retrieve session from manager
        session = await ctx.session_manager.get(session_id)

        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Build trace structure
        trace: dict[str, Any] = {
            "session_id": session.id,
            "metadata": {
                "method": str(session.current_method) if session.current_method else None,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "duration_seconds": session.duration,
                "error": session.error,
            },
            "config": {
                "max_depth": session.config.max_depth,
                "max_thoughts": session.config.max_thoughts,
                "timeout_seconds": session.config.timeout_seconds,
                "enable_branching": session.config.enable_branching,
                "max_branches": session.config.max_branches,
                "auto_prune": session.config.auto_prune,
                "min_confidence_threshold": session.config.min_confidence_threshold,
            },
            "metrics": {
                "total_thoughts": session.metrics.total_thoughts,
                "total_edges": session.metrics.total_edges,
                "max_depth_reached": session.metrics.max_depth_reached,
                "average_confidence": session.metrics.average_confidence,
                "average_quality": session.metrics.average_quality,
                "methods_used": session.metrics.methods_used,
                "thought_types": session.metrics.thought_types,
                "elapsed_time": session.metrics.elapsed_time,
            },
            "thoughts": [],
            "branches": {
                "total_created": session.metrics.branches_created,
                "total_merged": session.metrics.branches_merged,
                "total_pruned": session.metrics.branches_pruned,
                "branch_points": [],
            },
            "conclusions": [],
        }

        # Extract thoughts in chronological order
        sorted_thoughts = sorted(
            session.graph.nodes.values(),
            key=lambda n: n.created_at,
        )

        for idx, thought in enumerate(sorted_thoughts, start=1):
            thought_data = {
                "thought_number": idx,
                "id": thought.id,
                "type": thought.type.value,
                "method_id": str(thought.method_id),
                "content": thought.content,
                "summary": thought.summary,
                "evidence": thought.evidence,
                "confidence": thought.confidence,
                "quality_score": thought.quality_score,
                "is_valid": thought.is_valid,
                "depth": thought.depth,
                "step_number": thought.step_number,
                "parent_id": thought.parent_id,
                "children_ids": thought.children_ids,
                "branch_id": thought.branch_id,
                "created_at": thought.created_at.isoformat(),
            }
            trace["thoughts"].append(thought_data)

        # Identify branch points (thoughts that have a branch_id and are initial branch thoughts)
        branch_initiators = {}
        for thought in sorted_thoughts:
            if thought.branch_id and thought.branch_id not in branch_initiators:
                # This is the first thought in this branch
                branch_initiators[thought.branch_id] = {
                    "branch_id": thought.branch_id,
                    "thought_id": thought.id,
                    "parent_thought_id": thought.parent_id,
                    "created_at": thought.created_at.isoformat(),
                }

        trace["branches"]["branch_points"] = list(branch_initiators.values())

        # Identify conclusions (leaf nodes with high confidence or explicit conclusion type)
        leaf_ids = session.graph.leaf_ids
        for leaf_id in leaf_ids:
            if leaf_id in session.graph.nodes:
                thought = session.graph.nodes[leaf_id]
                # Include as conclusion if it's a leaf with reasonable confidence
                # or if it's explicitly marked as a conclusion type
                if thought.confidence >= 0.7 or thought.type.value == "conclusion":
                    trace["conclusions"].append(
                        {
                            "thought_id": thought.id,
                            "content": thought.content,
                            "confidence": thought.confidence,
                            "quality_score": thought.quality_score,
                            "branch_id": thought.branch_id,
                        }
                    )

        # If no high-confidence conclusions, include all leaf nodes
        if not trace["conclusions"] and leaf_ids:
            for leaf_id in leaf_ids:
                if leaf_id in session.graph.nodes:
                    thought = session.graph.nodes[leaf_id]
                    trace["conclusions"].append(
                        {
                            "thought_id": thought.id,
                            "content": thought.content,
                            "confidence": thought.confidence,
                            "quality_score": thought.quality_score,
                            "branch_id": thought.branch_id,
                        }
                    )

        # Convert to JSON with nice formatting
        return json.dumps(trace, indent=2, default=str)


__all__ = ["register_trace_resources"]
