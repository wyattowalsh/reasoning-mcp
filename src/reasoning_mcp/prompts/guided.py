"""MCP prompts for guided reasoning with reasoning-mcp.

This module provides MCP prompt endpoints for structured guidance on using
reasoning methods. Prompts help users apply specific methods effectively
or compare multiple methods for their problem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import PromptMessage, TextContent

from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.selector import MethodSelector, SelectionConstraint

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_guided_prompts(mcp: "FastMCP") -> None:
    """Register guided reasoning prompts with the server.

    This function registers two prompts:
    1. reason_with_method - Provides structured guidance for using a specific method
    2. compare_methods - Compares multiple methods for a problem

    Args:
        mcp: The FastMCP server instance to register prompts with

    Examples:
        >>> from mcp.server.fastmcp import FastMCP
        >>> mcp = FastMCP("test-server")
        >>> register_guided_prompts(mcp)
    """

    @mcp.prompt()
    async def reason_with_method(
        method_id: str,
        problem: str,
    ) -> list[PromptMessage]:
        """Generate a structured prompt for using a specific reasoning method.

        This prompt provides detailed guidance on applying a specific reasoning
        method to a problem, including:
        - Method description and capabilities
        - Step-by-step instructions for the method
        - Expected output format
        - Tips for effective use

        Args:
            method_id: The identifier of the reasoning method to use
                      (e.g., "chain_of_thought", "ethical_reasoning")
            problem: The problem or question to reason about

        Returns:
            List of PromptMessage objects containing the structured guidance

        Examples:
            >>> messages = await reason_with_method(
            ...     method_id="ethical_reasoning",
            ...     problem="Should we implement this feature?"
            ... )
        """
        # Get app context from server
        from reasoning_mcp.server import AppContext

        ctx: AppContext = mcp.app_context

        # Get method metadata
        metadata = ctx.registry.get_metadata(method_id)

        if metadata is None:
            # Method not found, provide fallback
            error_text = f"""Method '{method_id}' not found.

Available methods can be listed using the 'methods_list' tool.

For method recommendations, use 'methods_recommend' with your problem description."""

            return [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=error_text,
                    ),
                )
            ]

        # Build structured guidance
        guidance_parts = []

        # Header
        guidance_parts.append(f"# Reasoning with {metadata.name}")
        guidance_parts.append(f"\n**Method ID:** `{metadata.identifier}`")
        guidance_parts.append(f"\n**Category:** {metadata.category}")
        guidance_parts.append(f"\n**Complexity:** {metadata.complexity}/10")

        # Description
        guidance_parts.append(f"\n\n## Description\n\n{metadata.description}")

        # When to use
        if metadata.best_for:
            guidance_parts.append("\n\n## Best Used For")
            for use_case in metadata.best_for:
                guidance_parts.append(f"\n- {use_case}")

        # When not to use
        if metadata.not_recommended_for:
            guidance_parts.append("\n\n## Not Recommended For")
            for use_case in metadata.not_recommended_for:
                guidance_parts.append(f"\n- {use_case}")

        # Capabilities
        guidance_parts.append("\n\n## Capabilities")
        guidance_parts.append(f"\n- Supports branching: {'Yes' if metadata.supports_branching else 'No'}")
        guidance_parts.append(f"\n- Supports revision: {'Yes' if metadata.supports_revision else 'No'}")
        guidance_parts.append(
            f"\n- Expected depth: {metadata.min_thoughts}-{metadata.max_thoughts} thoughts"
        )

        # Step-by-step instructions
        guidance_parts.append("\n\n## How to Use This Method")
        guidance_parts.append("\n\n1. **Start the reasoning session**")
        guidance_parts.append(f"\n   - Use the `reason` tool with `method=\"{method_id}\"`")
        guidance_parts.append(f"\n   - Provide your problem: `{problem[:100]}{'...' if len(problem) > 100 else ''}`")

        guidance_parts.append("\n\n2. **Follow the method's structure**")
        if method_id == "chain_of_thought":
            guidance_parts.append("\n   - Break down the problem into clear, logical steps")
            guidance_parts.append("\n   - Show your work at each step")
            guidance_parts.append("\n   - Build each step on previous conclusions")
        elif method_id == "tree_of_thoughts":
            guidance_parts.append("\n   - Generate multiple possible approaches")
            guidance_parts.append("\n   - Evaluate each branch for promise")
            guidance_parts.append("\n   - Prune less promising paths")
            guidance_parts.append("\n   - Explore the most promising directions")
        elif method_id == "ethical_reasoning":
            guidance_parts.append("\n   - Identify stakeholders and their interests")
            guidance_parts.append("\n   - Apply multiple ethical frameworks (utilitarian, deontological, virtue ethics)")
            guidance_parts.append("\n   - Consider short and long-term consequences")
            guidance_parts.append("\n   - Synthesize insights across frameworks")
        elif method_id == "react":
            guidance_parts.append("\n   - Alternate between reasoning (Thought) and acting (Action)")
            guidance_parts.append("\n   - After each action, observe the result")
            guidance_parts.append("\n   - Use observations to inform next thought")
            guidance_parts.append("\n   - Iterate until the problem is solved")
        else:
            guidance_parts.append("\n   - Follow the natural flow of this method")
            guidance_parts.append(f"\n   - Expect {metadata.min_thoughts}-{metadata.max_thoughts} reasoning steps")
            if metadata.supports_branching:
                guidance_parts.append("\n   - Use branching when exploring alternatives")
            if metadata.supports_revision:
                guidance_parts.append("\n   - Revise earlier thoughts as understanding deepens")

        guidance_parts.append("\n\n3. **Continue the reasoning**")
        guidance_parts.append("\n   - Use `session_continue` to advance the reasoning")
        if metadata.supports_branching:
            guidance_parts.append("\n   - Use `session_branch` to explore alternatives")
        if metadata.supports_revision:
            guidance_parts.append("\n   - Revise previous thoughts if needed")

        guidance_parts.append("\n\n4. **Monitor progress**")
        guidance_parts.append("\n   - Use `session_inspect` to view current state")
        guidance_parts.append("\n   - Check thought graph with the session resource")
        guidance_parts.append("\n   - Ensure reasoning depth is appropriate")

        # Expected output format
        guidance_parts.append("\n\n## Expected Output Format")
        guidance_parts.append("\n\nEach reasoning step will produce:")
        guidance_parts.append("\n- **Thought content**: The actual reasoning for this step")
        guidance_parts.append("\n- **Confidence**: How confident the reasoning is (0.0-1.0)")
        guidance_parts.append("\n- **Type**: The type of thought (initial, continuation, branch, etc.)")
        guidance_parts.append("\n- **Metadata**: Additional context about the thought")

        # Tags for context
        if metadata.tags:
            guidance_parts.append("\n\n## Related Tags")
            guidance_parts.append(f"\n{', '.join(metadata.tags)}")

        # Your problem
        guidance_parts.append("\n\n## Your Problem")
        guidance_parts.append(f"\n{problem}")

        guidance_parts.append("\n\n## Next Steps")
        guidance_parts.append(f"\n\nTo begin reasoning with {metadata.name}:")
        guidance_parts.append(f"\n```python\nreason(problem=\"{problem[:80]}...\", method=\"{method_id}\")\n```")

        guidance_text = "".join(guidance_parts)

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=guidance_text,
                ),
            )
        ]

    @mcp.prompt()
    async def compare_methods(
        problem: str,
        method_ids: list[str] | None = None,
    ) -> list[PromptMessage]:
        """Generate a prompt comparing how different methods would approach a problem.

        This prompt provides a comparative analysis of multiple reasoning methods,
        helping you understand which method(s) might be most suitable for your
        specific problem. If method_ids are not provided, automatically selects
        the top 3 recommended methods.

        Args:
            problem: The problem or question to analyze
            method_ids: Optional list of method identifiers to compare.
                       If None, auto-selects top 3 recommendations.

        Returns:
            List of PromptMessage objects containing the comparative analysis

        Examples:
            >>> # Auto-select top 3 methods
            >>> messages = await compare_methods(
            ...     problem="Should we implement this privacy feature?"
            ... )

            >>> # Compare specific methods
            >>> messages = await compare_methods(
            ...     problem="Debug this race condition",
            ...     method_ids=["chain_of_thought", "react", "code_reasoning"]
            ... )
        """
        # Get app context from server
        from reasoning_mcp.server import AppContext

        ctx: AppContext = mcp.app_context

        # Create selector for recommendations
        selector = MethodSelector(ctx.registry)

        # If method_ids not provided, auto-select top 3
        if method_ids is None:
            recommendations = selector.recommend(
                problem=problem,
                max_recommendations=3,
            )
            method_ids = [rec.identifier for rec in recommendations]

            if not method_ids:
                # No recommendations available
                fallback_text = """No suitable methods could be recommended for this problem.

This might mean:
- The problem description is too vague
- The registry is empty (no methods registered yet)

Try:
1. Using `methods_list` to see available methods
2. Providing a more detailed problem description
3. Specifying method_ids explicitly"""

                return [
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=fallback_text,
                        ),
                    )
                ]

        # Get metadata for all methods
        method_metadata = []
        for method_id in method_ids:
            metadata = ctx.registry.get_metadata(method_id)
            if metadata:
                method_metadata.append(metadata)

        if not method_metadata:
            error_text = f"""None of the specified methods were found: {', '.join(method_ids)}

Use `methods_list` to see available methods."""

            return [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=error_text,
                    ),
                )
            ]

        # Get recommendations for scoring
        constraints = SelectionConstraint(
            allowed_methods=frozenset(method_ids),
        )
        recommendations = selector.recommend(
            problem=problem,
            constraints=constraints,
            max_recommendations=len(method_ids),
        )

        # Build scores dict
        scores: dict[str, float] = {}
        reasoning_map: dict[str, str] = {}
        for rec in recommendations:
            scores[rec.identifier] = rec.score
            reasoning_map[rec.identifier] = rec.reasoning

        # Build comparison text
        comparison_parts = []

        # Header
        comparison_parts.append("# Reasoning Method Comparison")
        comparison_parts.append(f"\n\n**Problem:** {problem}")
        comparison_parts.append(f"\n\n**Methods being compared:** {len(method_metadata)}")

        # Overview table
        comparison_parts.append("\n\n## Quick Comparison\n")
        comparison_parts.append("\n| Method | Category | Complexity | Score | Best For |")
        comparison_parts.append("\n|--------|----------|------------|-------|----------|")

        # Sort by score
        sorted_metadata = sorted(
            method_metadata,
            key=lambda m: scores.get(str(m.identifier), 0.0),
            reverse=True,
        )

        for metadata in sorted_metadata:
            method_id = str(metadata.identifier)
            score = scores.get(method_id, 0.0)
            best_for = metadata.best_for[0] if metadata.best_for else "General purpose"
            comparison_parts.append(
                f"\n| {metadata.name} | {metadata.category} | "
                f"{metadata.complexity}/10 | {score:.2f} | {best_for} |"
            )

        # Detailed analysis for each method
        comparison_parts.append("\n\n## Detailed Analysis\n")

        for i, metadata in enumerate(sorted_metadata, 1):
            method_id = str(metadata.identifier)
            score = scores.get(method_id, 0.0)

            comparison_parts.append(f"\n### {i}. {metadata.name} (Score: {score:.2f})")
            comparison_parts.append(f"\n**ID:** `{metadata.identifier}`")
            comparison_parts.append(f"\n\n{metadata.description}")

            # Why recommended/not recommended
            if method_id in reasoning_map:
                comparison_parts.append(f"\n\n**Why this score:** {reasoning_map[method_id]}")

            # Strengths
            comparison_parts.append("\n\n**Strengths:**")
            for strength in metadata.best_for[:3]:
                comparison_parts.append(f"\n- {strength}")

            # Limitations
            if metadata.not_recommended_for:
                comparison_parts.append("\n\n**Limitations:**")
                for limitation in metadata.not_recommended_for[:2]:
                    comparison_parts.append(f"\n- {limitation}")

            # Method-specific approach
            comparison_parts.append("\n\n**How it would approach this problem:**")

            if metadata.identifier == "chain_of_thought":
                comparison_parts.append(
                    "\n- Break the problem into sequential logical steps"
                    "\n- Show explicit reasoning at each step"
                    "\n- Build conclusions progressively"
                )
            elif metadata.identifier == "tree_of_thoughts":
                comparison_parts.append(
                    "\n- Generate multiple solution approaches"
                    "\n- Evaluate and compare alternatives"
                    "\n- Explore most promising paths deeply"
                )
            elif metadata.identifier == "ethical_reasoning":
                comparison_parts.append(
                    "\n- Identify all stakeholders and their interests"
                    "\n- Apply multiple ethical frameworks"
                    "\n- Consider consequences and principles"
                    "\n- Synthesize balanced recommendation"
                )
            elif metadata.identifier == "react":
                comparison_parts.append(
                    "\n- Interleave thinking with concrete actions"
                    "\n- Observe results after each action"
                    "\n- Adjust approach based on observations"
                )
            elif metadata.identifier == "code_reasoning":
                comparison_parts.append(
                    "\n- Analyze code structure and patterns"
                    "\n- Trace execution flow and state"
                    "\n- Identify potential issues or improvements"
                )
            else:
                comparison_parts.append(
                    f"\n- Use {metadata.name.lower()} techniques"
                    f"\n- Expected depth: {metadata.min_thoughts}-{metadata.max_thoughts} thoughts"
                )

            # Capabilities
            capabilities = []
            if metadata.supports_branching:
                capabilities.append("branching")
            if metadata.supports_revision:
                capabilities.append("revision")

            if capabilities:
                comparison_parts.append(f"\n\n**Supports:** {', '.join(capabilities)}")

            comparison_parts.append("\n")

        # Recommendation
        comparison_parts.append("\n## Recommendation\n")

        if sorted_metadata:
            top_method = sorted_metadata[0]
            top_score = scores.get(str(top_method.identifier), 0.0)

            if top_score > 0.5:
                comparison_parts.append(
                    f"\n**Recommended:** {top_method.name} has the highest score ({top_score:.2f}) "
                    "and appears well-suited for this problem."
                )
            elif top_score > 0.3:
                comparison_parts.append(
                    f"\n**Consider:** {top_method.name} (score: {top_score:.2f}) is a reasonable choice, "
                    "though other methods may also work."
                )
            else:
                comparison_parts.append(
                    "\n**Note:** None of these methods are particularly well-suited for this problem. "
                    "You may want to try `methods_recommend` to find better alternatives."
                )

            # Show how scores compare
            if len(sorted_metadata) > 1:
                second_method = sorted_metadata[1]
                second_score = scores.get(str(second_method.identifier), 0.0)
                gap = top_score - second_score

                if gap < 0.1:
                    comparison_parts.append(
                        f"\n\nThe scores are very close. {second_method.name} "
                        f"(score: {second_score:.2f}) is a competitive alternative."
                    )

        # Next steps
        comparison_parts.append("\n\n## Next Steps\n")
        comparison_parts.append("\nTo use the recommended method:")

        if sorted_metadata:
            top_id = str(sorted_metadata[0].identifier)
            comparison_parts.append(
                f"\n```python\nreason(problem=\"{problem[:80]}...\", method=\"{top_id}\")\n```"
            )

        comparison_parts.append(
            "\n\nTo explore how a specific method works:\n"
            "```python\n"
            "reason_with_method(method_id=\"METHOD_ID\", problem=\"YOUR_PROBLEM\")\n"
            "```"
        )

        comparison_text = "".join(comparison_parts)

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=comparison_text,
                ),
            )
        ]


__all__ = ["register_guided_prompts"]
