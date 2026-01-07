"""Primary reasoning tool for reasoning-mcp.

This module provides the main `reason` tool for initiating and continuing
reasoning processes. The tool supports automatic method selection and
manual method specification.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import ReasonHints, ReasonOutput
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.selector import MethodSelector


async def reason(
    problem: str,
    method: str | None = None,
    hints: ReasonHints | None = None,
) -> ReasonOutput:
    """Generate reasoning for a problem using specified or auto-selected method.

    This is the primary tool for initiating reasoning processes. It accepts a
    problem statement and either uses the specified method or automatically
    selects the most appropriate method based on problem analysis.

    The tool can:
    - Auto-select the best reasoning method if none is specified
    - Create a new reasoning session if needed
    - Generate the initial thought for the problem
    - Provide suggestions for next steps

    Args:
        problem: The problem or question to reason about. Should be a clear
            statement of what needs to be analyzed or solved.
        method: Optional method identifier to use (e.g., "chain_of_thought",
            "ethical_reasoning"). If None, the best method is automatically
            selected based on problem analysis.
        hints: Optional hints to guide method selection. Only used when method
            is None. Provides domain, complexity, and preference information
            to improve auto-selection.

    Returns:
        ReasonOutput containing:
        - session_id: UUID of the reasoning session (new or existing)
        - thought: The initial generated thought
        - method_used: The method that was applied
        - suggestions: Recommended next steps
        - metadata: Additional information about the reasoning process

    Examples:
        Auto-select method for ethical problem:
        >>> output = await reason(
        ...     "Should we implement this feature that might compromise user privacy?"
        ... )
        >>> assert output.method_used == MethodIdentifier.ETHICAL_REASONING
        >>> assert output.thought.type == ThoughtType.INITIAL

        Use specific method:
        >>> output = await reason(
        ...     "Calculate the optimal solution to this optimization problem",
        ...     method="mathematical_reasoning"
        ... )
        >>> assert output.method_used == MethodIdentifier.MATHEMATICAL_REASONING

        Use hints for better auto-selection:
        >>> hints = ReasonHints(
        ...     domain="code",
        ...     complexity="high",
        ...     prefer_methods=[MethodIdentifier.CODE_REASONING]
        ... )
        >>> output = await reason(
        ...     "Debug this complex async race condition",
        ...     hints=hints
        ... )
        >>> assert output.method_used == MethodIdentifier.CODE_REASONING

    Raises:
        ValueError: If the specified method doesn't exist or is invalid
        RuntimeError: If reasoning engine encounters an error
    """
    # Get the global registry
    # TODO: In production, this should be injected via AppContext
    registry = MethodRegistry()

    # Determine which method to use
    if method is None:
        # Auto-select best method
        selector = MethodSelector(registry)

        # Convert hints to constraints if provided
        constraints = None
        if hints:
            from reasoning_mcp.selector import SelectionConstraint

            constraints = SelectionConstraint(
                preferred_methods=frozenset(str(m) for m in hints.prefer_methods),
                excluded_methods=frozenset(str(m) for m in hints.avoid_methods),
            )

        # Select best method
        selected_method = selector.select_best(
            problem=problem,
            constraints=constraints,
        )

        if selected_method is None:
            # Fallback to chain_of_thought if no good match
            selected_method = "chain_of_thought"
            method_identifier = MethodIdentifier.CHAIN_OF_THOUGHT
        else:
            method_identifier = MethodIdentifier(selected_method)
    else:
        # Use specified method
        method_identifier = MethodIdentifier(method)

        # Validate that method exists (when registry has methods)
        # For now, we allow any valid MethodIdentifier since methods aren't implemented yet
        # TODO: Remove this check bypass once methods are registered
        metadata = registry.get_metadata(str(method_identifier))
        if metadata is None and len(registry.list_methods()) > 0:
            # Only raise error if registry has methods but this one isn't found
            raise ValueError(f"Method '{method}' not found in registry")

    # Create a new session
    # TODO: This should use the SessionManager from AppContext
    session_id = str(uuid4())

    # Generate initial thought
    # TODO: This should invoke the actual reasoning method
    # For now, create a placeholder thought
    problem_preview = problem[:200]
    if len(problem) > 200:
        problem_preview += "..."

    thought = ThoughtNode(
        id=str(uuid4()),
        type=ThoughtType.INITIAL,
        method_id=method_identifier,
        content=f"Initial analysis of the problem: {problem_preview}",
        confidence=0.7,
        step_number=1,
        depth=0,
        created_at=datetime.now(),
        metadata={
            "problem": problem,
            "auto_selected": method is None,
        },
    )

    # Generate suggestions for next steps
    suggestions = [
        "Continue reasoning with session_continue",
        "Explore alternative approaches with session_branch",
        "Inspect current reasoning state with session_inspect",
    ]

    # Build metadata
    metadata = {
        "auto_selected": method is None,
        "hints_provided": hints is not None,
        "problem_length": len(problem),
    }

    if hints:
        metadata["hints"] = {
            "domain": hints.domain,
            "complexity": hints.complexity,
            "preferred_methods": [str(m) for m in hints.prefer_methods],
            "avoided_methods": [str(m) for m in hints.avoid_methods],
        }

    return ReasonOutput(
        session_id=session_id,
        thought=thought,
        method_used=method_identifier,
        suggestions=suggestions,
        metadata=metadata,
    )


__all__ = ["reason"]
