"""LLM Sampling utilities for reasoning-mcp.

This module provides sampling utilities that wrap FastMCP v2.14+'s ctx.sample()
and ctx.sample_step() functionality for use by reasoning methods.

FastMCP v2.14+ Sampling Features:
- ctx.sample(): Complete LLM sampling with message history
- ctx.sample_step(): Single-step sampling for multi-turn reasoning
- Pydantic result_type for structured outputs
- Support for OpenAI, Anthropic, and custom providers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

# Type alias for Context with Any type parameters
ContextType = "Context[Any, Any, Any]"

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class SamplingConfig:
    """Configuration for LLM sampling operations.

    Attributes:
        model: The model to use for sampling (e.g., "gpt-4o-mini", "claude-3-opus")
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
        system_prompt: Optional system prompt to prepend
        stop_sequences: Optional list of stop sequences
    """

    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: str | None = None
    stop_sequences: list[str] = field(default_factory=list)


async def sample_reasoning_step(
    ctx: Context[Any, Any, Any],
    prompt: str,
    *,
    config: SamplingConfig | None = None,
    result_type: type[T] | None = None,
    include_thinking: bool = False,
) -> T | str:
    """Sample a reasoning step using the LLM.

    This function uses FastMCP's ctx.sample() to generate LLM responses
    for reasoning steps. It supports both raw text and structured Pydantic outputs.

    Args:
        ctx: FastMCP Context with sampling capabilities
        prompt: The prompt for this reasoning step
        config: Optional sampling configuration
        result_type: Optional Pydantic model for structured output
        include_thinking: Whether to request chain-of-thought reasoning

    Returns:
        Either the Pydantic model instance if result_type is provided,
        or the raw string response

    Example:
        >>> class ReasoningStep(BaseModel):
        ...     thought: str
        ...     confidence: float
        ...     next_action: str
        ...
        >>> step = await sample_reasoning_step(
        ...     ctx,
        ...     "Analyze this problem: What is 2+2?",
        ...     result_type=ReasoningStep,
        ... )
        >>> print(step.thought)
    """
    if config is None:
        config = SamplingConfig()

    # Build the messages list
    messages: list[dict[str, Any]] = []

    if config.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": config.system_prompt,
            }
        )

    # Add thinking instruction if requested
    user_content = prompt
    if include_thinking:
        user_content = (
            f"{prompt}\n\nPlease think through this step-by-step before providing your answer."
        )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    try:
        # Use FastMCP's ctx.sample() for LLM completion
        # This is available in FastMCP v2.14+
        if result_type is not None:
            # Use structured output with Pydantic model
            result = await ctx.sample(  # type: ignore[attr-defined]
                messages=messages,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                result_type=result_type,
                stop_sequences=config.stop_sequences or None,
            )
            return result
        else:
            # Raw text response
            result = await ctx.sample(  # type: ignore[attr-defined]
                messages=messages,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                stop_sequences=config.stop_sequences or None,
            )
            return str(result)

    except AttributeError as e:
        # ctx.sample() not available - fall back to placeholder
        logger.warning(
            "ctx.sample() not available. Sampling requires FastMCP v2.14+ "
            "with a configured sampling handler. Returning placeholder."
        )
        if result_type is not None:
            # Create minimal instance of the Pydantic model
            # This is a fallback - should not happen in production
            raise RuntimeError(
                "Structured sampling requires FastMCP v2.14+ with sampling handler"
            ) from e
        return f"[Sampling not available - placeholder for: {prompt[:100]}...]"


async def sample_multi_step_reasoning(
    ctx: Context[Any, Any, Any],
    problem: str,
    steps: int = 3,
    *,
    config: SamplingConfig | None = None,
) -> list[str]:
    """Sample multiple reasoning steps for a problem.

    Uses ctx.sample_step() to maintain conversation history across
    multiple reasoning steps.

    Args:
        ctx: FastMCP Context with sampling capabilities
        problem: The problem to reason about
        steps: Number of reasoning steps to generate
        config: Optional sampling configuration

    Returns:
        List of reasoning step outputs
    """
    if config is None:
        config = SamplingConfig()

    results: list[str] = []

    try:
        # Initial step
        initial_prompt = (
            f"Let's reason through this problem step by step.\n\n"
            f"Problem: {problem}\n\n"
            f"Step 1: What is the first thing we should consider?"
        )

        # Use sample_step for multi-turn conversation
        step1 = await ctx.sample_step(  # type: ignore[attr-defined]
            messages=[{"role": "user", "content": initial_prompt}],
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        results.append(str(step1))

        # Continue with additional steps
        for i in range(2, steps + 1):
            continuation_prompt = (
                f"Good. Now for Step {i}: What should we consider next? "
                f"Build on your previous analysis."
            )

            step_result = await ctx.sample_step(  # type: ignore[attr-defined]
                messages=[{"role": "user", "content": continuation_prompt}],
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            results.append(str(step_result))

        return results

    except AttributeError:
        logger.warning(
            "ctx.sample_step() not available. Multi-step sampling requires "
            "FastMCP v2.14+ with a configured sampling handler."
        )
        # Return placeholder steps
        return [f"[Step {i + 1} placeholder for: {problem[:50]}...]" for i in range(steps)]


async def sample_with_tools(
    ctx: Context[Any, Any, Any],
    prompt: str,
    tools: list[Any] | None = None,
    *,
    system_prompt: str | None = None,
    result_type: type[T] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_iterations: int = 10,
) -> T | str:
    """Sample from LLM with tool access for agentic workflows.

    FastMCP v2.14+ feature (SEP-1577): The LLM can call provided tools
    and iterate until it produces a final response. This enables
    agentic reasoning patterns where the model decides which tools to use.

    Args:
        ctx: FastMCP Context with sampling capabilities
        prompt: The prompt for the LLM
        tools: List of tool functions/callables to make available
        system_prompt: Optional system prompt
        result_type: Optional Pydantic model for structured output
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum tokens (default 4096)
        max_iterations: Max tool iterations (default 10)

    Returns:
        Pydantic model instance if result_type provided, else string

    Example:
        >>> def calculator(expression: str) -> str:
        ...     return str(eval(expression))
        >>>
        >>> result = await sample_with_tools(
        ...     ctx,
        ...     "Calculate 15 * 23 and explain",
        ...     tools=[calculator],
        ... )
    """
    if tools is None:
        tools = []

    # Build messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Use FastMCP's ctx.sample() with tools parameter (v2.14+)
        result = await ctx.sample(  # type: ignore[attr-defined]
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            result_type=result_type,
            max_iterations=max_iterations,
        )

        if result_type is not None:
            return result
        return str(result) if not isinstance(result, str) else result

    except AttributeError:
        logger.warning(
            "ctx.sample() with tools not available. Requires FastMCP v2.14+ with sampling handler."
        )
        if result_type is not None:
            raise RuntimeError("Structured sampling with tools requires FastMCP v2.14+")
        return f"[Tool sampling not available - placeholder for: {prompt[:100]}...]"


class ReasoningPrompts:
    """Pre-built prompts for common reasoning patterns.

    These prompts are designed to work well with LLM sampling for
    various reasoning methods.
    """

    @staticmethod
    def chain_of_thought(problem: str) -> str:
        """Generate a chain-of-thought prompt."""
        return (
            f"Please analyze the following problem step by step, "
            f"showing your reasoning at each stage.\n\n"
            f"Problem: {problem}\n\n"
            f"Let's think through this carefully:"
        )

    @staticmethod
    def tree_of_thoughts(problem: str, branches: int = 3) -> str:
        """Generate a tree-of-thoughts prompt."""
        return (
            f"Consider the following problem and generate {branches} "
            f"distinct approaches to solving it.\n\n"
            f"Problem: {problem}\n\n"
            f"For each approach:\n"
            f"1. Describe the approach briefly\n"
            f"2. List the key steps\n"
            f"3. Identify potential challenges\n"
            f"4. Rate the approach's likelihood of success (1-10)"
        )

    @staticmethod
    def self_consistency(problem: str, solutions: list[str]) -> str:
        """Generate a self-consistency evaluation prompt."""
        solutions_text = "\n".join(f"Solution {i + 1}: {s}" for i, s in enumerate(solutions))
        return (
            f"Given the following problem and multiple proposed solutions, "
            f"determine the most consistent and correct answer.\n\n"
            f"Problem: {problem}\n\n"
            f"Proposed Solutions:\n{solutions_text}\n\n"
            f"Analyze these solutions and provide:\n"
            f"1. The most consistent answer\n"
            f"2. Your confidence level (0-1)\n"
            f"3. Brief explanation of why this answer is most consistent"
        )

    @staticmethod
    def reflection(problem: str, initial_solution: str) -> str:
        """Generate a reflection prompt for improving a solution."""
        return (
            f"Review and improve the following solution.\n\n"
            f"Problem: {problem}\n\n"
            f"Initial Solution: {initial_solution}\n\n"
            f"Please:\n"
            f"1. Identify any errors or weaknesses\n"
            f"2. Suggest improvements\n"
            f"3. Provide a refined solution"
        )

    @staticmethod
    def verification(problem: str, solution: str) -> str:
        """Generate a verification prompt."""
        return (
            f"Verify the correctness of the following solution.\n\n"
            f"Problem: {problem}\n\n"
            f"Proposed Solution: {solution}\n\n"
            f"Please:\n"
            f"1. Check each step of the solution\n"
            f"2. Identify any logical errors\n"
            f"3. Verify the final answer\n"
            f"4. Provide a confidence score (0-1)"
        )


__all__ = [
    "SamplingConfig",
    "sample_reasoning_step",
    "sample_multi_step_reasoning",
    "sample_with_tools",
    "ReasoningPrompts",
]
