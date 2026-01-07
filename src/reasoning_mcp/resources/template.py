"""MCP resources for predefined reasoning pipeline templates.

This module provides MCP resources that expose predefined pipeline templates
for common reasoning patterns. Templates enable users to quickly bootstrap
complex reasoning workflows without manually constructing pipelines.

Resource URIs:
- template://{template_id} - Returns predefined pipeline template as JSON

Available Templates:
- deep_analysis: Sequential analyze → synthesize → conclude
- debate: Dialectic thesis → antithesis → synthesis
- verify: Self-consistency with multiple reasoning paths
- debug: Code reasoning with ReAct loop
- brainstorm: Parallel creative exploration with multiple methods
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from reasoning_mcp.models.core import MethodIdentifier
from reasoning_mcp.models.pipeline import (
    Accumulator,
    Condition,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    SequencePipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ============================================================================
# Template Definitions
# ============================================================================


def _create_deep_analysis_template() -> dict[str, Any]:
    """Create deep analysis template: analyze → synthesize → conclude.

    This template provides a structured three-stage analytical workflow:
    1. Initial analysis using Chain of Thought
    2. Synthesis using Self-Reflection to critique and refine
    3. Final conclusion using Sequential Thinking to summarize

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="deep_analysis",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="analyze",
                description="Initial analytical reasoning to break down the problem",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="synthesize",
                description="Reflect on and synthesize the initial analysis",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.SEQUENTIAL_THINKING,
                name="conclude",
                description="Generate final conclusion based on synthesis",
                max_thoughts=5,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "deep_analysis",
            "use_case": "Thorough analysis requiring structured reasoning",
            "estimated_thoughts": 30,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_debate_template() -> dict[str, Any]:
    """Create dialectic debate template: thesis → antithesis → synthesis.

    This template implements a dialectical reasoning pattern:
    1. Generate initial thesis using Chain of Thought
    2. Challenge with antithesis using Dialectic reasoning
    3. Synthesize both perspectives into unified conclusion

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="debate",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="thesis",
                description="Develop initial position or hypothesis",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.DIALECTIC,
                name="antithesis",
                description="Challenge and critique the initial position",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="synthesis",
                description="Synthesize thesis and antithesis into balanced conclusion",
                max_thoughts=8,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "debate",
            "use_case": "Exploring multiple perspectives and finding balanced solutions",
            "estimated_thoughts": 30,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_verify_template() -> dict[str, Any]:
    """Create verification template using self-consistency.

    This template uses parallel reasoning paths to verify conclusions:
    1. Generate multiple independent reasoning paths in parallel
    2. Merge results using voting/consensus strategy
    3. Verify consistency across paths

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = ParallelPipeline(
        name="verify",
        branches=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="path_1",
                description="First independent reasoning path",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="path_2",
                description="Second independent reasoning path",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="path_3",
                description="Third independent reasoning path",
                max_thoughts=10,
            ),
        ],
        merge_strategy=MergeStrategy(
            name="vote",
            selection_criteria="most_common_conclusion",
            metadata={
                "min_agreement": 0.6,
                "description": "Select conclusion with majority agreement",
            },
        ),
        max_concurrency=3,
        timeout_seconds=180.0,
        metadata={
            "template_id": "verify",
            "use_case": "High-confidence answers through self-consistency",
            "estimated_thoughts": 30,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_debug_template() -> dict[str, Any]:
    """Create code debugging template with ReAct loop.

    This template implements an iterative debugging workflow:
    1. Initial code analysis
    2. Loop with ReAct (reason and act) until solution found
    3. Final verification

    Returns:
        Dictionary representation of the pipeline template
    """
    loop_body = SequencePipeline(
        name="debug_iteration",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.REACT,
                name="reason_and_act",
                description="Reason about the code and identify actions to take",
                max_thoughts=8,
            ),
            MethodStage(
                method_id=MethodIdentifier.CODE_REASONING,
                name="analyze_code",
                description="Detailed code analysis and reasoning",
                max_thoughts=10,
            ),
        ],
    )

    pipeline = SequencePipeline(
        name="debug",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CODE_REASONING,
                name="initial_analysis",
                description="Initial code inspection and problem identification",
                max_thoughts=12,
            ),
            LoopPipeline(
                name="iterative_debugging",
                body=loop_body,
                condition=Condition(
                    name="solution_found",
                    expression="is_solved == False",
                    operator="==",
                    field="is_solved",
                    metadata={
                        "description": "Continue until solution is found",
                    },
                ),
                max_iterations=5,
                accumulator=Accumulator(
                    name="debug_history",
                    initial_value=[],
                    operation="append",
                    field="content",
                    metadata={
                        "description": "Collect debugging attempts for final summary",
                    },
                ),
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="verify_solution",
                description="Verify and validate the proposed solution",
                max_thoughts=5,
            ),
        ],
        stop_on_error=False,
        metadata={
            "template_id": "debug",
            "use_case": "Iterative code debugging and problem solving",
            "estimated_thoughts": "variable (up to ~100)",
        },
    )
    return pipeline.model_dump(mode="json")


def _create_brainstorm_template() -> dict[str, Any]:
    """Create brainstorming template with parallel creative exploration.

    This template enables creative exploration through multiple methods:
    1. Parallel execution of diverse reasoning approaches
    2. Lateral thinking for creative solutions
    3. Analogical reasoning for cross-domain insights
    4. Tree of Thoughts for exploring solution space
    5. Merge all perspectives into comprehensive results

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = ParallelPipeline(
        name="brainstorm",
        branches=[
            MethodStage(
                method_id=MethodIdentifier.LATERAL_THINKING,
                name="creative_path",
                description="Explore creative and unconventional solutions",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.ANALOGICAL,
                name="analogy_path",
                description="Find solutions through analogies and similar problems",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                name="exploration_path",
                description="Systematically explore the solution space",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.STEP_BACK,
                name="abstraction_path",
                description="Step back to find higher-level principles",
                max_thoughts=8,
            ),
        ],
        merge_strategy=MergeStrategy(
            name="combine",
            selection_criteria="all_results",
            aggregation="synthesize",
            metadata={
                "description": "Combine all creative perspectives into comprehensive insights",
            },
        ),
        max_concurrency=4,
        timeout_seconds=240.0,
        metadata={
            "template_id": "brainstorm",
            "use_case": "Creative problem solving and idea generation",
            "estimated_thoughts": 45,
        },
    )
    return pipeline.model_dump(mode="json")


# ============================================================================
# Template Registry
# ============================================================================

_TEMPLATES: dict[str, Callable[[], dict[str, Any]]] = {
    "deep_analysis": _create_deep_analysis_template,
    "debate": _create_debate_template,
    "verify": _create_verify_template,
    "debug": _create_debug_template,
    "brainstorm": _create_brainstorm_template,
}


def get_available_templates() -> list[str]:
    """Get list of available template IDs.

    Returns:
        List of template identifiers
    """
    return sorted(_TEMPLATES.keys())


def get_template(template_id: str) -> dict[str, Any] | None:
    """Get a template by ID.

    Args:
        template_id: The template identifier

    Returns:
        Template definition as dictionary, or None if not found
    """
    if template_id not in _TEMPLATES:
        return None
    return _TEMPLATES[template_id]()


# ============================================================================
# MCP Resource Registration
# ============================================================================


def register_template_resources(mcp: FastMCP) -> None:
    """Register template-related MCP resources with the server.

    This function registers the template resource endpoint:
    - template://{template_id} - Returns predefined pipeline template

    Args:
        mcp: The FastMCP server instance to register resources with

    Example:
        >>> from reasoning_mcp.server import mcp
        >>> register_template_resources(mcp)
    """

    @mcp.resource("template://{template_id}")
    async def get_pipeline_template(template_id: str) -> str:
        """Get a predefined pipeline template by ID.

        This resource returns a complete pipeline definition for common
        reasoning patterns. The template can be used directly with the
        compose tool or modified to fit specific needs.

        Args:
            template_id: The template identifier (e.g., "deep_analysis", "debug")

        Returns:
            JSON string containing the complete pipeline definition

        Resource URI:
            template://{template_id}

        Available Templates:
            - deep_analysis: Sequential analyze → synthesize → conclude
            - debate: Dialectic thesis → antithesis → synthesis
            - verify: Self-consistency with multiple reasoning paths
            - debug: Code reasoning with ReAct loop
            - brainstorm: Parallel creative exploration

        Example Response:
            {
                "id": "pipeline-uuid",
                "stage_type": "sequence",
                "name": "deep_analysis",
                "stages": [
                    {
                        "id": "stage-uuid-1",
                        "stage_type": "method",
                        "method_id": "chain_of_thought",
                        "name": "analyze",
                        "max_thoughts": 15
                    },
                    ...
                ],
                "metadata": {
                    "template_id": "deep_analysis",
                    "use_case": "Thorough analysis requiring structured reasoning",
                    "estimated_thoughts": 30
                }
            }

        Raises:
            ValueError: If template_id is not found
        """
        # Get template definition
        template = get_template(template_id)

        if template is None:
            available = get_available_templates()
            logger.warning(f"Template not found: {template_id}. Available: {available}")
            raise ValueError(
                f"Template '{template_id}' not found. Available templates: {', '.join(available)}"
            )

        logger.debug(f"Retrieved template: {template_id}")

        # Add metadata about available templates to help discovery
        template["_available_templates"] = {
            tid: _TEMPLATES[tid]().get("metadata", {}).get("use_case", "")
            for tid in get_available_templates()
        }

        return json.dumps(template, indent=2)

    logger.info("Registered template resource: template://{template_id}")


__all__ = [
    "register_template_resources",
    "get_available_templates",
    "get_template",
]
