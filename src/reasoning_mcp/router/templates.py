"""Pipeline templates for the Reasoning Router.

This module defines pre-built pipeline templates that can be selected
by the router for specific problem types.

Templates are defined using proper Pydantic pipeline models to ensure
schema compatibility with the pipeline execution engine.
"""

from __future__ import annotations

from typing import Any

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


def _create_verified_reasoning_template() -> dict[str, Any]:
    """Create verified reasoning template: CoT → CoVe.

    This template provides reasoning with verification:
    1. Initial analysis using Chain of Thought
    2. Verification using Chain of Verification

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="verified_reasoning",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="initial_reasoning",
                description="Initial analytical reasoning to solve the problem",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
                name="verify",
                description="Verify the reasoning through systematic checking",
                max_thoughts=8,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "verified_reasoning",
            "best_for": ["high-stakes decisions", "factual accuracy", "verification"],
            "description": "Chain of thought with verification",
            "estimated_thoughts": 18,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_iterative_improve_template() -> dict[str, Any]:
    """Create iterative improvement template with loop.

    This template implements iterative refinement:
    1. Initial reasoning with Chain of Thought
    2. Loop: refine until quality threshold met

    Returns:
        Dictionary representation of the pipeline template
    """
    loop_body = SequencePipeline(
        name="improve_iteration",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="reason",
                description="Continue reasoning about the problem",
                max_thoughts=5,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFINE,
                name="refine",
                description="Refine and improve the reasoning",
                max_thoughts=5,
            ),
        ],
    )

    pipeline = LoopPipeline(
        name="iterative_improve",
        body=loop_body,
        condition=Condition(
            name="quality_check",
            expression="quality_score < 0.9",
            operator="<",
            threshold=0.9,
            field="quality_score",
            metadata={"description": "Continue until quality threshold is met"},
        ),
        max_iterations=3,
        accumulator=Accumulator(
            name="improvement_history",
            initial_value=[],
            operation="append",
            field="content",
            metadata={"description": "Track improvement iterations"},
        ),
        metadata={
            "template_id": "iterative_improve",
            "best_for": ["quality improvement", "iterative refinement"],
            "description": "Loop of reasoning and refinement until quality threshold",
            "estimated_thoughts": 30,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_analyze_refine_template() -> dict[str, Any]:
    """Create analyze and refine template: CoT → Self-Refine.

    This template provides deep analysis followed by refinement:
    1. Analytical reasoning with Chain of Thought
    2. Self-refinement for clarity

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="analyze_refine",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="analyze",
                description="Deep analytical reasoning",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFINE,
                name="refine",
                description="Refine and clarify the analysis",
                max_thoughts=8,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "analyze_refine",
            "best_for": ["analytical problems", "clarity improvement"],
            "description": "Deep analysis followed by refinement",
            "estimated_thoughts": 20,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_ethical_multi_view_template() -> dict[str, Any]:
    """Create ethical multi-view template with parallel perspectives.

    This template analyzes ethical issues from multiple perspectives:
    1. Parallel ethical reasoning, dialectic, and socratic analysis
    2. Synthesis of all perspectives

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="ethical_multi_view",
        stages=[
            ParallelPipeline(
                name="multi_perspective",
                branches=[
                    MethodStage(
                        method_id=MethodIdentifier.ETHICAL_REASONING,
                        name="ethical_analysis",
                        description="Ethical framework analysis",
                        max_thoughts=10,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.DIALECTIC,
                        name="dialectic_analysis",
                        description="Dialectic exploration of tensions",
                        max_thoughts=10,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.SOCRATIC,
                        name="socratic_questioning",
                        description="Socratic questioning of assumptions",
                        max_thoughts=10,
                    ),
                ],
                merge_strategy=MergeStrategy(
                    name="collect",
                    selection_criteria="all_results",
                    aggregation="synthesize",
                ),
                max_concurrency=3,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="synthesize",
                description="Synthesize all perspectives into balanced conclusion",
                max_thoughts=10,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "ethical_multi_view",
            "best_for": ["ethical dilemmas", "moral reasoning", "stakeholder analysis"],
            "description": "Multi-perspective ethical analysis with synthesis",
            "estimated_thoughts": 40,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_math_proof_template() -> dict[str, Any]:
    """Create mathematical proof template.

    This template provides rigorous mathematical reasoning:
    1. Step back for principles
    2. Mathematical reasoning
    3. Logical verification

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="math_proof",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.STEP_BACK,
                name="identify_principles",
                description="Identify relevant mathematical principles",
                max_thoughts=8,
            ),
            MethodStage(
                method_id=MethodIdentifier.MATHEMATICAL_REASONING,
                name="formal_proof",
                description="Develop formal mathematical proof",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.LOGIC_OF_THOUGHT,
                name="logical_verification",
                description="Verify logical consistency",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
                name="final_check",
                description="Final verification of proof",
                max_thoughts=5,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "math_proof",
            "best_for": ["proofs", "mathematical problems", "formal logic"],
            "description": "Rigorous mathematical reasoning with verification",
            "estimated_thoughts": 38,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_debug_code_template() -> dict[str, Any]:
    """Create code debugging template with ReAct loop.

    This template implements iterative debugging:
    1. Initial code analysis
    2. ReAct reasoning and action
    3. Verification of fix

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="debug_code",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CODE_REASONING,
                name="initial_analysis",
                description="Initial code inspection and problem identification",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.REACT,
                name="reason_and_act",
                description="Reason about the code and identify actions",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_VERIFICATION,
                name="verify_fix",
                description="Verify the proposed solution",
                max_thoughts=8,
            ),
        ],
        stop_on_error=False,
        metadata={
            "template_id": "debug_code",
            "best_for": ["debugging", "code fixes", "error resolution"],
            "description": "Code analysis with ReAct execution and verification",
            "estimated_thoughts": 35,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_creative_explore_template() -> dict[str, Any]:
    """Create creative exploration template.

    This template enables creative problem solving through multiple approaches:
    1. Parallel creative exploration
    2. Selection of best ideas

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="creative_explore",
        stages=[
            ParallelPipeline(
                name="creative_paths",
                branches=[
                    MethodStage(
                        method_id=MethodIdentifier.LATERAL_THINKING,
                        name="lateral",
                        description="Explore unconventional solutions",
                        max_thoughts=10,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                        name="exploration",
                        description="Explore solution space systematically",
                        max_thoughts=12,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.COUNTERFACTUAL,
                        name="counterfactual",
                        description="Explore alternative scenarios",
                        max_thoughts=10,
                    ),
                ],
                merge_strategy=MergeStrategy(
                    name="collect",
                    selection_criteria="all_results",
                    aggregation="synthesize",
                ),
                max_concurrency=3,
            ),
            MethodStage(
                method_id=MethodIdentifier.BEST_OF_N,
                name="select_best",
                description="Select the best ideas from exploration",
                max_thoughts=8,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "creative_explore",
            "best_for": ["brainstorming", "creative problems", "ideation"],
            "description": "Multi-path creative exploration with selection",
            "estimated_thoughts": 40,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_scientific_method_template() -> dict[str, Any]:
    """Create scientific method template.

    This template follows the scientific method:
    1. Form hypothesis
    2. Experimental design
    3. Verification
    4. Refinement

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="scientific_method",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="form_hypothesis",
                description="Form initial hypothesis",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.STEP_BACK,
                name="experimental_design",
                description="Design experimental approach",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_VERIFICATION,
                name="verify",
                description="Verify through experimental reasoning",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFINE,
                name="refine_conclusion",
                description="Refine the conclusion",
                max_thoughts=8,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "scientific_method",
            "best_for": ["scientific problems", "hypothesis testing", "research"],
            "description": "Hypothesis, experiment design, verification, refinement",
            "estimated_thoughts": 38,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_decompose_solve_template() -> dict[str, Any]:
    """Create decompose and solve template.

    This template breaks down complex problems:
    1. Decompose the problem
    2. Progressive solving with least-to-most
    3. Cumulative synthesis

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="decompose_solve",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.DECOMPOSED_PROMPTING,
                name="decompose",
                description="Break problem into sub-problems",
                max_thoughts=10,
            ),
            MethodStage(
                method_id=MethodIdentifier.LEAST_TO_MOST,
                name="progressive_solve",
                description="Solve from simplest to most complex",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.CUMULATIVE_REASONING,
                name="synthesize",
                description="Synthesize solutions cumulatively",
                max_thoughts=10,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "decompose_solve",
            "best_for": ["complex problems", "multi-part problems", "step-by-step"],
            "description": "Complex problem decomposition with progressive solving",
            "estimated_thoughts": 35,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_multi_agent_debate_template() -> dict[str, Any]:
    """Create multi-agent debate template.

    This template simulates debate between perspectives:
    1. Parallel generation of different perspectives
    2. Debate and challenge
    3. Consensus synthesis

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="multi_agent_debate",
        stages=[
            ParallelPipeline(
                name="perspectives",
                branches=[
                    MethodStage(
                        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                        name="optimist",
                        description="Optimistic perspective",
                        max_thoughts=8,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                        name="skeptic",
                        description="Skeptical perspective",
                        max_thoughts=8,
                    ),
                    MethodStage(
                        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                        name="pragmatist",
                        description="Pragmatic perspective",
                        max_thoughts=8,
                    ),
                ],
                merge_strategy=MergeStrategy(
                    name="collect",
                    selection_criteria="all_results",
                    aggregation="synthesize",
                ),
                max_concurrency=3,
            ),
            MethodStage(
                method_id=MethodIdentifier.MULTI_AGENT_DEBATE,
                name="debate",
                description="Debate between perspectives",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="consensus",
                description="Synthesize consensus",
                max_thoughts=10,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "multi_agent_debate",
            "best_for": ["controversial topics", "decision making", "consensus building"],
            "description": "Multiple perspectives debate to consensus",
            "estimated_thoughts": 46,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_deep_research_template() -> dict[str, Any]:
    """Create deep research template.

    This template enables deep research:
    1. Retrieval augmented thinking
    2. Knowledge graph exploration
    3. Cumulative synthesis

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="deep_research",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.RETRIEVAL_AUGMENTED_THOUGHTS,
                name="retrieve",
                description="Retrieve relevant knowledge",
                max_thoughts=12,
            ),
            MethodStage(
                method_id=MethodIdentifier.THINK_ON_GRAPH,
                name="graph_explore",
                description="Explore knowledge graph",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.CUMULATIVE_REASONING,
                name="synthesize",
                description="Synthesize findings",
                max_thoughts=10,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "deep_research",
            "best_for": ["research", "knowledge synthesis", "complex queries"],
            "description": "Retrieval-augmented research with knowledge graph",
            "estimated_thoughts": 37,
        },
    )
    return pipeline.model_dump(mode="json")


def _create_decision_matrix_template() -> dict[str, Any]:
    """Create decision matrix template.

    This template helps with complex decisions:
    1. Explore options with Tree of Thoughts
    2. Monte Carlo Tree Search for optimization
    3. Outcome evaluation

    Returns:
        Dictionary representation of the pipeline template
    """
    pipeline = SequencePipeline(
        name="decision_matrix",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                name="explore_options",
                description="Explore decision options",
                max_thoughts=15,
            ),
            MethodStage(
                method_id=MethodIdentifier.MCTS,
                name="optimize",
                description="Optimize using Monte Carlo Tree Search",
                max_thoughts=20,
            ),
            MethodStage(
                method_id=MethodIdentifier.OUTCOME_REWARD_MODEL,
                name="evaluate",
                description="Evaluate outcomes",
                max_thoughts=10,
            ),
        ],
        stop_on_error=True,
        metadata={
            "template_id": "decision_matrix",
            "best_for": ["decision making", "optimization", "strategy"],
            "description": "Tree exploration with MCTS optimization",
            "estimated_thoughts": 45,
        },
    )
    return pipeline.model_dump(mode="json")


# ============================================================================
# Template Registry
# ============================================================================

_TEMPLATE_CREATORS = {
    "verified_reasoning": _create_verified_reasoning_template,
    "iterative_improve": _create_iterative_improve_template,
    "analyze_refine": _create_analyze_refine_template,
    "ethical_multi_view": _create_ethical_multi_view_template,
    "math_proof": _create_math_proof_template,
    "debug_code": _create_debug_code_template,
    "creative_explore": _create_creative_explore_template,
    "scientific_method": _create_scientific_method_template,
    "decompose_solve": _create_decompose_solve_template,
    "multi_agent_debate": _create_multi_agent_debate_template,
    "deep_research": _create_deep_research_template,
    "decision_matrix": _create_decision_matrix_template,
}

# Cache for templates
_TEMPLATE_CACHE: dict[str, dict[str, Any]] = {}


def get_template(template_id: str) -> dict[str, Any] | None:
    """Get a pipeline template by ID.

    Args:
        template_id: Template identifier

    Returns:
        Template definition dict or None if not found
    """
    if template_id not in _TEMPLATE_CREATORS:
        return None

    if template_id not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[template_id] = _TEMPLATE_CREATORS[template_id]()

    return _TEMPLATE_CACHE[template_id]


def get_available_templates() -> list[str]:
    """Get list of available template IDs.

    Returns:
        List of template identifiers
    """
    return sorted(_TEMPLATE_CREATORS.keys())


def get_template_metadata(template_id: str) -> dict[str, Any] | None:
    """Get metadata for a template.

    Args:
        template_id: Template identifier

    Returns:
        Template metadata dict or None if not found
    """
    template = get_template(template_id)
    if template:
        metadata = template.get("metadata", {})
        return dict(metadata) if metadata else {}
    return None


def get_templates_for_domain(domain: str) -> list[str]:
    """Get templates that are good for a specific domain.

    Args:
        domain: Domain name (e.g., "ethical", "mathematical")

    Returns:
        List of template IDs that match the domain
    """
    matching = []
    domain_lower = domain.lower()

    for template_id in _TEMPLATE_CREATORS:
        metadata = get_template_metadata(template_id)
        if metadata:
            best_for = metadata.get("best_for", [])
            for use_case in best_for:
                if domain_lower in use_case.lower():
                    matching.append(template_id)
                    break

    return matching


__all__ = [
    "get_template",
    "get_available_templates",
    "get_template_metadata",
    "get_templates_for_domain",
]
