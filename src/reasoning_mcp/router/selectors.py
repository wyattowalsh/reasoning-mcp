"""Route selectors for the Reasoning Router.

This module provides selectors that choose the optimal routing
decision based on problem analysis and constraints.

Tier 1 (Fast): Rule-based with domain→method mappings
Tier 2 (Standard): Matrix factorization scoring
Tier 3 (Complex): LLM-based selection with synthesis
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Scoring Constants
# =============================================================================

# Base score for any method/template match
SCORE_BASE: float = 0.1

# Domain matching score boosts (primary priority - highest weights)
SCORE_PRIMARY_DOMAIN_MATCH: float = 0.7  # Strong boost for primary domain match
SCORE_SECONDARY_DOMAIN_MATCH: float = 0.45  # Moderate boost for secondary match

# Secondary domain matching (lower priority)
SCORE_SECONDARY_DOMAIN_PRIMARY: float = 0.2  # Secondary domain's primary method
SCORE_SECONDARY_DOMAIN_SECONDARY: float = 0.12  # Secondary domain's secondary method

# Capability matching (should enhance, not override domain match)
CAPABILITY_SCORE_MULTIPLIER: float = 0.7  # Scale down capability scores
CAPABILITY_MAX_CONTRIBUTION: float = 0.35  # Cap capability boost

# Template scoring
TEMPLATE_DOMAIN_PRIMARY_BOOST: float = 0.3
TEMPLATE_DOMAIN_SECONDARY_BOOST: float = 0.15
TEMPLATE_CAPABILITY_MULTIPLIER: float = 0.2
TEMPLATE_COMPLEXITY_BOOST: float = 0.15
TEMPLATE_SCORE_THRESHOLD: float = 0.15  # Min score before applying score_boost

# Maximum allowed score
SCORE_MAX: float = 1.0

# =============================================================================
# Resource Estimation Constants
# =============================================================================

RESOURCE_BASE_TOKENS: int = 1000
RESOURCE_BASE_LATENCY_MS: int = 1000
RESOURCE_BASE_THOUGHTS: int = 3

# Complexity scaling: multiplier = 0.5 + (complexity / 10)
COMPLEXITY_BASE_OFFSET: float = 0.5
COMPLEXITY_DIVISOR: float = 10.0

# Single method confidence
SINGLE_METHOD_CONFIDENCE: float = 0.7

# Pipeline template multipliers
PIPELINE_TOKEN_MULTIPLIER: float = 2.0
PIPELINE_LATENCY_MULTIPLIER: float = 1.5
PIPELINE_THOUGHTS_MULTIPLIER: float = 2.0
PIPELINE_CONFIDENCE: float = 0.6

# Ensemble multipliers
ENSEMBLE_TOKEN_MULTIPLIER: float = 3.0
ENSEMBLE_LATENCY_MULTIPLIER: float = 2.0
ENSEMBLE_THOUGHTS_MULTIPLIER: float = 3.0
ENSEMBLE_CONFIDENCE: float = 0.5

# =============================================================================
# Selection Constants
# =============================================================================

# Confidence multipliers for route decisions
METHOD_CONFIDENCE_MULTIPLIER: float = 0.9
TEMPLATE_CONFIDENCE_MULTIPLIER: float = 0.85

# Result limits and thresholds
MIN_METHOD_SCORE_THRESHOLD: float = 0.2  # Filter low-scoring methods
MIN_TEMPLATE_SCORE_THRESHOLD: float = 0.2  # Filter low-scoring templates
MAX_RESULTS: int = 5
MIN_RESULTS_BEFORE_TEMPLATE_FILTER: int = 3

# Standard tier score/confidence boost
STANDARD_TIER_BOOST: float = 0.05

from reasoning_mcp.router.models import (
    DOMAIN_METHOD_MAPPING,
    ProblemAnalysis,
    ProblemDomain,
    RequiredCapability,
    ResourceBudget,
    ResourceEstimate,
    RouteDecision,
    RouterTier,
    RouteType,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

    from reasoning_mcp.registry import MethodRegistry


class RouteSelector(ABC):
    """Abstract base class for route selectors."""

    @property
    @abstractmethod
    def tier(self) -> RouterTier:
        """Return the tier of this selector."""
        ...

    @abstractmethod
    async def select(
        self,
        analysis: ProblemAnalysis,
        budget: ResourceBudget | None = None,
    ) -> list[RouteDecision]:
        """Select routes based on problem analysis."""
        ...


class TemplateTrigger(TypedDict, total=False):
    """Type definition for template trigger configuration.

    Attributes:
        capabilities: Required capabilities for this template
        domains: Problem domains this template handles
        min_complexity: Minimum complexity threshold (1-10)
        score_boost: Additional score boost when conditions match
    """

    capabilities: set[RequiredCapability]
    domains: set[ProblemDomain]
    min_complexity: int
    score_boost: float


# Method scoring rules based on capabilities
CAPABILITY_METHOD_SCORES: dict[RequiredCapability, dict[str, float]] = {
    RequiredCapability.BRANCHING: {
        "tree_of_thoughts": 0.4,
        "mcts": 0.3,
        "beam_search": 0.3,
        "self_consistency": 0.2,
    },
    RequiredCapability.ITERATION: {
        "self_refine": 0.4,
        "reflexion": 0.4,
        "iterative_refinement": 0.4,
        "chain_of_verification": 0.3,
    },
    RequiredCapability.EXTERNAL_TOOLS: {
        "react": 0.5,
        "code_reasoning": 0.3,
        "program_of_thoughts": 0.3,
    },
    RequiredCapability.VERIFICATION: {
        "chain_of_verification": 0.5,
        "self_verification": 0.4,
        "self_consistency": 0.3,
    },
    RequiredCapability.DECOMPOSITION: {
        "least_to_most": 0.4,
        "decomposed_reasoning": 0.4,
        "plan_and_solve": 0.3,
        "step_back": 0.3,
    },
    RequiredCapability.SYNTHESIS: {
        "cumulative_reasoning": 0.4,
        "multi_agent_debate": 0.3,
        "self_consistency": 0.3,
    },
    RequiredCapability.FORMAL_LOGIC: {
        "logic_of_thought": 0.5,
        "mathematical_reasoning": 0.4,
        "chain_of_thought": 0.2,
    },
    RequiredCapability.CREATIVITY: {
        "lateral_thinking": 0.5,
        "tree_of_thoughts": 0.3,
        "counterfactual": 0.3,
    },
    RequiredCapability.MEMORY: {
        "buffer_of_thoughts": 0.4,
        "cumulative_reasoning": 0.3,
        "meta_cot": 0.3,
    },
    RequiredCapability.MULTI_AGENT: {
        "multi_agent_debate": 0.5,
        "mutual_reasoning": 0.4,
        "dialectic": 0.3,
    },
}

# Pipeline template triggers based on analysis
TEMPLATE_TRIGGERS: dict[str, TemplateTrigger] = {
    "verified_reasoning": {
        "capabilities": {RequiredCapability.VERIFICATION},
        "min_complexity": 5,
        "score_boost": 0.3,
    },
    "iterative_improve": {
        "capabilities": {RequiredCapability.ITERATION},
        "min_complexity": 4,
        "score_boost": 0.25,
    },
    "analyze_refine": {
        "domains": {ProblemDomain.ANALYTICAL},
        "score_boost": 0.2,
    },
    "ethical_multi_view": {
        "domains": {ProblemDomain.ETHICAL},
        "score_boost": 0.35,
    },
    "math_proof": {
        "domains": {ProblemDomain.MATHEMATICAL},
        "capabilities": {RequiredCapability.FORMAL_LOGIC, RequiredCapability.VERIFICATION},
        "score_boost": 0.3,
    },
    "debug_code": {
        "domains": {ProblemDomain.CODE},
        "capabilities": {RequiredCapability.EXTERNAL_TOOLS},
        "score_boost": 0.3,
    },
    "creative_explore": {
        "domains": {ProblemDomain.CREATIVE},
        "capabilities": {RequiredCapability.CREATIVITY, RequiredCapability.BRANCHING},
        "score_boost": 0.3,
    },
    "scientific_method": {
        "domains": {ProblemDomain.SCIENTIFIC},
        "capabilities": {RequiredCapability.VERIFICATION},
        "score_boost": 0.25,
    },
    "decompose_solve": {
        "capabilities": {RequiredCapability.DECOMPOSITION},
        "min_complexity": 6,
        "score_boost": 0.3,
    },
    "multi_agent_debate": {
        "capabilities": {RequiredCapability.MULTI_AGENT, RequiredCapability.SYNTHESIS},
        "min_complexity": 7,
        "score_boost": 0.35,
    },
    "decision_matrix": {
        "domains": {ProblemDomain.DECISION},
        "capabilities": {RequiredCapability.BRANCHING},
        "min_complexity": 5,
        "score_boost": 0.3,
    },
}


def _score_method(method_id: str, analysis: ProblemAnalysis) -> float:
    """Score a method based on problem analysis.

    Scoring priority (highest to lowest):
    1. Primary domain match: Strong boost ensures domain-specific methods win
    2. Secondary domain match: Moderate boost
    3. Capability matching: Smaller boost, should not override domain match

    This ensures ethical problems route to ethical_reasoning, code problems
    to code_reasoning, etc., even when capability patterns also match.
    """
    score = SCORE_BASE

    # Domain matching - PRIMARY PRIORITY
    # Use strong weights to ensure domain-matched methods beat capability-only matches
    primary_method, secondary_method = DOMAIN_METHOD_MAPPING.get(
        analysis.primary_domain, ("chain_of_thought", None)
    )
    if method_id == primary_method:
        score += SCORE_PRIMARY_DOMAIN_MATCH
    elif method_id == secondary_method:
        score += SCORE_SECONDARY_DOMAIN_MATCH

    # Secondary domain matching - still relevant but lower priority
    for domain in analysis.secondary_domains:
        domain_primary, domain_secondary = DOMAIN_METHOD_MAPPING.get(domain, (None, None))
        if method_id == domain_primary:
            score += SCORE_SECONDARY_DOMAIN_PRIMARY
        elif method_id == domain_secondary:
            score += SCORE_SECONDARY_DOMAIN_SECONDARY

    # Capability matching - should enhance, not override domain match
    # Scale down capability scores to prevent them from beating domain matches
    capability_score = 0.0
    for capability in analysis.capabilities:
        method_scores = CAPABILITY_METHOD_SCORES.get(capability, {})
        if method_id in method_scores:
            capability_score += method_scores[method_id]

    # Cap capability contribution to prevent it from dominating
    score += min(CAPABILITY_MAX_CONTRIBUTION, capability_score * CAPABILITY_SCORE_MULTIPLIER)

    return min(SCORE_MAX, score)


def _score_template(template_id: str, analysis: ProblemAnalysis) -> float:
    """Score a pipeline template based on problem analysis."""
    trigger = TEMPLATE_TRIGGERS.get(template_id)
    if not trigger:
        return 0.0

    score = SCORE_BASE

    # Check domain match
    domains = trigger.get("domains")
    if domains is not None:
        if analysis.primary_domain in domains:
            score += TEMPLATE_DOMAIN_PRIMARY_BOOST
        if analysis.secondary_domains & domains:
            score += TEMPLATE_DOMAIN_SECONDARY_BOOST

    # Check capability match
    caps = trigger.get("capabilities")
    if caps is not None:
        matched = analysis.capabilities & caps
        if matched:
            score += TEMPLATE_CAPABILITY_MULTIPLIER * (len(matched) / len(caps))

    # Check complexity threshold
    min_complexity = trigger.get("min_complexity")
    if min_complexity is not None:
        if analysis.complexity >= min_complexity:
            score += TEMPLATE_COMPLEXITY_BOOST

    # Apply score boost if any condition matched
    score_boost = trigger.get("score_boost")
    if score > TEMPLATE_SCORE_THRESHOLD and score_boost is not None:
        score += score_boost

    return min(SCORE_MAX, score)


def _estimate_resources(
    route_type: RouteType,
    method_id: str | None = None,
    template_id: str | None = None,
    analysis: ProblemAnalysis | None = None,
) -> ResourceEstimate:
    """Estimate resource usage for a route."""
    complexity_multiplier = 1.0
    if analysis:
        complexity_multiplier = COMPLEXITY_BASE_OFFSET + (analysis.complexity / COMPLEXITY_DIVISOR)

    if route_type == RouteType.SINGLE_METHOD:
        return ResourceEstimate(
            estimated_tokens=int(RESOURCE_BASE_TOKENS * complexity_multiplier),
            estimated_latency_ms=int(RESOURCE_BASE_LATENCY_MS * complexity_multiplier),
            estimated_thoughts=int(RESOURCE_BASE_THOUGHTS * complexity_multiplier),
            confidence=SINGLE_METHOD_CONFIDENCE,
        )
    elif route_type == RouteType.PIPELINE_TEMPLATE:
        # Templates typically involve multiple steps
        return ResourceEstimate(
            estimated_tokens=int(
                RESOURCE_BASE_TOKENS * complexity_multiplier * PIPELINE_TOKEN_MULTIPLIER
            ),
            estimated_latency_ms=int(
                RESOURCE_BASE_LATENCY_MS * complexity_multiplier * PIPELINE_LATENCY_MULTIPLIER
            ),
            estimated_thoughts=int(
                RESOURCE_BASE_THOUGHTS * complexity_multiplier * PIPELINE_THOUGHTS_MULTIPLIER
            ),
            confidence=PIPELINE_CONFIDENCE,
        )
    elif route_type == RouteType.METHOD_ENSEMBLE:
        return ResourceEstimate(
            estimated_tokens=int(
                RESOURCE_BASE_TOKENS * complexity_multiplier * ENSEMBLE_TOKEN_MULTIPLIER
            ),
            estimated_latency_ms=int(
                RESOURCE_BASE_LATENCY_MS * complexity_multiplier * ENSEMBLE_LATENCY_MULTIPLIER
            ),
            estimated_thoughts=int(
                RESOURCE_BASE_THOUGHTS * complexity_multiplier * ENSEMBLE_THOUGHTS_MULTIPLIER
            ),
            confidence=ENSEMBLE_CONFIDENCE,
        )
    else:
        return ResourceEstimate()


class FastRouteSelector(RouteSelector):
    """Fast tier selector using rule-based scoring.

    Latency target: <5ms
    """

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        """Initialize with optional method registry."""
        self._registry = registry

    @property
    def tier(self) -> RouterTier:
        return RouterTier.FAST

    async def select(
        self,
        analysis: ProblemAnalysis,
        budget: ResourceBudget | None = None,
    ) -> list[RouteDecision]:
        """Select routes using rule-based scoring."""
        start = time.perf_counter()
        budget = budget or ResourceBudget()

        decisions: list[RouteDecision] = []

        # Score methods
        method_scores: dict[str, float] = {}

        # Get primary and secondary methods from domain mapping
        primary_method, secondary_method = DOMAIN_METHOD_MAPPING.get(
            analysis.primary_domain, ("chain_of_thought", None)
        )

        # Score primary method
        method_scores[primary_method] = _score_method(primary_method, analysis)

        # Score secondary method if exists
        if secondary_method:
            method_scores[secondary_method] = _score_method(secondary_method, analysis)

        # Score methods from capability mappings
        for capability in analysis.capabilities:
            for method_id in CAPABILITY_METHOD_SCORES.get(capability, {}):
                if method_id not in method_scores:
                    method_scores[method_id] = _score_method(method_id, analysis)

        # Score pipeline templates
        template_scores: dict[str, float] = {}
        for template_id in TEMPLATE_TRIGGERS:
            score = _score_template(template_id, analysis)
            if score > MIN_TEMPLATE_SCORE_THRESHOLD:
                template_scores[template_id] = score

        latency_ms = (time.perf_counter() - start) * 1000

        # Build decisions - methods first
        for method_id, score in sorted(method_scores.items(), key=lambda x: -x[1]):
            if score < MIN_METHOD_SCORE_THRESHOLD:
                continue

            resource_est = _estimate_resources(
                RouteType.SINGLE_METHOD, method_id=method_id, analysis=analysis
            )

            # Check budget constraints
            if budget.max_tokens and resource_est.estimated_tokens > budget.max_tokens:
                continue
            if budget.max_thoughts and resource_est.estimated_thoughts > budget.max_thoughts:
                continue

            reasoning_parts = [f"Domain match: {analysis.primary_domain.value}"]
            if analysis.capabilities:
                reasoning_parts.append(
                    f"Capabilities: {', '.join(c.value for c in analysis.capabilities)}"
                )

            decisions.append(
                RouteDecision(
                    route_type=RouteType.SINGLE_METHOD,
                    method_id=method_id,
                    score=score,
                    confidence=analysis.confidence * METHOD_CONFIDENCE_MULTIPLIER,
                    reasoning="; ".join(reasoning_parts),
                    matched_criteria=(analysis.primary_domain.value,),
                    router_tier=RouterTier.FAST,
                    latency_ms=latency_ms,
                )
            )

            if len(decisions) >= MAX_RESULTS:
                break

        # Add template decisions if they score higher than lowest method
        min_method_score = decisions[-1].score if decisions else 0.0

        for template_id, score in sorted(template_scores.items(), key=lambda x: -x[1]):
            if score <= min_method_score and len(decisions) >= MIN_RESULTS_BEFORE_TEMPLATE_FILTER:
                continue

            resource_est = _estimate_resources(
                RouteType.PIPELINE_TEMPLATE, template_id=template_id, analysis=analysis
            )

            # Check budget constraints
            if budget.max_tokens and resource_est.estimated_tokens > budget.max_tokens:
                continue

            decisions.append(
                RouteDecision(
                    route_type=RouteType.PIPELINE_TEMPLATE,
                    pipeline_id=template_id,
                    score=score,
                    confidence=analysis.confidence * TEMPLATE_CONFIDENCE_MULTIPLIER,
                    reasoning=f"Template match for {analysis.primary_domain.value} problems",
                    matched_criteria=(template_id,),
                    router_tier=RouterTier.FAST,
                    latency_ms=latency_ms,
                )
            )

        # Sort by score and return top results
        decisions.sort(key=lambda d: -d.score)
        return decisions[:MAX_RESULTS]


class StandardRouteSelector(RouteSelector):
    """Standard tier selector using matrix factorization.

    Latency target: ~20ms

    Note: Full MF implementation requires embeddings.
    Currently uses enhanced rule-based scoring.
    """

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        """Initialize with optional method registry."""
        self._registry = registry
        self._fast_selector = FastRouteSelector(registry)

    @property
    def tier(self) -> RouterTier:
        return RouterTier.STANDARD

    async def select(
        self,
        analysis: ProblemAnalysis,
        budget: ResourceBudget | None = None,
    ) -> list[RouteDecision]:
        """Select routes using matrix factorization (with fast fallback)."""
        start = time.perf_counter()

        # For now, use fast selector and enhance scores
        fast_results = await self._fast_selector.select(analysis, budget)

        # TODO: Add embedding-based matrix factorization scoring
        # - Method factors matrix (num_methods × dim)
        # - Problem encoder → embedding
        # - Score = dot(method_factors, problem_embedding)

        latency_ms = (time.perf_counter() - start) * 1000

        # Update tier in results
        enhanced_results = []
        for result in fast_results:
            enhanced_results.append(
                RouteDecision(
                    route_type=result.route_type,
                    method_id=result.method_id,
                    pipeline_id=result.pipeline_id,
                    pipeline_definition=result.pipeline_definition,
                    ensemble_methods=result.ensemble_methods,
                    ensemble_strategy=result.ensemble_strategy,
                    score=min(SCORE_MAX, result.score + STANDARD_TIER_BOOST),
                    confidence=min(SCORE_MAX, result.confidence + STANDARD_TIER_BOOST),
                    reasoning=result.reasoning,
                    matched_criteria=result.matched_criteria,
                    router_tier=RouterTier.STANDARD,
                    latency_ms=latency_ms,
                )
            )

        return enhanced_results


class ComplexRouteSelector(RouteSelector):
    """Complex tier selector using LLM-based analysis.

    Latency target: ~200ms

    Uses MCP sampling for intelligent method/pipeline selection
    and dynamic pipeline synthesis.
    """

    def __init__(
        self,
        registry: MethodRegistry | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> None:
        """Initialize with registry and MCP context."""
        self._registry = registry
        self._ctx = ctx
        self._standard_selector = StandardRouteSelector(registry)

    @property
    def tier(self) -> RouterTier:
        return RouterTier.COMPLEX

    async def select(
        self,
        analysis: ProblemAnalysis,
        budget: ResourceBudget | None = None,
    ) -> list[RouteDecision]:
        """Select routes using LLM (with standard fallback)."""
        start = time.perf_counter()

        if self._ctx is None:
            # Fall back to standard selector
            results = await self._standard_selector.select(analysis, budget)
            latency_ms = (time.perf_counter() - start) * 1000

            return [
                RouteDecision(
                    route_type=r.route_type,
                    method_id=r.method_id,
                    pipeline_id=r.pipeline_id,
                    pipeline_definition=r.pipeline_definition,
                    ensemble_methods=r.ensemble_methods,
                    ensemble_strategy=r.ensemble_strategy,
                    score=r.score,
                    confidence=r.confidence,
                    reasoning=r.reasoning,
                    matched_criteria=r.matched_criteria,
                    router_tier=RouterTier.COMPLEX,
                    latency_ms=latency_ms,
                )
                for r in results
            ]

        # Use LLM-based selection via MCP sampling
        from reasoning_mcp.router.llm_provider import (
            LLMProvider,
            parse_route_type,
        )

        llm_provider = LLMProvider(ctx=self._ctx)

        # Build analysis summary for LLM
        analysis_summary = (
            f"Domain: {analysis.primary_domain.value}, "
            f"Intent: {analysis.intent.value}, "
            f"Complexity: {analysis.complexity}/10, "
            f"Capabilities: {', '.join(c.value for c in analysis.capabilities) if analysis.capabilities else 'none'}"
        )

        # Get available methods and templates for context
        available_methods = list(DOMAIN_METHOD_MAPPING.values())
        # Flatten to unique method names
        unique_methods = list({m for pair in available_methods for m in pair if m is not None})
        available_templates = list(TEMPLATE_TRIGGERS.keys())

        # Get original problem text from keywords/entities for context
        problem_context = ", ".join(analysis.keywords) if analysis.keywords else "general problem"

        llm_result = await llm_provider.select_routes(
            problem=problem_context,
            analysis_summary=analysis_summary,
            available_methods=unique_methods,
            available_templates=available_templates,
        )

        if llm_result is not None and llm_result.recommendations:
            # Successfully got LLM recommendations - convert to RouteDecisions
            latency_ms = (time.perf_counter() - start) * 1000
            decisions: list[RouteDecision] = []

            for rec in llm_result.recommendations:
                route_type = parse_route_type(rec.route_type)

                # Build the RouteDecision based on type
                if route_type == RouteType.SINGLE_METHOD and rec.method_id:
                    decisions.append(
                        RouteDecision(
                            route_type=RouteType.SINGLE_METHOD,
                            method_id=rec.method_id,
                            score=rec.score,
                            confidence=llm_result.confidence,
                            reasoning=rec.reasoning or f"LLM recommended {rec.method_id}",
                            matched_criteria=(analysis.primary_domain.value, analysis.intent.value),
                            router_tier=RouterTier.COMPLEX,
                            latency_ms=latency_ms,
                        )
                    )
                elif route_type == RouteType.PIPELINE_TEMPLATE and rec.pipeline_id:
                    decisions.append(
                        RouteDecision(
                            route_type=RouteType.PIPELINE_TEMPLATE,
                            pipeline_id=rec.pipeline_id,
                            score=rec.score,
                            confidence=llm_result.confidence,
                            reasoning=rec.reasoning
                            or f"LLM recommended template {rec.pipeline_id}",
                            matched_criteria=(rec.pipeline_id,),
                            router_tier=RouterTier.COMPLEX,
                            latency_ms=latency_ms,
                        )
                    )
                elif route_type == RouteType.METHOD_ENSEMBLE and rec.ensemble_methods:
                    decisions.append(
                        RouteDecision(
                            route_type=RouteType.METHOD_ENSEMBLE,
                            ensemble_methods=tuple(rec.ensemble_methods),
                            ensemble_strategy="vote",
                            score=rec.score,
                            confidence=llm_result.confidence,
                            reasoning=rec.reasoning
                            or f"LLM recommended ensemble: {', '.join(rec.ensemble_methods)}",
                            matched_criteria=tuple(rec.ensemble_methods),
                            router_tier=RouterTier.COMPLEX,
                            latency_ms=latency_ms,
                        )
                    )

            if decisions:
                logger.debug(
                    "LLM selection returned %d decisions for %s problem",
                    len(decisions),
                    analysis.primary_domain.value,
                )
                # Sort by score and return
                decisions.sort(key=lambda d: -d.score)
                return decisions[:5]

            # LLM returned empty recommendations - fall through to fallback
            logger.warning("LLM returned no valid recommendations, falling back to standard")

        else:
            # LLM selection failed - fall back to standard selector
            logger.warning("LLM selection failed, falling back to standard selector")

        # Fallback: use standard selector
        results = await self._standard_selector.select(analysis, budget)
        latency_ms = (time.perf_counter() - start) * 1000

        return [
            RouteDecision(
                route_type=r.route_type,
                method_id=r.method_id,
                pipeline_id=r.pipeline_id,
                pipeline_definition=r.pipeline_definition,
                ensemble_methods=r.ensemble_methods,
                ensemble_strategy=r.ensemble_strategy,
                score=r.score,
                confidence=r.confidence,
                reasoning=r.reasoning,
                matched_criteria=r.matched_criteria,
                router_tier=RouterTier.COMPLEX,
                latency_ms=latency_ms,
            )
            for r in results
        ]


def get_selector(
    tier: RouterTier,
    registry: MethodRegistry | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> RouteSelector:
    """Get a selector for the specified tier."""
    if tier == RouterTier.FAST:
        return FastRouteSelector(registry)
    elif tier == RouterTier.STANDARD:
        return StandardRouteSelector(registry)
    elif tier == RouterTier.COMPLEX:
        return ComplexRouteSelector(registry, ctx)
    else:
        return FastRouteSelector(registry)  # Default fallback
