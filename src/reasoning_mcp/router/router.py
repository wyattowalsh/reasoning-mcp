"""Main Reasoning Router implementation.

This module provides the ReasoningRouter class that orchestrates
problem analysis and route selection across multiple tiers.

FastMCP v2.14+ Features:
- Response caching for route() and recommend() operations
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, cast

from reasoning_mcp.router.analyzers import (
    ComplexProblemAnalyzer,
    FastProblemAnalyzer,
    ProblemAnalyzer,
    StandardProblemAnalyzer,
)
from reasoning_mcp.router.models import (
    ProblemAnalysis,
    ResourceBudget,
    ResourceEstimate,
    RouteDecision,
    RouterResult,
    RouterTier,
    RouteType,
)
from reasoning_mcp.router.selectors import (
    ComplexRouteSelector,
    FastRouteSelector,
    RouteSelector,
    StandardRouteSelector,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

    from reasoning_mcp.middleware import ResponseCacheMiddleware
    from reasoning_mcp.registry import MethodRegistry


class ReasoningRouter:
    """Multi-tier intelligent router for reasoning method selection.

    The router uses a tiered strategy:
    - Tier 1 (Fast, <5ms): Embedding similarity + regex patterns
    - Tier 2 (Standard, ~20ms): ML classifiers + matrix factorization
    - Tier 3 (Complex, ~200ms): LLM-based analysis + pipeline synthesis

    Usage:
        router = ReasoningRouter(registry)
        result = await router.route("How do I solve this problem?")

        if result.primary_route.route_type == RouteType.SINGLE_METHOD:
            method_id = result.primary_route.method_id
        elif result.primary_route.route_type == RouteType.PIPELINE_TEMPLATE:
            template_id = result.primary_route.pipeline_id
    """

    def __init__(
        self,
        registry: MethodRegistry | None = None,
        ctx: Context[Any, Any, Any] | None = None,
        default_tier: RouterTier = RouterTier.STANDARD,
        enable_ml_routing: bool = True,
        enable_llm_routing: bool = True,
        cache: ResponseCacheMiddleware | None = None,
        cache_routing_ttl: int = 60,
        cache_recommend_ttl: int = 300,
        cache_analysis_ttl: int = 300,
    ) -> None:
        """Initialize the router.

        Args:
            registry: Optional method registry for method validation
            ctx: Optional MCP context for LLM-based routing
            default_tier: Default routing tier
            enable_ml_routing: Enable Tier 2 ML-based routing
            enable_llm_routing: Enable Tier 3 LLM-based routing
            cache: Optional response cache middleware (FastMCP v2.14+)
            cache_routing_ttl: TTL in seconds for routing cache (default 60)
            cache_recommend_ttl: TTL in seconds for recommendation cache (default 300)
            cache_analysis_ttl: TTL in seconds for analysis cache (default 300)
        """
        self._registry = registry
        self._ctx = ctx
        self._default_tier = default_tier
        self._enable_ml_routing = enable_ml_routing
        self._enable_llm_routing = enable_llm_routing
        self._cache = cache
        self._cache_routing_ttl = cache_routing_ttl
        self._cache_recommend_ttl = cache_recommend_ttl
        self._cache_analysis_ttl = cache_analysis_ttl

        # Initialize analyzers with cache support
        self._fast_analyzer = FastProblemAnalyzer(cache=cache)
        self._standard_analyzer = (
            StandardProblemAnalyzer(cache=cache) if enable_ml_routing else None
        )
        self._complex_analyzer = (
            ComplexProblemAnalyzer(ctx=ctx, cache=cache, cache_ttl=cache_analysis_ttl)
            if enable_llm_routing
            else None
        )

        # Initialize selectors
        self._fast_selector = FastRouteSelector(registry)
        self._standard_selector = StandardRouteSelector(registry) if enable_ml_routing else None
        self._complex_selector = ComplexRouteSelector(registry, ctx) if enable_llm_routing else None

    def _make_cache_key(
        self,
        namespace: str,
        problem: str,
        **kwargs: object,
    ) -> str:
        """Generate a cache key for routing operations.

        Args:
            namespace: Cache namespace (e.g., "route", "recommend")
            problem: The problem text
            **kwargs: Additional key components

        Returns:
            32-character hex digest cache key
        """
        # Hash the problem for the key
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
        sorted_kwargs = sorted((k, str(v)) for k, v in kwargs.items())
        key_data = f"{namespace}:{problem_hash}:{sorted_kwargs}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _select_tier(
        self,
        problem: str,
        budget: ResourceBudget | None = None,
        force_tier: RouterTier | None = None,
    ) -> RouterTier:
        """Select the appropriate routing tier.

        Args:
            problem: The problem text
            budget: Optional resource budget
            force_tier: Force a specific tier

        Returns:
            Selected routing tier
        """
        if force_tier is not None:
            # Validate tier is enabled
            if force_tier == RouterTier.STANDARD and not self._enable_ml_routing:
                return RouterTier.FAST
            if force_tier == RouterTier.COMPLEX and not self._enable_llm_routing:
                return RouterTier.STANDARD if self._enable_ml_routing else RouterTier.FAST
            return force_tier

        budget = budget or ResourceBudget()

        # Speed-first preference
        if budget.prefer_speed or (budget.max_latency_ms and budget.max_latency_ms < 50):
            return RouterTier.FAST

        # Short problems use fast tier
        word_count = len(problem.split())
        if word_count < 20:
            return RouterTier.FAST

        # Quality-first preference for complex problems
        if budget.prefer_quality and word_count > 200 and self._enable_llm_routing:
            return RouterTier.COMPLEX

        # Default to configured tier
        return self._default_tier

    def _get_analyzer(self, tier: RouterTier) -> ProblemAnalyzer:
        """Get the analyzer for a tier."""
        if tier == RouterTier.FAST:
            return self._fast_analyzer
        elif tier == RouterTier.STANDARD and self._standard_analyzer:
            return self._standard_analyzer
        elif tier == RouterTier.COMPLEX and self._complex_analyzer:
            return self._complex_analyzer
        return self._fast_analyzer  # Fallback

    def _get_selector(self, tier: RouterTier) -> RouteSelector:
        """Get the selector for a tier."""
        if tier == RouterTier.FAST:
            return self._fast_selector
        elif tier == RouterTier.STANDARD and self._standard_selector:
            return self._standard_selector
        elif tier == RouterTier.COMPLEX and self._complex_selector:
            return self._complex_selector
        return self._fast_selector  # Fallback

    async def analyze(
        self,
        problem: str,
        force_tier: RouterTier | None = None,
    ) -> ProblemAnalysis:
        """Analyze a problem without selecting a route.

        Useful for understanding how the router perceives a problem.

        Args:
            problem: The problem text
            force_tier: Force a specific analysis tier

        Returns:
            Problem analysis result
        """
        tier = self._select_tier(problem, force_tier=force_tier)
        analyzer = self._get_analyzer(tier)
        return await analyzer.analyze(problem)

    async def route(
        self,
        problem: str,
        budget: ResourceBudget | None = None,
        force_tier: RouterTier | None = None,
        use_cache: bool = True,
    ) -> RouterResult:
        """Route a problem to the optimal method/pipeline.

        This is the main entry point for routing decisions.

        Args:
            problem: The problem text to route
            budget: Optional resource constraints
            force_tier: Force a specific routing tier
            use_cache: Whether to use cached results (default True)

        Returns:
            Complete routing result with primary and fallback routes
        """
        # Check cache first (FastMCP v2.14+ feature)
        cache_key = None
        if self._cache and use_cache:
            cache_key = self._make_cache_key(
                "route",
                problem,
                budget_max_latency=budget.max_latency_ms if budget else None,
                budget_max_tokens=budget.max_tokens if budget else None,
                budget_prefer_speed=budget.prefer_speed if budget else None,
                budget_prefer_quality=budget.prefer_quality if budget else None,
                force_tier=force_tier.value if force_tier else None,
            )
            cached = self._cache.get(cache_key)
            if cached is not None and isinstance(cached, RouterResult):
                return cast("RouterResult", cached)

        start = time.perf_counter()

        # Select tier
        tier = self._select_tier(problem, budget, force_tier)

        # Analyze problem
        analyzer = self._get_analyzer(tier)
        analysis = await analyzer.analyze(problem)

        # Select routes
        selector = self._get_selector(tier)
        decisions = await selector.select(analysis, budget)

        total_latency_ms = (time.perf_counter() - start) * 1000

        # Build result
        if not decisions:
            # Fallback to chain_of_thought
            primary = RouteDecision(
                route_type=RouteType.SINGLE_METHOD,
                method_id="chain_of_thought",
                score=0.5,
                confidence=0.5,
                reasoning="Fallback to default method",
                router_tier=tier,
                latency_ms=total_latency_ms,
            )
            fallbacks: tuple[RouteDecision, ...] = ()
        else:
            primary = decisions[0]
            fallbacks = tuple(decisions[1:])

        # Estimate resources for primary route
        resource_estimate = self._estimate_resources(primary, analysis)

        result = RouterResult(
            problem_analysis=analysis,
            primary_route=primary,
            fallback_routes=fallbacks,
            resource_estimate=resource_estimate,
            total_latency_ms=total_latency_ms,
        )

        # Cache the result
        if self._cache and cache_key:
            self._cache.set(cache_key, result, ttl=self._cache_routing_ttl)

        return result

    def _estimate_resources(
        self,
        route: RouteDecision,
        analysis: ProblemAnalysis,
    ) -> ResourceEstimate:
        """Estimate resources for a route."""
        base_tokens = 1000
        base_latency = 1000
        base_thoughts = 3

        complexity_multiplier = 0.5 + (analysis.complexity / 10)

        if route.route_type == "single_method":
            return ResourceEstimate(
                estimated_tokens=int(base_tokens * complexity_multiplier),
                estimated_latency_ms=int(base_latency * complexity_multiplier),
                estimated_thoughts=int(base_thoughts * complexity_multiplier),
                confidence=0.7,
            )
        elif route.route_type == "pipeline_template":
            return ResourceEstimate(
                estimated_tokens=int(base_tokens * complexity_multiplier * 2),
                estimated_latency_ms=int(base_latency * complexity_multiplier * 1.5),
                estimated_thoughts=int(base_thoughts * complexity_multiplier * 2),
                confidence=0.6,
            )
        else:
            return ResourceEstimate(
                estimated_tokens=int(base_tokens * complexity_multiplier * 3),
                estimated_latency_ms=int(base_latency * complexity_multiplier * 2),
                estimated_thoughts=int(base_thoughts * complexity_multiplier * 3),
                confidence=0.5,
            )

    async def recommend(
        self,
        problem: str,
        budget: ResourceBudget | None = None,
        max_recommendations: int = 5,
        use_cache: bool = True,
    ) -> list[RouteDecision]:
        """Get multiple route recommendations.

        Useful for presenting options to users.

        Args:
            problem: The problem text
            budget: Optional resource constraints
            max_recommendations: Maximum recommendations to return
            use_cache: Whether to use cached results (default True)

        Returns:
            List of route recommendations sorted by score
        """
        # Check cache first (FastMCP v2.14+ feature)
        cache_key = None
        if self._cache and use_cache:
            cache_key = self._make_cache_key(
                "recommend",
                problem,
                max_recommendations=max_recommendations,
            )
            cached = self._cache.get(cache_key)
            if cached is not None and isinstance(cached, list):
                return list(cached)

        result = await self.route(problem, budget, use_cache=use_cache)
        recommendations = [result.primary_route, *result.fallback_routes]
        recommendations = recommendations[:max_recommendations]

        # Cache the result
        if self._cache and cache_key:
            self._cache.set(cache_key, recommendations, ttl=self._cache_recommend_ttl)

        return recommendations

    def update_context(self, ctx: Context[Any, Any, Any]) -> None:
        """Update the MCP context for LLM-based routing.

        Call this when the context changes (e.g., new session).

        Args:
            ctx: New MCP context
        """
        self._ctx = ctx

        # Update complex analyzer and selector with cache support
        if self._enable_llm_routing:
            self._complex_analyzer = ComplexProblemAnalyzer(
                ctx=ctx,
                cache=self._cache,
                cache_ttl=self._cache_analysis_ttl,
            )
            self._complex_selector = ComplexRouteSelector(self._registry, ctx)
