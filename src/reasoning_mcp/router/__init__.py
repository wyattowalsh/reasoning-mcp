"""Reasoning Router package.

This package provides intelligent routing of problems to the optimal
reasoning method, pipeline, or ensemble based on problem analysis.

The router uses a multi-tier strategy:
- Tier 1 (Fast, <5ms): Embedding similarity + regex patterns
- Tier 2 (Standard, ~20ms): ML classifiers + matrix factorization
- Tier 3 (Complex, ~200ms): LLM-based analysis + pipeline synthesis
"""

from reasoning_mcp.router.evaluation import (
    BenchmarkQuery,
    EvaluationMetrics,
    EvaluationResult,
    RouterEvaluationHarness,
)
from reasoning_mcp.router.models import (
    EmbeddingProvider,
    LearningMode,
    ProblemAnalysis,
    ProblemDomain,
    ProblemIntent,
    RequiredCapability,
    ResourceBudget,
    ResourceEstimate,
    RouteDecision,
    RouteOutcome,
    RouterFeedback,
    RouterMetrics,
    RouterResult,
    RouterTier,
    RouteType,
)
from reasoning_mcp.router.router import ReasoningRouter
from reasoning_mcp.router.telemetry import (
    RoutingTelemetry,
    TelemetryLogger,
    compute_query_hash,
    configure_telemetry,
    create_telemetry_from_route,
    get_telemetry_logger,
)

__all__ = [
    # Core router
    "ReasoningRouter",
    # Enums
    "ProblemDomain",
    "ProblemIntent",
    "RequiredCapability",
    "RouteType",
    "RouterTier",
    "EmbeddingProvider",
    "LearningMode",
    # Models
    "ProblemAnalysis",
    "ResourceBudget",
    "ResourceEstimate",
    "RouteDecision",
    "RouterResult",
    "RouteOutcome",
    "RouterFeedback",
    "RouterMetrics",
    # Telemetry (Phase 1.5)
    "RoutingTelemetry",
    "TelemetryLogger",
    "compute_query_hash",
    "configure_telemetry",
    "create_telemetry_from_route",
    "get_telemetry_logger",
    # Evaluation (Phase 1.5)
    "BenchmarkQuery",
    "EvaluationMetrics",
    "EvaluationResult",
    "RouterEvaluationHarness",
]
