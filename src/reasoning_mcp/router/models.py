"""Data models for the Reasoning Router.

This module defines all the core data structures used by the router
for problem analysis, routing decisions, and learning.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProblemDomain(StrEnum):
    """Domain classification for problems."""

    MATHEMATICAL = "mathematical"
    CODE = "code"
    ETHICAL = "ethical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CAUSAL = "causal"
    DECISION = "decision"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    MEDICAL = "medical"
    PHILOSOPHICAL = "philosophical"
    GENERAL = "general"


class ProblemIntent(StrEnum):
    """Intent classification for problems."""

    SOLVE = "solve"  # Find a solution
    ANALYZE = "analyze"  # Break down and understand
    EVALUATE = "evaluate"  # Assess quality/correctness
    GENERATE = "generate"  # Create something new
    EXPLAIN = "explain"  # Provide understanding
    VERIFY = "verify"  # Check correctness
    OPTIMIZE = "optimize"  # Improve existing solution
    DEBUG = "debug"  # Find and fix issues
    COMPARE = "compare"  # Contrast alternatives
    SYNTHESIZE = "synthesize"  # Combine multiple inputs


class RequiredCapability(StrEnum):
    """Capabilities that may be required for a problem."""

    BRANCHING = "branching"  # Explore multiple paths
    ITERATION = "iteration"  # Refine iteratively
    EXTERNAL_TOOLS = "external_tools"  # Execute code/actions
    VERIFICATION = "verification"  # Self-check reasoning
    DECOMPOSITION = "decomposition"  # Break into subproblems
    SYNTHESIS = "synthesis"  # Combine multiple views
    FORMAL_LOGIC = "formal_logic"  # Logical deduction
    CREATIVITY = "creativity"  # Novel solutions
    MEMORY = "memory"  # Track past reasoning
    MULTI_AGENT = "multi_agent"  # Multiple perspectives


class RouteType(StrEnum):
    """Type of routing decision."""

    SINGLE_METHOD = "single_method"
    PIPELINE_TEMPLATE = "pipeline_template"
    SYNTHESIZED_PIPELINE = "synthesized_pipeline"
    METHOD_ENSEMBLE = "method_ensemble"


class RouterTier(StrEnum):
    """Router tier for analysis/selection."""

    FAST = "fast"  # <5ms, embedding + regex
    STANDARD = "standard"  # ~20ms, classifiers + matrix factorization
    COMPLEX = "complex"  # ~200ms, LLM-based


class EmbeddingProvider(StrEnum):
    """Embedding model providers."""

    LOCAL_MINILM = "local:all-MiniLM-L6-v2"
    LOCAL_BGE = "local:BAAI/bge-small-en-v1.5"
    OPENAI_SMALL = "openai:text-embedding-3-small"
    OPENAI_LARGE = "openai:text-embedding-3-large"
    COHERE = "cohere:embed-english-v3.0"


class LearningMode(StrEnum):
    """Learning mode for the router."""

    OFF = "off"  # No learning, static routing
    OBSERVE = "observe"  # Collect data only
    INCREMENTAL = "incremental"  # Update after each feedback
    BATCH = "batch"  # Accumulate N samples, batch update
    MANUAL = "manual"  # Export for offline training


class ProblemAnalysis(BaseModel):
    """Comprehensive analysis of a problem for routing."""

    model_config = ConfigDict(frozen=True)

    # Core dimensions
    primary_domain: ProblemDomain
    secondary_domains: frozenset[ProblemDomain] = Field(default_factory=frozenset)
    intent: ProblemIntent

    # Complexity assessment (1-10 scale)
    complexity: int = Field(default=5, ge=1, le=10)
    ambiguity: int = Field(default=5, ge=1, le=10)
    depth_required: int = Field(default=5, ge=1, le=10)
    breadth_required: int = Field(default=5, ge=1, le=10)

    # Required capabilities
    capabilities: frozenset[RequiredCapability] = Field(default_factory=frozenset)

    # Extracted features
    keywords: frozenset[str] = Field(default_factory=frozenset)
    entities: frozenset[str] = Field(default_factory=frozenset)

    # Confidence and metadata
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    analysis_latency_ms: float = Field(default=0.0, ge=0.0)
    analyzer_tier: RouterTier = RouterTier.FAST


class ResourceBudget(BaseModel):
    """Resource constraints for routing decisions."""

    model_config = ConfigDict(frozen=True)

    max_latency_ms: int = Field(default=30000, ge=100)
    max_tokens: int = Field(default=50000, ge=1000)
    max_thoughts: int = Field(default=50, ge=1)
    max_branches: int = Field(default=10, ge=1)
    prefer_speed: bool = False
    prefer_quality: bool = False


class ResourceEstimate(BaseModel):
    """Estimated resource usage for a route."""

    model_config = ConfigDict(frozen=True)

    estimated_tokens: int = Field(default=1000, ge=0)
    estimated_latency_ms: int = Field(default=1000, ge=0)
    estimated_thoughts: int = Field(default=5, ge=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RouteDecision(BaseModel):
    """A routing decision with confidence and explanation."""

    model_config = ConfigDict(frozen=True)

    route_type: RouteType
    method_id: str | None = None  # For SINGLE_METHOD
    pipeline_id: str | None = None  # For PIPELINE_TEMPLATE
    pipeline_definition: dict[str, Any] | None = None  # For SYNTHESIZED_PIPELINE
    ensemble_methods: tuple[str, ...] = ()  # For ENSEMBLE
    ensemble_strategy: Literal["vote", "best", "aggregate", "sequential"] = "vote"

    # Scoring
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Explanation
    reasoning: str = ""
    matched_criteria: tuple[str, ...] = ()

    # Metadata
    router_tier: RouterTier = RouterTier.FAST
    latency_ms: float = Field(default=0.0, ge=0.0)


class RouterResult(BaseModel):
    """Complete result from the reasoning router."""

    model_config = ConfigDict(frozen=True)

    problem_analysis: ProblemAnalysis
    primary_route: RouteDecision
    fallback_routes: tuple[RouteDecision, ...] = ()
    resource_estimate: ResourceEstimate = Field(default_factory=ResourceEstimate)
    total_latency_ms: float = Field(default=0.0, ge=0.0)


class RouteOutcome(BaseModel):
    """Outcome of a routing decision for learning."""

    model_config = ConfigDict(frozen=True)

    route_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    route_decision: RouteDecision
    problem_analysis: ProblemAnalysis

    # Outcome metrics
    success: bool = True
    quality_score: float | None = None  # 0.0-1.0
    actual_tokens: int = 0
    actual_latency_ms: float = 0.0

    # User feedback
    user_rating: int | None = None  # 1-5
    user_feedback: str | None = None


class RouterFeedback(BaseModel):
    """Feedback for a routing decision."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    route_id: str
    rating: Literal["good", "neutral", "bad"]
    was_method_appropriate: bool | None = None
    preferred_alternative: str | None = None
    feedback_text: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class RouterMetrics(BaseModel):
    """Aggregated metrics for router performance."""

    model_config = ConfigDict(frozen=True)

    # Latency
    avg_routing_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Accuracy
    method_selection_accuracy: float = 0.0
    domain_detection_accuracy: float = 0.0
    complexity_estimation_mae: float = 0.0

    # Outcomes
    route_success_rate: float = 0.0
    avg_quality_score: float = 0.0

    # Tier usage
    fast_tier_usage: float = 0.0
    standard_tier_usage: float = 0.0
    complex_tier_usage: float = 0.0

    # Efficiency
    avg_tokens_saved: float = 0.0
    avg_latency_saved_ms: float = 0.0

    # Sample counts
    total_routes: int = 0
    total_feedback: int = 0


# Domain → Method mappings (default rules)
DOMAIN_METHOD_MAPPING: dict[ProblemDomain, tuple[str, str | None]] = {
    ProblemDomain.MATHEMATICAL: ("mathematical_reasoning", "chain_of_thought"),
    ProblemDomain.CODE: ("code_reasoning", "react"),
    ProblemDomain.ETHICAL: ("ethical_reasoning", "dialectic"),
    ProblemDomain.CREATIVE: ("lateral_thinking", "tree_of_thoughts"),
    ProblemDomain.CAUSAL: ("causal_reasoning", "counterfactual"),
    ProblemDomain.ANALYTICAL: ("chain_of_thought", "self_consistency"),
    ProblemDomain.DECISION: ("tree_of_thoughts", "mcts"),
    ProblemDomain.SCIENTIFIC: ("step_back", "mathematical_reasoning"),
    ProblemDomain.LEGAL: ("dialectic", "chain_of_thought"),
    ProblemDomain.MEDICAL: ("chain_of_thought", "step_back"),
    ProblemDomain.PHILOSOPHICAL: ("socratic", "dialectic"),
    ProblemDomain.GENERAL: ("chain_of_thought", "sequential_thinking"),
}

# Intent → Capability mappings
INTENT_CAPABILITY_MAPPING: dict[ProblemIntent, frozenset[RequiredCapability]] = {
    ProblemIntent.SOLVE: frozenset({RequiredCapability.DECOMPOSITION}),
    ProblemIntent.ANALYZE: frozenset({RequiredCapability.DECOMPOSITION}),
    ProblemIntent.EVALUATE: frozenset({RequiredCapability.VERIFICATION}),
    ProblemIntent.GENERATE: frozenset({RequiredCapability.CREATIVITY}),
    ProblemIntent.EXPLAIN: frozenset(),
    ProblemIntent.VERIFY: frozenset(
        {RequiredCapability.VERIFICATION, RequiredCapability.FORMAL_LOGIC}
    ),
    ProblemIntent.OPTIMIZE: frozenset({RequiredCapability.ITERATION}),
    ProblemIntent.DEBUG: frozenset(
        {RequiredCapability.EXTERNAL_TOOLS, RequiredCapability.ITERATION}
    ),
    ProblemIntent.COMPARE: frozenset({RequiredCapability.BRANCHING}),
    ProblemIntent.SYNTHESIZE: frozenset(
        {RequiredCapability.SYNTHESIS, RequiredCapability.MULTI_AGENT}
    ),
}
