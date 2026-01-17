"""Evaluation harness for the Reasoning Router.

This module provides a comprehensive evaluation framework for measuring
router performance against ground truth benchmarks.

Task 1.5.2.1: Create RouterEvaluationHarness class skeleton
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from reasoning_mcp.router.models import ProblemDomain, ProblemIntent, RouterTier

if TYPE_CHECKING:
    from reasoning_mcp.router.router import ReasoningRouter


@dataclass
class BenchmarkQuery:
    """A single benchmark query with ground truth labels.

    Each query in the benchmark suite includes:
    - The problem text
    - Expected domain classification
    - Expected intent
    - Expected complexity range
    - Recommended methods/pipelines
    """

    query: str
    expected_domain: ProblemDomain
    expected_intent: ProblemIntent
    expected_complexity: int  # 1-10
    recommended_methods: list[str]
    recommended_pipeline: str | None = None
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvaluationMetrics:
    """Metrics computed from evaluation run."""

    # Domain classification
    domain_accuracy: float = 0.0
    domain_precision_per_class: dict[str, float] = field(default_factory=dict)
    domain_recall_per_class: dict[str, float] = field(default_factory=dict)
    domain_f1_per_class: dict[str, float] = field(default_factory=dict)

    # Intent classification
    intent_accuracy: float = 0.0
    intent_precision_per_class: dict[str, float] = field(default_factory=dict)
    intent_recall_per_class: dict[str, float] = field(default_factory=dict)
    intent_f1_per_class: dict[str, float] = field(default_factory=dict)

    # Complexity estimation
    complexity_mae: float = 0.0  # Mean Absolute Error
    complexity_rmse: float = 0.0  # Root Mean Square Error
    complexity_correlation: float = 0.0  # Pearson correlation

    # Method recommendation
    method_hit_rate: float = 0.0  # % of times recommended method was in top-3
    method_mrr: float = 0.0  # Mean Reciprocal Rank
    pipeline_accuracy: float = 0.0  # % of correct pipeline recommendations

    # Latency
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Sample counts
    total_queries: int = 0
    successful_routes: int = 0
    failed_routes: int = 0


@dataclass
class EvaluationResult:
    """Result of a single query evaluation."""

    query: BenchmarkQuery
    actual_domain: ProblemDomain | None = None
    actual_intent: ProblemIntent | None = None
    actual_complexity: int | None = None
    recommended_method: str | None = None
    recommended_pipeline: str | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    tier_used: RouterTier = RouterTier.FAST

    # Correctness flags
    domain_correct: bool = False
    intent_correct: bool = False
    method_in_recommendations: bool = False
    pipeline_correct: bool = False
    error: str | None = None


@dataclass
class BaselineResult:
    """Result from a baseline comparison."""

    baseline_name: str
    metrics: EvaluationMetrics
    description: str = ""


class RouterEvaluationHarness:
    """Evaluation harness for measuring router performance.

    This harness provides:
    - Precision/recall/F1 metrics per domain
    - Latency percentile measurements
    - Baseline comparisons (random, popularity-weighted)
    - Tier-specific ablation studies

    Usage:
        harness = RouterEvaluationHarness(router)
        metrics = await harness.evaluate(benchmark_queries)
        report = harness.generate_report(metrics)
    """

    def __init__(self, router: ReasoningRouter | None = None):
        """Initialize the evaluation harness.

        Args:
            router: The ReasoningRouter to evaluate. If None, must be set before evaluation.
        """
        self._router = router
        self._results: list[EvaluationResult] = []
        self._baseline_results: list[BaselineResult] = []

    def set_router(self, router: ReasoningRouter) -> None:
        """Set the router to evaluate.

        Args:
            router: The ReasoningRouter instance
        """
        self._router = router

    async def evaluate(
        self,
        queries: list[BenchmarkQuery],
        force_tier: RouterTier | None = None,
    ) -> EvaluationMetrics:
        """Evaluate the router on a benchmark suite.

        Args:
            queries: List of benchmark queries with ground truth
            force_tier: Optional tier to force for all evaluations

        Returns:
            EvaluationMetrics with computed performance metrics

        Raises:
            ValueError: If router is not set
        """
        if self._router is None:
            raise ValueError("Router not set. Call set_router() first.")

        self._results = []

        for query in queries:
            result = await self._evaluate_single(query, force_tier)
            self._results.append(result)

        return self._compute_metrics(self._results)

    async def _evaluate_single(
        self,
        query: BenchmarkQuery,
        force_tier: RouterTier | None = None,
    ) -> EvaluationResult:
        """Evaluate a single benchmark query.

        Args:
            query: The benchmark query to evaluate
            force_tier: Optional tier to force

        Returns:
            EvaluationResult with actual vs expected comparison
        """
        result = EvaluationResult(query=query)

        try:
            # Route the query
            assert self._router is not None  # Guaranteed by evaluate() check
            route_result = await self._router.route(query.query, force_tier=force_tier)

            # Extract results
            result.actual_domain = route_result.problem_analysis.primary_domain
            result.actual_intent = route_result.problem_analysis.intent
            result.actual_complexity = route_result.problem_analysis.complexity
            result.confidence = route_result.primary_route.confidence
            result.latency_ms = route_result.total_latency_ms
            result.tier_used = route_result.primary_route.router_tier

            # Get recommended method/pipeline
            if route_result.primary_route.method_id:
                result.recommended_method = route_result.primary_route.method_id
            if route_result.primary_route.pipeline_id:
                result.recommended_pipeline = route_result.primary_route.pipeline_id

            # Compute correctness
            result.domain_correct = result.actual_domain == query.expected_domain
            result.intent_correct = result.actual_intent == query.expected_intent
            result.method_in_recommendations = (
                result.recommended_method in query.recommended_methods
                if result.recommended_method
                else False
            )
            result.pipeline_correct = result.recommended_pipeline == query.recommended_pipeline

        except Exception as e:
            result.error = str(e)

        return result

    def _compute_metrics(self, results: list[EvaluationResult]) -> EvaluationMetrics:
        """Compute aggregate metrics from evaluation results.

        Args:
            results: List of individual evaluation results

        Returns:
            EvaluationMetrics with computed values
        """
        metrics = EvaluationMetrics()
        metrics.total_queries = len(results)

        successful = [r for r in results if r.error is None]
        metrics.successful_routes = len(successful)
        metrics.failed_routes = len(results) - len(successful)

        if not successful:
            return metrics

        # Domain accuracy
        domain_correct = sum(1 for r in successful if r.domain_correct)
        metrics.domain_accuracy = domain_correct / len(successful)

        # Intent accuracy
        intent_correct = sum(1 for r in successful if r.intent_correct)
        metrics.intent_accuracy = intent_correct / len(successful)

        # Method hit rate
        method_hits = sum(1 for r in successful if r.method_in_recommendations)
        metrics.method_hit_rate = method_hits / len(successful)

        # Pipeline accuracy
        pipeline_correct = sum(1 for r in successful if r.pipeline_correct)
        metrics.pipeline_accuracy = pipeline_correct / len(successful) if successful else 0.0

        # Complexity MAE
        complexity_errors = [
            abs(r.actual_complexity - r.query.expected_complexity)
            for r in successful
            if r.actual_complexity is not None
        ]
        if complexity_errors:
            metrics.complexity_mae = statistics.mean(complexity_errors)
            sum_squared = sum(e**2 for e in complexity_errors)
            metrics.complexity_rmse = (sum_squared / len(complexity_errors)) ** 0.5

        # Latency percentiles
        latencies = sorted(r.latency_ms for r in successful)
        if latencies:
            metrics.avg_latency_ms = statistics.mean(latencies)
            metrics.p50_latency_ms = self._percentile(latencies, 50)
            metrics.p95_latency_ms = self._percentile(latencies, 95)
            metrics.p99_latency_ms = self._percentile(latencies, 99)

        # Per-class metrics
        metrics.domain_precision_per_class = self._compute_precision_per_class(
            successful, "domain", ProblemDomain
        )
        metrics.domain_recall_per_class = self._compute_recall_per_class(
            successful, "domain", ProblemDomain
        )
        metrics.domain_f1_per_class = self._compute_f1_per_class(
            metrics.domain_precision_per_class, metrics.domain_recall_per_class
        )

        metrics.intent_precision_per_class = self._compute_precision_per_class(
            successful, "intent", ProblemIntent
        )
        metrics.intent_recall_per_class = self._compute_recall_per_class(
            successful, "intent", ProblemIntent
        )
        metrics.intent_f1_per_class = self._compute_f1_per_class(
            metrics.intent_precision_per_class, metrics.intent_recall_per_class
        )

        return metrics

    def _percentile(self, sorted_values: list[float], p: int) -> float:
        """Compute percentile from sorted values."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def _compute_precision_per_class(
        self, results: list[EvaluationResult], field: str, enum_class: type
    ) -> dict[str, float]:
        """Compute precision for each class."""
        precision = {}
        for cls in enum_class:
            if field == "domain":
                predicted_as_cls = [r for r in results if r.actual_domain == cls]
                true_positives = [r for r in predicted_as_cls if r.query.expected_domain == cls]
            else:  # intent
                predicted_as_cls = [r for r in results if r.actual_intent == cls]
                true_positives = [r for r in predicted_as_cls if r.query.expected_intent == cls]

            if predicted_as_cls:
                precision[cls.value] = len(true_positives) / len(predicted_as_cls)
            else:
                precision[cls.value] = 0.0
        return precision

    def _compute_recall_per_class(
        self, results: list[EvaluationResult], field: str, enum_class: type
    ) -> dict[str, float]:
        """Compute recall for each class."""
        recall = {}
        for cls in enum_class:
            if field == "domain":
                actual_cls = [r for r in results if r.query.expected_domain == cls]
                true_positives = [r for r in actual_cls if r.actual_domain == cls]
            else:  # intent
                actual_cls = [r for r in results if r.query.expected_intent == cls]
                true_positives = [r for r in actual_cls if r.actual_intent == cls]

            if actual_cls:
                recall[cls.value] = len(true_positives) / len(actual_cls)
            else:
                recall[cls.value] = 0.0
        return recall

    def _compute_f1_per_class(
        self, precision: dict[str, float], recall: dict[str, float]
    ) -> dict[str, float]:
        """Compute F1 score from precision and recall."""
        f1 = {}
        for cls in precision:
            p, r = precision[cls], recall.get(cls, 0.0)
            if p + r > 0:
                f1[cls] = 2 * p * r / (p + r)
            else:
                f1[cls] = 0.0
        return f1

    async def evaluate_tier(
        self, queries: list[BenchmarkQuery], tier: RouterTier
    ) -> EvaluationMetrics:
        """Evaluate a specific tier in isolation.

        Args:
            queries: Benchmark queries
            tier: The tier to evaluate

        Returns:
            Metrics for that tier only
        """
        return await self.evaluate(queries, force_tier=tier)

    async def ablation_study(
        self, queries: list[BenchmarkQuery]
    ) -> dict[RouterTier, EvaluationMetrics]:
        """Run ablation study comparing all tiers.

        Args:
            queries: Benchmark queries

        Returns:
            Dict mapping tier to its metrics
        """
        results = {}
        for tier in RouterTier:
            results[tier] = await self.evaluate_tier(queries, tier)
        return results

    async def compare_to_baselines(self, queries: list[BenchmarkQuery]) -> list[BaselineResult]:
        """Compare router to baseline strategies.

        Baselines:
        - Random: Randomly select method from available methods
        - Popularity: Select most frequently used method
        - Domain-Only: Use domain→method mapping without analysis

        Args:
            queries: Benchmark queries

        Returns:
            List of baseline results for comparison
        """
        self._baseline_results = []

        # Random baseline
        random_metrics = self._evaluate_random_baseline(queries)
        self._baseline_results.append(
            BaselineResult(
                baseline_name="random",
                metrics=random_metrics,
                description="Random method selection from available methods",
            )
        )

        # Popularity baseline (always selects chain_of_thought)
        popularity_metrics = self._evaluate_popularity_baseline(queries)
        self._baseline_results.append(
            BaselineResult(
                baseline_name="popularity",
                metrics=popularity_metrics,
                description="Always selects most popular method (chain_of_thought)",
            )
        )

        return self._baseline_results

    def _evaluate_random_baseline(self, queries: list[BenchmarkQuery]) -> EvaluationMetrics:
        """Evaluate random baseline (for comparison)."""
        metrics = EvaluationMetrics()
        metrics.total_queries = len(queries)

        # With random selection, expected accuracy is ~1/n where n is number of methods
        # For simplicity, estimate method hit rate at ~10% (assuming ~10 reasonable methods)
        metrics.method_hit_rate = 0.10
        metrics.domain_accuracy = 1.0 / len(ProblemDomain)  # ~8%
        metrics.intent_accuracy = 1.0 / len(ProblemIntent)  # ~10%
        metrics.avg_latency_ms = 0.1  # Instant (no real routing)

        return metrics

    def _evaluate_popularity_baseline(self, queries: list[BenchmarkQuery]) -> EvaluationMetrics:
        """Evaluate popularity baseline (always chain_of_thought)."""
        metrics = EvaluationMetrics()
        metrics.total_queries = len(queries)

        # Count how many queries have chain_of_thought in recommendations
        hits = sum(1 for q in queries if "chain_of_thought" in q.recommended_methods)
        metrics.method_hit_rate = hits / len(queries) if queries else 0.0
        metrics.avg_latency_ms = 0.1  # Instant (no analysis)

        return metrics

    def compute_go_nogo_criteria(self, metrics: EvaluationMetrics) -> dict[str, Any]:
        """Check if ML tier investment is justified based on metrics.

        Criteria from spec:
        - Accuracy < 80% (domain + intent avg)
        - Confidence < 0.75 (would need telemetry data)
        - User override rate > 15% (would need telemetry data)

        Args:
            metrics: Evaluation metrics

        Returns:
            Dictionary with go/no-go decision and reasoning
        """
        accuracy = (metrics.domain_accuracy + metrics.intent_accuracy) / 2

        # From spec: ML_TIER_JUSTIFIED if accuracy < 0.80
        ml_justified = accuracy < 0.80

        return {
            "avg_accuracy": accuracy,
            "domain_accuracy": metrics.domain_accuracy,
            "intent_accuracy": metrics.intent_accuracy,
            "method_hit_rate": metrics.method_hit_rate,
            "ml_tier_justified": ml_justified,
            "reason": (
                f"ML tier justified: accuracy {accuracy:.1%} < 80%"
                if ml_justified
                else f"ML tier NOT justified: accuracy {accuracy:.1%} >= 80%"
            ),
            "recommendation": (
                "Proceed to Phase 3 (ML-Based Routing)"
                if ml_justified
                else "Skip Phase 3, continue with Fast + LLM tiers"
            ),
        }

    def generate_report(
        self,
        metrics: EvaluationMetrics,
        include_baselines: bool = True,
    ) -> str:
        """Generate a human-readable evaluation report.

        Args:
            metrics: Computed evaluation metrics
            include_baselines: Include baseline comparisons

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "REASONING ROUTER EVALUATION REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Total queries evaluated: {metrics.total_queries}",
            f"Successful routes: {metrics.successful_routes}",
            f"Failed routes: {metrics.failed_routes}",
            "",
            "CLASSIFICATION ACCURACY",
            "-" * 40,
            f"Domain accuracy: {metrics.domain_accuracy:.1%}",
            f"Intent accuracy: {metrics.intent_accuracy:.1%}",
            f"Method hit rate: {metrics.method_hit_rate:.1%}",
            f"Pipeline accuracy: {metrics.pipeline_accuracy:.1%}",
            "",
            "COMPLEXITY ESTIMATION",
            "-" * 40,
            f"MAE: {metrics.complexity_mae:.2f}",
            f"RMSE: {metrics.complexity_rmse:.2f}",
            "",
            "LATENCY",
            "-" * 40,
            f"Average: {metrics.avg_latency_ms:.2f}ms",
            f"p50: {metrics.p50_latency_ms:.2f}ms",
            f"p95: {metrics.p95_latency_ms:.2f}ms",
            f"p99: {metrics.p99_latency_ms:.2f}ms",
            "",
        ]

        if include_baselines and self._baseline_results:
            lines.extend(
                [
                    "BASELINE COMPARISONS",
                    "-" * 40,
                ]
            )
            for baseline in self._baseline_results:
                lines.extend(
                    [
                        f"  {baseline.baseline_name}:",
                        f"    Method hit rate: {baseline.metrics.method_hit_rate:.1%}",
                        f"    Description: {baseline.description}",
                    ]
                )
            lines.append("")

        # Go/no-go decision
        go_nogo = self.compute_go_nogo_criteria(metrics)
        lines.extend(
            [
                "GO/NO-GO DECISION",
                "-" * 40,
                f"Average accuracy: {go_nogo['avg_accuracy']:.1%}",
                f"Decision: {go_nogo['reason']}",
                f"Recommendation: {go_nogo['recommendation']}",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    def get_results(self) -> list[EvaluationResult]:
        """Get individual evaluation results."""
        return self._results

    def get_baseline_results(self) -> list[BaselineResult]:
        """Get baseline comparison results."""
        return self._baseline_results
