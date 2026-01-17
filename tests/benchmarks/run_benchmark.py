#!/usr/bin/env python3
"""Run the benchmark suite against the Reasoning Router.

This script evaluates Phase 1 router (Fast tier) on the 100-query benchmark
suite and establishes accuracy baselines for the go/no-go ML tier decision.

Task 1.5.3.1: Run Phase 1 router on benchmark suite
Task 1.5.3.2: Establish accuracy baseline per domain

Usage:
    python -m tests.benchmarks.run_benchmark
    # Or from project root:
    uv run python tests/benchmarks/run_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from reasoning_mcp.router.evaluation import (
    EvaluationMetrics,
    RouterEvaluationHarness,
)
from reasoning_mcp.router.models import ProblemDomain, RouterTier
from reasoning_mcp.router.router import ReasoningRouter
from tests.benchmarks.queries import (
    ALL_BENCHMARK_QUERIES,
    get_benchmark_statistics,
    get_queries_by_domain,
)


async def run_full_evaluation() -> tuple[EvaluationMetrics, dict]:
    """Run the full benchmark evaluation.

    Returns:
        Tuple of (metrics, detailed_results)
    """
    # Initialize router (Fast tier only for Phase 1)
    router = ReasoningRouter(
        enable_ml_routing=False,  # Phase 1 - no ML
        enable_llm_routing=False,  # Phase 1 - no LLM
        default_tier=RouterTier.FAST,
    )

    # Initialize evaluation harness
    harness = RouterEvaluationHarness(router)

    print(f"Running benchmark on {len(ALL_BENCHMARK_QUERIES)} queries...")
    print("-" * 60)

    # Run evaluation
    metrics = await harness.evaluate(ALL_BENCHMARK_QUERIES)

    # Get baseline comparisons
    baselines = await harness.compare_to_baselines(ALL_BENCHMARK_QUERIES)

    # Get individual results for detailed analysis
    results = harness.get_results()

    # Compute per-domain metrics
    domain_metrics = {}
    for domain in ProblemDomain:
        domain_queries = get_queries_by_domain(domain)
        if domain_queries:
            domain_results = [r for r in results if r.query.expected_domain == domain]
            if domain_results:
                correct_domain = sum(1 for r in domain_results if r.domain_correct)
                correct_intent = sum(1 for r in domain_results if r.intent_correct)
                method_hits = sum(1 for r in domain_results if r.method_in_recommendations)

                domain_metrics[domain.value] = {
                    "count": len(domain_results),
                    "domain_accuracy": correct_domain / len(domain_results),
                    "intent_accuracy": correct_intent / len(domain_results),
                    "method_hit_rate": method_hits / len(domain_results),
                    "avg_latency_ms": sum(r.latency_ms for r in domain_results)
                    / len(domain_results),
                }

    # Generate report
    report = harness.generate_report(metrics)
    print(report)

    # Compute go/no-go criteria
    go_nogo = harness.compute_go_nogo_criteria(metrics)
    print("\n" + "=" * 60)
    print("GO/NO-GO CRITERIA DETAILS")
    print("=" * 60)
    for key, value in go_nogo.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_stats": get_benchmark_statistics(),
        "overall_metrics": {
            "domain_accuracy": metrics.domain_accuracy,
            "intent_accuracy": metrics.intent_accuracy,
            "method_hit_rate": metrics.method_hit_rate,
            "pipeline_accuracy": metrics.pipeline_accuracy,
            "complexity_mae": metrics.complexity_mae,
            "complexity_rmse": metrics.complexity_rmse,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p50_latency_ms": metrics.p50_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "total_queries": metrics.total_queries,
            "successful_routes": metrics.successful_routes,
            "failed_routes": metrics.failed_routes,
        },
        "domain_metrics": domain_metrics,
        "per_class_metrics": {
            "domain_precision": metrics.domain_precision_per_class,
            "domain_recall": metrics.domain_recall_per_class,
            "domain_f1": metrics.domain_f1_per_class,
            "intent_precision": metrics.intent_precision_per_class,
            "intent_recall": metrics.intent_recall_per_class,
            "intent_f1": metrics.intent_f1_per_class,
        },
        "baselines": [
            {
                "name": b.baseline_name,
                "description": b.description,
                "method_hit_rate": b.metrics.method_hit_rate,
                "domain_accuracy": b.metrics.domain_accuracy,
                "intent_accuracy": b.metrics.intent_accuracy,
            }
            for b in baselines
        ],
        "go_nogo": go_nogo,
    }

    return metrics, detailed_results


async def run_tier_comparison() -> dict:
    """Compare Fast, Standard, and Complex tiers (ablation study).

    Note: Currently only Fast tier is implemented, so this will
    show FAST for all since ML/LLM routing is disabled.

    Returns:
        Dict of tier -> metrics
    """
    router = ReasoningRouter(
        enable_ml_routing=False,
        enable_llm_routing=False,
    )
    harness = RouterEvaluationHarness(router)

    # Run ablation on subset for speed
    subset = ALL_BENCHMARK_QUERIES[:20]  # Just 20 queries for ablation

    print("\nRunning tier ablation study (20 queries)...")
    results = await harness.ablation_study(subset)

    tier_comparison = {}
    for tier, metrics in results.items():
        tier_comparison[tier.value] = {
            "domain_accuracy": metrics.domain_accuracy,
            "intent_accuracy": metrics.intent_accuracy,
            "avg_latency_ms": metrics.avg_latency_ms,
            "successful_routes": metrics.successful_routes,
        }
        print(
            f"  {tier.value}: domain={metrics.domain_accuracy:.1%} "
            f"intent={metrics.intent_accuracy:.1%} "
            f"latency={metrics.avg_latency_ms:.2f}ms"
        )

    return tier_comparison


def save_results(detailed_results: dict, output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def print_summary(metrics: EvaluationMetrics, go_nogo: dict) -> None:
    """Print a summary for easy reference."""
    print("\n" + "=" * 60)
    print("PHASE 1.5 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Domain Accuracy:    {metrics.domain_accuracy:.1%}")
    print(f"Intent Accuracy:    {metrics.intent_accuracy:.1%}")
    print(f"Method Hit Rate:    {metrics.method_hit_rate:.1%}")
    print(f"Average Accuracy:   {go_nogo['avg_accuracy']:.1%}")
    print(f"Avg Latency:        {metrics.avg_latency_ms:.2f}ms")
    print(f"p95 Latency:        {metrics.p95_latency_ms:.2f}ms")
    print("-" * 60)
    print(f"ML Tier Justified:  {go_nogo['ml_tier_justified']}")
    print(f"Reason:             {go_nogo['reason']}")
    print(f"Recommendation:     {go_nogo['recommendation']}")
    print("=" * 60)


async def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("REASONING ROUTER BENCHMARK - Phase 1.5")
    print("=" * 60)
    print()

    # Run full evaluation
    metrics, detailed_results = await run_full_evaluation()

    # Run tier comparison
    tier_comparison = await run_tier_comparison()
    detailed_results["tier_comparison"] = tier_comparison

    # Save results
    output_path = Path(__file__).parent / "results" / "phase1_baseline.json"
    save_results(detailed_results, output_path)

    # Print summary
    print_summary(metrics, detailed_results["go_nogo"])

    # Return exit code based on go/no-go
    # 0 = success (baseline established)
    # Non-zero could indicate issues
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
