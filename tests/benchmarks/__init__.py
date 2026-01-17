"""Benchmark suite for Reasoning Router evaluation.

This package provides:
- 100-query benchmark suite stratified by domain
- Ground truth labels for domain, intent, complexity
- Recommended methods and pipelines for each query
- Utilities for loading and filtering benchmark queries

Task A4: Test infrastructure setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.router.evaluation import BenchmarkQuery

# Import benchmark queries when available
try:
    from tests.benchmarks.queries import (
        ALL_BENCHMARK_QUERIES,
        ANALYTICAL_QUERIES,
        CAUSAL_QUERIES,
        CODE_QUERIES,
        CREATIVE_QUERIES,
        DECISION_QUERIES,
        ETHICAL_QUERIES,
        GENERAL_QUERIES,
        MATH_QUERIES,
        SCIENTIFIC_QUERIES,
        get_queries_by_difficulty,
        get_queries_by_domain,
        get_queries_by_tag,
    )

    __all__ = [
        # Full benchmark suite
        "ALL_BENCHMARK_QUERIES",
        # Domain-specific query sets
        "MATH_QUERIES",
        "CODE_QUERIES",
        "ETHICAL_QUERIES",
        "CREATIVE_QUERIES",
        "ANALYTICAL_QUERIES",
        "CAUSAL_QUERIES",
        "DECISION_QUERIES",
        "SCIENTIFIC_QUERIES",
        "GENERAL_QUERIES",
        # Utility functions
        "get_queries_by_domain",
        "get_queries_by_difficulty",
        "get_queries_by_tag",
    ]
except ImportError:
    # Queries not yet defined
    ALL_BENCHMARK_QUERIES: list[BenchmarkQuery] = []
    __all__ = ["ALL_BENCHMARK_QUERIES"]
