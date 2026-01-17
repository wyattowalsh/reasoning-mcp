"""Unit tests for the evaluation harness.

Tests for:
- Task 1.5.2.1: RouterEvaluationHarness class
- Task 1.5.2.3: Precision/recall/F1 per domain
- Task 1.5.2.4: Latency percentile measurements
- Task 1.5.2.5: Baseline comparisons
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.router.evaluation import (
    BaselineResult,
    BenchmarkQuery,
    EvaluationMetrics,
    EvaluationResult,
    RouterEvaluationHarness,
)
from reasoning_mcp.router.models import (
    ProblemAnalysis,
    ProblemDomain,
    ProblemIntent,
    ResourceEstimate,
    RouteDecision,
    RouterResult,
    RouterTier,
    RouteType,
)


class TestBenchmarkQuery:
    """Tests for the BenchmarkQuery dataclass."""

    def test_create_basic_query(self):
        """Test creating a basic benchmark query."""
        query = BenchmarkQuery(
            query="What is 2 + 2?",
            expected_domain=ProblemDomain.MATHEMATICAL,
            expected_intent=ProblemIntent.SOLVE,
            expected_complexity=2,
            recommended_methods=["chain_of_thought", "mathematical_reasoning"],
        )

        assert query.query == "What is 2 + 2?"
        assert query.expected_domain == ProblemDomain.MATHEMATICAL
        assert query.expected_intent == ProblemIntent.SOLVE
        assert query.expected_complexity == 2
        assert "chain_of_thought" in query.recommended_methods

    def test_query_with_pipeline(self):
        """Test query with recommended pipeline."""
        query = BenchmarkQuery(
            query="Is it ethical to lie?",
            expected_domain=ProblemDomain.ETHICAL,
            expected_intent=ProblemIntent.ANALYZE,
            expected_complexity=6,
            recommended_methods=["ethical_reasoning", "dialectic"],
            recommended_pipeline="ethical_multi_view",
        )

        assert query.recommended_pipeline == "ethical_multi_view"

    def test_query_with_tags_and_difficulty(self):
        """Test query with tags and difficulty."""
        query = BenchmarkQuery(
            query="Complex mathematical proof",
            expected_domain=ProblemDomain.MATHEMATICAL,
            expected_intent=ProblemIntent.SOLVE,
            expected_complexity=9,
            recommended_methods=["mathematical_reasoning"],
            tags=["proof", "advanced", "graduate-level"],
            difficulty="hard",
        )

        assert query.tags == ["proof", "advanced", "graduate-level"]
        assert query.difficulty == "hard"

    def test_default_values(self):
        """Test default values for optional fields."""
        query = BenchmarkQuery(
            query="Simple question",
            expected_domain=ProblemDomain.GENERAL,
            expected_intent=ProblemIntent.EXPLAIN,
            expected_complexity=3,
            recommended_methods=["chain_of_thought"],
        )

        assert query.recommended_pipeline is None
        assert query.tags == []
        assert query.difficulty == "medium"


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = EvaluationMetrics()

        assert metrics.domain_accuracy == 0.0
        assert metrics.intent_accuracy == 0.0
        assert metrics.method_hit_rate == 0.0
        assert metrics.p50_latency_ms == 0.0
        assert metrics.total_queries == 0

    def test_per_class_metrics_empty_by_default(self):
        """Test that per-class metrics are empty dicts by default."""
        metrics = EvaluationMetrics()

        assert metrics.domain_precision_per_class == {}
        assert metrics.domain_recall_per_class == {}
        assert metrics.domain_f1_per_class == {}


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self):
        """Test creating an evaluation result."""
        query = BenchmarkQuery(
            query="Test",
            expected_domain=ProblemDomain.CODE,
            expected_intent=ProblemIntent.DEBUG,
            expected_complexity=5,
            recommended_methods=["code_reasoning"],
        )

        result = EvaluationResult(
            query=query,
            actual_domain=ProblemDomain.CODE,
            actual_intent=ProblemIntent.DEBUG,
            actual_complexity=5,
            recommended_method="code_reasoning",
            confidence=0.9,
            latency_ms=10.5,
            domain_correct=True,
            intent_correct=True,
            method_in_recommendations=True,
        )

        assert result.domain_correct is True
        assert result.intent_correct is True
        assert result.method_in_recommendations is True

    def test_result_with_error(self):
        """Test result with error."""
        query = BenchmarkQuery(
            query="Test",
            expected_domain=ProblemDomain.CODE,
            expected_intent=ProblemIntent.DEBUG,
            expected_complexity=5,
            recommended_methods=["code_reasoning"],
        )

        result = EvaluationResult(query=query, error="Router failed")

        assert result.error == "Router failed"


class TestRouterEvaluationHarness:
    """Tests for RouterEvaluationHarness."""

    def _create_mock_router(self):
        """Create a mock router for testing."""
        router = MagicMock()
        return router

    def _create_sample_router_result(
        self,
        domain: ProblemDomain = ProblemDomain.MATHEMATICAL,
        intent: ProblemIntent = ProblemIntent.SOLVE,
        complexity: int = 5,
        method_id: str = "chain_of_thought",
        confidence: float = 0.85,
        latency_ms: float = 5.0,
    ) -> RouterResult:
        """Create a sample RouterResult for testing."""
        analysis = ProblemAnalysis(
            primary_domain=domain,
            secondary_domains=frozenset(),
            intent=intent,
            complexity=complexity,
            capabilities=frozenset(),
            keywords=frozenset(),
            entities=frozenset(),
            confidence=confidence,
            analysis_latency_ms=2.0,
            analyzer_tier="fast",
        )

        decision = RouteDecision(
            route_type=RouteType.SINGLE_METHOD,
            method_id=method_id,
            pipeline_id=None,
            confidence=confidence,
            score=confidence,
            reasoning="Test decision",
            router_tier=RouterTier.FAST,
            latency_ms=latency_ms,
        )

        return RouterResult(
            problem_analysis=analysis,
            primary_route=decision,
            fallback_routes=(),
            resource_estimate=ResourceEstimate(),
            total_latency_ms=latency_ms,
        )

    @pytest.mark.asyncio
    async def test_evaluate_requires_router(self):
        """Test that evaluate raises if router not set."""
        harness = RouterEvaluationHarness()

        queries = [
            BenchmarkQuery(
                query="Test",
                expected_domain=ProblemDomain.GENERAL,
                expected_intent=ProblemIntent.EXPLAIN,
                expected_complexity=3,
                recommended_methods=["chain_of_thought"],
            )
        ]

        with pytest.raises(ValueError, match="Router not set"):
            await harness.evaluate(queries)

    @pytest.mark.asyncio
    async def test_evaluate_single_query(self):
        """Test evaluating a single query."""
        router = self._create_mock_router()
        router.route = AsyncMock(
            return_value=self._create_sample_router_result(
                domain=ProblemDomain.MATHEMATICAL,
                intent=ProblemIntent.SOLVE,
                complexity=5,
                method_id="chain_of_thought",
            )
        )

        harness = RouterEvaluationHarness(router)

        queries = [
            BenchmarkQuery(
                query="What is 2 + 2?",
                expected_domain=ProblemDomain.MATHEMATICAL,
                expected_intent=ProblemIntent.SOLVE,
                expected_complexity=5,
                recommended_methods=["chain_of_thought"],
            )
        ]

        metrics = await harness.evaluate(queries)

        assert metrics.total_queries == 1
        assert metrics.successful_routes == 1
        assert metrics.domain_accuracy == 1.0
        assert metrics.intent_accuracy == 1.0
        assert metrics.method_hit_rate == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self):
        """Test evaluating multiple queries."""
        router = self._create_mock_router()

        # Alternate between correct and incorrect predictions
        results = [
            self._create_sample_router_result(
                domain=ProblemDomain.MATHEMATICAL,
                method_id="chain_of_thought",
            ),
            self._create_sample_router_result(
                domain=ProblemDomain.CODE,  # Wrong domain
                method_id="code_reasoning",
            ),
        ]
        router.route = AsyncMock(side_effect=results)

        harness = RouterEvaluationHarness(router)

        queries = [
            BenchmarkQuery(
                query="Math query",
                expected_domain=ProblemDomain.MATHEMATICAL,
                expected_intent=ProblemIntent.SOLVE,
                expected_complexity=5,
                recommended_methods=["chain_of_thought"],
            ),
            BenchmarkQuery(
                query="Math query 2",
                expected_domain=ProblemDomain.MATHEMATICAL,  # Expect MATH but get CODE
                expected_intent=ProblemIntent.SOLVE,
                expected_complexity=5,
                recommended_methods=["chain_of_thought"],
            ),
        ]

        metrics = await harness.evaluate(queries)

        assert metrics.total_queries == 2
        assert metrics.domain_accuracy == 0.5  # 1 correct out of 2

    @pytest.mark.asyncio
    async def test_evaluate_handles_errors(self):
        """Test that evaluation handles router errors gracefully."""
        router = self._create_mock_router()
        router.route = AsyncMock(side_effect=ValueError("Router error"))

        harness = RouterEvaluationHarness(router)

        queries = [
            BenchmarkQuery(
                query="Test",
                expected_domain=ProblemDomain.GENERAL,
                expected_intent=ProblemIntent.EXPLAIN,
                expected_complexity=3,
                recommended_methods=["chain_of_thought"],
            )
        ]

        metrics = await harness.evaluate(queries)

        assert metrics.total_queries == 1
        assert metrics.failed_routes == 1
        assert metrics.successful_routes == 0


class TestLatencyPercentiles:
    """Tests for latency percentile calculations (Task 1.5.2.4)."""

    def test_percentile_empty(self):
        """Test percentile with empty list."""
        harness = RouterEvaluationHarness()
        assert harness._percentile([], 50) == 0.0

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        harness = RouterEvaluationHarness()
        assert harness._percentile([10.0], 50) == 10.0

    def test_percentile_p50(self):
        """Test p50 (median) calculation."""
        harness = RouterEvaluationHarness()

        # Sorted list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        values = sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        p50 = harness._percentile(values, 50)

        # p50 should be around 5.5 (between 5 and 6)
        assert 5.0 <= p50 <= 6.0

    def test_percentile_p95(self):
        """Test p95 calculation."""
        harness = RouterEvaluationHarness()

        values = sorted([float(i) for i in range(1, 101)])  # 1-100
        p95 = harness._percentile(values, 95)

        # p95 should be around 95
        assert 94.0 <= p95 <= 96.0

    def test_percentile_p99(self):
        """Test p99 calculation."""
        harness = RouterEvaluationHarness()

        values = sorted([float(i) for i in range(1, 101)])  # 1-100
        p99 = harness._percentile(values, 99)

        # p99 should be around 99
        assert 98.0 <= p99 <= 100.0


class TestPrecisionRecallF1:
    """Tests for precision/recall/F1 calculations (Task 1.5.2.3)."""

    def _create_result(
        self,
        expected_domain: ProblemDomain,
        actual_domain: ProblemDomain,
        expected_intent: ProblemIntent = ProblemIntent.SOLVE,
        actual_intent: ProblemIntent = ProblemIntent.SOLVE,
    ) -> EvaluationResult:
        """Create a test result."""
        query = BenchmarkQuery(
            query="Test",
            expected_domain=expected_domain,
            expected_intent=expected_intent,
            expected_complexity=5,
            recommended_methods=["chain_of_thought"],
        )
        return EvaluationResult(
            query=query,
            actual_domain=actual_domain,
            actual_intent=actual_intent,
            domain_correct=(expected_domain == actual_domain),
            intent_correct=(expected_intent == actual_intent),
        )

    def test_compute_precision_per_class(self):
        """Test precision calculation per class."""
        harness = RouterEvaluationHarness()

        # 2 predicted as MATH: 1 correct, 1 wrong (should be CODE)
        results = [
            self._create_result(ProblemDomain.MATHEMATICAL, ProblemDomain.MATHEMATICAL),
            self._create_result(ProblemDomain.CODE, ProblemDomain.MATHEMATICAL),
        ]

        precision = harness._compute_precision_per_class(results, "domain", ProblemDomain)

        # MATH precision: 1 TP / 2 predicted = 0.5
        assert precision[ProblemDomain.MATHEMATICAL.value] == 0.5
        # CODE precision: 0 TP / 0 predicted = 0.0
        assert precision[ProblemDomain.CODE.value] == 0.0

    def test_compute_recall_per_class(self):
        """Test recall calculation per class."""
        harness = RouterEvaluationHarness()

        # 2 actual MATH: 1 predicted correctly, 1 predicted wrong
        results = [
            self._create_result(ProblemDomain.MATHEMATICAL, ProblemDomain.MATHEMATICAL),
            self._create_result(ProblemDomain.MATHEMATICAL, ProblemDomain.CODE),
        ]

        recall = harness._compute_recall_per_class(results, "domain", ProblemDomain)

        # MATH recall: 1 TP / 2 actual = 0.5
        assert recall[ProblemDomain.MATHEMATICAL.value] == 0.5

    def test_compute_f1_per_class(self):
        """Test F1 score calculation."""
        harness = RouterEvaluationHarness()

        precision = {"mathematical": 0.8, "code": 0.6}
        recall = {"mathematical": 0.6, "code": 0.8}

        f1 = harness._compute_f1_per_class(precision, recall)

        # F1 = 2 * P * R / (P + R)
        expected_math_f1 = 2 * 0.8 * 0.6 / (0.8 + 0.6)  # 0.686
        expected_code_f1 = 2 * 0.6 * 0.8 / (0.6 + 0.8)  # 0.686

        assert abs(f1["mathematical"] - expected_math_f1) < 0.01
        assert abs(f1["code"] - expected_code_f1) < 0.01

    def test_f1_handles_zero_division(self):
        """Test F1 handles zero precision and recall."""
        harness = RouterEvaluationHarness()

        precision = {"mathematical": 0.0}
        recall = {"mathematical": 0.0}

        f1 = harness._compute_f1_per_class(precision, recall)
        assert f1["mathematical"] == 0.0


class TestBaselineComparisons:
    """Tests for baseline comparisons (Task 1.5.2.5)."""

    def _create_sample_queries(self, count: int = 10) -> list[BenchmarkQuery]:
        """Create sample benchmark queries."""
        queries = []
        for i in range(count):
            queries.append(
                BenchmarkQuery(
                    query=f"Query {i}",
                    expected_domain=ProblemDomain.MATHEMATICAL,
                    expected_intent=ProblemIntent.SOLVE,
                    expected_complexity=5,
                    recommended_methods=["chain_of_thought", "mathematical_reasoning"],
                )
            )
        return queries

    @pytest.mark.asyncio
    async def test_compare_to_baselines_returns_results(self):
        """Test that compare_to_baselines returns baseline results."""
        harness = RouterEvaluationHarness()
        queries = self._create_sample_queries(10)

        baselines = await harness.compare_to_baselines(queries)

        assert len(baselines) == 2
        assert any(b.baseline_name == "random" for b in baselines)
        assert any(b.baseline_name == "popularity" for b in baselines)

    @pytest.mark.asyncio
    async def test_random_baseline_metrics(self):
        """Test random baseline produces expected metrics."""
        harness = RouterEvaluationHarness()
        queries = self._create_sample_queries(10)

        baselines = await harness.compare_to_baselines(queries)
        random_baseline = next(b for b in baselines if b.baseline_name == "random")

        # Random baseline should have low accuracy
        assert random_baseline.metrics.method_hit_rate == 0.10  # ~10% expected
        assert random_baseline.metrics.total_queries == 10

    @pytest.mark.asyncio
    async def test_popularity_baseline_metrics(self):
        """Test popularity baseline with chain_of_thought in recommendations."""
        harness = RouterEvaluationHarness()

        # All queries have chain_of_thought in recommendations
        queries = self._create_sample_queries(10)

        baselines = await harness.compare_to_baselines(queries)
        popularity = next(b for b in baselines if b.baseline_name == "popularity")

        # Should hit 100% since all queries recommend chain_of_thought
        assert popularity.metrics.method_hit_rate == 1.0


class TestGoNoGoCriteria:
    """Tests for go/no-go ML tier decision criteria."""

    def test_ml_justified_when_accuracy_low(self):
        """Test ML tier justified when accuracy < 80%."""
        harness = RouterEvaluationHarness()

        metrics = EvaluationMetrics(
            domain_accuracy=0.70,
            intent_accuracy=0.70,
            method_hit_rate=0.80,
        )

        result = harness.compute_go_nogo_criteria(metrics)

        assert result["ml_tier_justified"] is True
        assert result["avg_accuracy"] == 0.70
        assert "Proceed to Phase 3" in result["recommendation"]

    def test_ml_not_justified_when_accuracy_high(self):
        """Test ML tier NOT justified when accuracy >= 80%."""
        harness = RouterEvaluationHarness()

        metrics = EvaluationMetrics(
            domain_accuracy=0.90,
            intent_accuracy=0.85,
            method_hit_rate=0.80,
        )

        result = harness.compute_go_nogo_criteria(metrics)

        assert result["ml_tier_justified"] is False
        assert result["avg_accuracy"] == 0.875
        assert "Skip Phase 3" in result["recommendation"]


class TestReportGeneration:
    """Tests for evaluation report generation."""

    def test_generate_report_format(self):
        """Test that report has expected sections."""
        harness = RouterEvaluationHarness()

        metrics = EvaluationMetrics(
            total_queries=100,
            successful_routes=95,
            failed_routes=5,
            domain_accuracy=0.85,
            intent_accuracy=0.80,
            method_hit_rate=0.75,
            complexity_mae=1.2,
            complexity_rmse=1.5,
            avg_latency_ms=5.0,
            p50_latency_ms=4.0,
            p95_latency_ms=12.0,
            p99_latency_ms=20.0,
        )

        report = harness.generate_report(metrics)

        # Check sections exist
        assert "REASONING ROUTER EVALUATION REPORT" in report
        assert "OVERALL METRICS" in report
        assert "CLASSIFICATION ACCURACY" in report
        assert "COMPLEXITY ESTIMATION" in report
        assert "LATENCY" in report
        assert "GO/NO-GO DECISION" in report

        # Check values appear
        assert "100" in report  # total_queries
        assert "85" in report or "0.85" in report or "85.0%" in report  # domain accuracy

    def test_generate_report_includes_baselines(self):
        """Test that report includes baselines when available."""
        harness = RouterEvaluationHarness()

        # Add baseline results
        harness._baseline_results = [
            BaselineResult(
                baseline_name="random",
                metrics=EvaluationMetrics(method_hit_rate=0.10),
                description="Random baseline",
            )
        ]

        metrics = EvaluationMetrics(total_queries=100)
        report = harness.generate_report(metrics, include_baselines=True)

        assert "BASELINE COMPARISONS" in report
        assert "random" in report


class TestTierEvaluation:
    """Tests for tier-specific evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_tier_forces_tier(self):
        """Test that evaluate_tier forces the specified tier."""
        router = MagicMock()

        analysis = ProblemAnalysis(
            primary_domain=ProblemDomain.MATHEMATICAL,
            secondary_domains=frozenset(),
            intent=ProblemIntent.SOLVE,
            complexity=5,
            capabilities=frozenset(),
            keywords=frozenset(),
            entities=frozenset(),
            confidence=0.9,
            analysis_latency_ms=2.0,
            analyzer_tier="fast",
        )

        decision = RouteDecision(
            route_type=RouteType.SINGLE_METHOD,
            method_id="chain_of_thought",
            confidence=0.9,
            score=0.9,
            reasoning="Test",
            router_tier=RouterTier.FAST,
            latency_ms=5.0,
        )

        result = RouterResult(
            problem_analysis=analysis,
            primary_route=decision,
            fallback_routes=(),
            resource_estimate=ResourceEstimate(),
            total_latency_ms=5.0,
        )

        router.route = AsyncMock(return_value=result)

        harness = RouterEvaluationHarness(router)

        queries = [
            BenchmarkQuery(
                query="Test",
                expected_domain=ProblemDomain.MATHEMATICAL,
                expected_intent=ProblemIntent.SOLVE,
                expected_complexity=5,
                recommended_methods=["chain_of_thought"],
            )
        ]

        await harness.evaluate_tier(queries, RouterTier.FAST)

        # Verify force_tier was passed
        router.route.assert_called_once()
        call_kwargs = router.route.call_args
        assert call_kwargs[1].get("force_tier") == RouterTier.FAST

    @pytest.mark.asyncio
    async def test_ablation_study_all_tiers(self):
        """Test that ablation study evaluates all tiers."""
        router = MagicMock()

        analysis = ProblemAnalysis(
            primary_domain=ProblemDomain.MATHEMATICAL,
            secondary_domains=frozenset(),
            intent=ProblemIntent.SOLVE,
            complexity=5,
            capabilities=frozenset(),
            keywords=frozenset(),
            entities=frozenset(),
            confidence=0.9,
            analysis_latency_ms=2.0,
            analyzer_tier="fast",
        )

        decision = RouteDecision(
            route_type=RouteType.SINGLE_METHOD,
            method_id="chain_of_thought",
            confidence=0.9,
            score=0.9,
            reasoning="Test",
            router_tier=RouterTier.FAST,
            latency_ms=5.0,
        )

        result = RouterResult(
            problem_analysis=analysis,
            primary_route=decision,
            fallback_routes=(),
            resource_estimate=ResourceEstimate(),
            total_latency_ms=5.0,
        )

        router.route = AsyncMock(return_value=result)

        harness = RouterEvaluationHarness(router)

        queries = [
            BenchmarkQuery(
                query="Test",
                expected_domain=ProblemDomain.MATHEMATICAL,
                expected_intent=ProblemIntent.SOLVE,
                expected_complexity=5,
                recommended_methods=["chain_of_thought"],
            )
        ]

        ablation_results = await harness.ablation_study(queries)

        # Should have results for all tiers
        assert RouterTier.FAST in ablation_results
        assert RouterTier.STANDARD in ablation_results
        assert RouterTier.COMPLEX in ablation_results
