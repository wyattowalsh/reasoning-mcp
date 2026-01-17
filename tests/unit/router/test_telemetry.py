"""Unit tests for the telemetry infrastructure.

Tests for:
- Task 1.5.1.1: RoutingTelemetry schema
- Task 1.5.1.2: Structured logging
- Task 1.5.1.3: JSON export
- Task 1.5.1.4: Prometheus/StatsD metrics hooks
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from reasoning_mcp.router.models import RouterTier
from reasoning_mcp.router.telemetry import (
    RoutingTelemetry,
    TelemetryLogger,
    compute_query_hash,
    configure_telemetry,
    get_telemetry_logger,
)


class TestRoutingTelemetry:
    """Tests for the RoutingTelemetry schema (Task 1.5.1.1)."""

    def test_create_basic_telemetry(self):
        """Test creating a basic telemetry record."""
        telemetry = RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

        assert telemetry.query_hash == "abc123"
        assert telemetry.method_selected == "chain_of_thought"
        assert telemetry.confidence == 0.85
        assert telemetry.tier_used == RouterTier.FAST
        assert telemetry.latency_ms == 5.2

    def test_telemetry_optional_fields(self):
        """Test that optional fields default correctly."""
        telemetry = RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

        assert telemetry.user_override is None
        assert telemetry.success_signal is None
        assert telemetry.quality_score is None
        assert telemetry.actual_tokens is None
        assert telemetry.pipeline_id is None

    def test_telemetry_with_outcome_fields(self):
        """Test telemetry with outcome fields populated."""
        telemetry = RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
            user_override=True,
            override_method="tree_of_thoughts",
            success_signal=True,
            quality_score=0.9,
            actual_tokens=1500,
            actual_latency_ms=250.0,
        )

        assert telemetry.user_override is True
        assert telemetry.override_method == "tree_of_thoughts"
        assert telemetry.success_signal is True
        assert telemetry.quality_score == 0.9
        assert telemetry.actual_tokens == 1500

    def test_telemetry_immutable(self):
        """Test that telemetry records are immutable (frozen)."""
        telemetry = RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

        with pytest.raises(ValidationError):
            telemetry.confidence = 0.5

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            RoutingTelemetry(
                query_hash="abc123",
                session_id="session-001",
                route_id="route-001",
                method_selected="chain_of_thought",
                confidence=1.5,  # Invalid: > 1
                tier_used=RouterTier.FAST,
                latency_ms=5.2,
                domain_detected="mathematical",
                intent_detected="solve",
                complexity_score=5,
            )

    def test_complexity_bounds(self):
        """Test that complexity must be between 1 and 10."""
        with pytest.raises(ValidationError):
            RoutingTelemetry(
                query_hash="abc123",
                session_id="session-001",
                route_id="route-001",
                method_selected="chain_of_thought",
                confidence=0.85,
                tier_used=RouterTier.FAST,
                latency_ms=5.2,
                domain_detected="mathematical",
                intent_detected="solve",
                complexity_score=15,  # Invalid: > 10
            )


class TestComputeQueryHash:
    """Tests for the query hash function."""

    def test_hash_deterministic(self):
        """Test that the same query produces the same hash."""
        problem = "What is 2 + 2?"
        hash1 = compute_query_hash(problem)
        hash2 = compute_query_hash(problem)
        assert hash1 == hash2

    def test_hash_different_queries(self):
        """Test that different queries produce different hashes."""
        hash1 = compute_query_hash("What is 2 + 2?")
        hash2 = compute_query_hash("What is 3 + 3?")
        assert hash1 != hash2

    def test_hash_format(self):
        """Test that hash is a valid hex string."""
        hash_value = compute_query_hash("test")
        assert len(hash_value) == 64  # SHA256 = 64 hex chars
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestTelemetryLoggerStructuredLogging:
    """Tests for structured logging (Task 1.5.1.2)."""

    def _create_sample_telemetry(self, route_id: str = "route-001") -> RoutingTelemetry:
        """Create a sample telemetry record for testing."""
        return RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id=route_id,
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

    def test_log_routing_decision_stores_record(self):
        """Test that logging a decision stores the record."""
        logger_instance = TelemetryLogger()
        telemetry = self._create_sample_telemetry()

        logger_instance.log_routing_decision(telemetry)

        records = logger_instance.get_records()
        assert len(records) == 1
        assert records[0].route_id == "route-001"

    def test_log_outcome_updates_record(self):
        """Test that logging an outcome updates the record."""
        logger_instance = TelemetryLogger()
        telemetry = self._create_sample_telemetry()

        logger_instance.log_routing_decision(telemetry)
        logger_instance.log_outcome(
            route_id="route-001",
            success=True,
            quality_score=0.9,
            actual_tokens=1500,
            actual_latency_ms=250.0,
        )

        records = logger_instance.get_records()
        assert len(records) == 1
        assert records[0].success_signal is True
        assert records[0].quality_score == 0.9
        assert records[0].actual_tokens == 1500

    def test_log_user_override(self):
        """Test logging a user override."""
        logger_instance = TelemetryLogger()
        telemetry = self._create_sample_telemetry()

        logger_instance.log_routing_decision(telemetry)
        logger_instance.log_user_override("route-001", "tree_of_thoughts")

        records = logger_instance.get_records()
        assert records[0].user_override is True
        assert records[0].override_method == "tree_of_thoughts"

    def test_multiple_records(self):
        """Test storing multiple records."""
        logger_instance = TelemetryLogger()

        for i in range(5):
            telemetry = self._create_sample_telemetry(f"route-{i:03d}")
            logger_instance.log_routing_decision(telemetry)

        records = logger_instance.get_records()
        assert len(records) == 5

    def test_filter_by_tier(self):
        """Test filtering records by tier."""
        logger_instance = TelemetryLogger()

        # Add FAST tier record
        fast_telemetry = RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )
        logger_instance.log_routing_decision(fast_telemetry)

        # Add STANDARD tier record
        standard_telemetry = RoutingTelemetry(
            query_hash="def456",
            session_id="session-002",
            route_id="route-002",
            method_selected="tree_of_thoughts",
            confidence=0.9,
            tier_used=RouterTier.STANDARD,
            latency_ms=150.0,
            domain_detected="code",
            intent_detected="debug",
            complexity_score=7,
        )
        logger_instance.log_routing_decision(standard_telemetry)

        fast_records = logger_instance.get_records(tier=RouterTier.FAST)
        assert len(fast_records) == 1
        assert fast_records[0].tier_used == RouterTier.FAST

    def test_filter_with_outcomes_only(self):
        """Test filtering for records with outcomes."""
        logger_instance = TelemetryLogger()

        # Add record without outcome
        telemetry1 = self._create_sample_telemetry("route-001")
        logger_instance.log_routing_decision(telemetry1)

        # Add record with outcome
        telemetry2 = self._create_sample_telemetry("route-002")
        logger_instance.log_routing_decision(telemetry2)
        logger_instance.log_outcome("route-002", success=True)

        with_outcomes = logger_instance.get_records(with_outcomes=True)
        assert len(with_outcomes) == 1
        assert with_outcomes[0].route_id == "route-002"


class TestTelemetryLoggerJSONExport:
    """Tests for JSON export (Task 1.5.1.3)."""

    def _create_sample_telemetry(self) -> RoutingTelemetry:
        return RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

    def test_export_json_returns_string(self):
        """Test that export_json returns a valid JSON string."""
        logger_instance = TelemetryLogger()
        telemetry = self._create_sample_telemetry()
        logger_instance.log_routing_decision(telemetry)

        json_str = logger_instance.export_json()

        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["route_id"] == "route-001"

    def test_export_json_to_file(self):
        """Test exporting JSON to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "telemetry.json"
            logger_instance = TelemetryLogger(json_export_path=export_path)

            telemetry = self._create_sample_telemetry()
            logger_instance.log_routing_decision(telemetry)
            logger_instance.export_json()

            assert export_path.exists()
            data = json.loads(export_path.read_text())
            assert len(data) == 1

    def test_export_json_override_path(self):
        """Test exporting to a different path than configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "override.json"
            logger_instance = TelemetryLogger()

            telemetry = self._create_sample_telemetry()
            logger_instance.log_routing_decision(telemetry)
            logger_instance.export_json(path=override_path)

            assert override_path.exists()

    def test_export_json_creates_directories(self):
        """Test that export creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "telemetry.json"
            logger_instance = TelemetryLogger()

            telemetry = self._create_sample_telemetry()
            logger_instance.log_routing_decision(telemetry)
            logger_instance.export_json(path=nested_path)

            assert nested_path.exists()


class TestTelemetryLoggerMetricsHooks:
    """Tests for metrics hooks (Task 1.5.1.4)."""

    def _create_sample_telemetry(self) -> RoutingTelemetry:
        return RoutingTelemetry(
            query_hash="abc123",
            session_id="session-001",
            route_id="route-001",
            method_selected="chain_of_thought",
            confidence=0.85,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
        )

    def test_register_metrics_hook(self):
        """Test registering a metrics hook."""
        logger_instance = TelemetryLogger(enable_metrics=True)
        mock_hook = MagicMock()

        logger_instance.register_metrics_hook(mock_hook)

        telemetry = self._create_sample_telemetry()
        logger_instance.log_routing_decision(telemetry)

        mock_hook.assert_called_once_with(telemetry)

    def test_multiple_metrics_hooks(self):
        """Test multiple metrics hooks are all called."""
        logger_instance = TelemetryLogger(enable_metrics=True)
        hooks = [MagicMock() for _ in range(3)]

        for hook in hooks:
            logger_instance.register_metrics_hook(hook)

        telemetry = self._create_sample_telemetry()
        logger_instance.log_routing_decision(telemetry)

        for hook in hooks:
            hook.assert_called_once_with(telemetry)

    def test_metrics_hooks_disabled_by_default(self):
        """Test that metrics hooks are not called when disabled."""
        logger_instance = TelemetryLogger(enable_metrics=False)
        mock_hook = MagicMock()

        logger_instance.register_metrics_hook(mock_hook)

        telemetry = self._create_sample_telemetry()
        logger_instance.log_routing_decision(telemetry)

        mock_hook.assert_not_called()

    def test_metrics_hook_error_handling(self, caplog):
        """Test that a failing hook doesn't break logging."""
        logger_instance = TelemetryLogger(enable_metrics=True)

        def failing_hook(telemetry):
            raise ValueError("Hook failed!")

        logger_instance.register_metrics_hook(failing_hook)

        telemetry = self._create_sample_telemetry()

        # Should not raise
        with caplog.at_level(logging.WARNING):
            logger_instance.log_routing_decision(telemetry)

        # Record should still be stored
        assert len(logger_instance.get_records()) == 1


class TestGoNoGoMetrics:
    """Tests for go/no-go ML tier decision metrics."""

    def _create_telemetry_with_outcome(
        self,
        route_id: str,
        confidence: float,
        success: bool,
        user_override: bool = False,
    ) -> RoutingTelemetry:
        return RoutingTelemetry(
            query_hash=f"hash-{route_id}",
            session_id="session-001",
            route_id=route_id,
            method_selected="chain_of_thought",
            confidence=confidence,
            tier_used=RouterTier.FAST,
            latency_ms=5.2,
            domain_detected="mathematical",
            intent_detected="solve",
            complexity_score=5,
            success_signal=success,
            user_override=user_override,
        )

    def test_insufficient_samples(self):
        """Test that go/no-go requires 100+ samples."""
        logger_instance = TelemetryLogger()

        # Add only 50 samples
        for i in range(50):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.8, success=True
            )
            logger_instance._records.append(telemetry)

        result = logger_instance.compute_go_nogo_metrics()
        assert result["ml_tier_justified"] is None
        assert "Insufficient samples" in result["reason"]

    def test_ml_justified_by_misrouting(self):
        """Test ML tier justified when misrouting rate > 20%."""
        logger_instance = TelemetryLogger()

        # 75 success, 25 failures (25% failure = misrouting)
        for i in range(75):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.9, success=True
            )
            logger_instance._records.append(telemetry)

        for i in range(75, 100):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.9, success=False
            )
            logger_instance._records.append(telemetry)

        result = logger_instance.compute_go_nogo_metrics()
        assert result["ml_tier_justified"] is True
        assert result["misrouting_rate"] == 0.25

    def test_ml_justified_by_low_confidence(self):
        """Test ML tier justified when avg confidence < 0.75."""
        logger_instance = TelemetryLogger()

        # All success but low confidence
        for i in range(100):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.6, success=True
            )
            logger_instance._records.append(telemetry)

        result = logger_instance.compute_go_nogo_metrics()
        assert result["ml_tier_justified"] is True
        assert result["avg_confidence"] == 0.6

    def test_ml_justified_by_override_rate(self):
        """Test ML tier justified when override rate > 15%."""
        logger_instance = TelemetryLogger()

        # 80 normal, 20 with user overrides
        for i in range(80):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.9, success=True
            )
            logger_instance._records.append(telemetry)

        for i in range(80, 100):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.9, success=True, user_override=True
            )
            logger_instance._records.append(telemetry)

        result = logger_instance.compute_go_nogo_metrics()
        assert result["ml_tier_justified"] is True
        assert result["user_override_rate"] == 0.20

    def test_ml_not_justified(self):
        """Test ML tier NOT justified when all metrics are good."""
        logger_instance = TelemetryLogger()

        # All success, high confidence, no overrides
        for i in range(100):
            telemetry = self._create_telemetry_with_outcome(
                f"route-{i}", confidence=0.9, success=True
            )
            logger_instance._records.append(telemetry)

        result = logger_instance.compute_go_nogo_metrics()
        assert result["ml_tier_justified"] is False
        assert "NOT justified" in result["reason"]


class TestGlobalTelemetryLogger:
    """Tests for global telemetry logger functions."""

    def test_get_telemetry_logger_singleton(self):
        """Test that get_telemetry_logger returns singleton."""
        # Reset global state
        import reasoning_mcp.router.telemetry as telemetry_module

        telemetry_module._telemetry_logger = None

        logger1 = get_telemetry_logger()
        logger2 = get_telemetry_logger()
        assert logger1 is logger2

    def test_configure_telemetry(self):
        """Test configuring the global telemetry logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "telemetry.json"

            logger_instance = configure_telemetry(
                log_level=logging.DEBUG,
                json_export_path=export_path,
                enable_metrics=True,
            )

            assert logger_instance._log_level == logging.DEBUG
            assert logger_instance._json_export_path == export_path
            assert logger_instance._enable_metrics is True


class TestTelemetryLoggerClear:
    """Tests for clearing telemetry records."""

    def test_clear_records(self):
        """Test clearing all records."""
        logger_instance = TelemetryLogger()

        # Add some records
        for i in range(10):
            telemetry = RoutingTelemetry(
                query_hash=f"hash-{i}",
                session_id="session-001",
                route_id=f"route-{i}",
                method_selected="chain_of_thought",
                confidence=0.85,
                tier_used=RouterTier.FAST,
                latency_ms=5.2,
                domain_detected="mathematical",
                intent_detected="solve",
                complexity_score=5,
            )
            logger_instance.log_routing_decision(telemetry)

        assert len(logger_instance.get_records()) == 10

        logger_instance.clear()
        assert len(logger_instance.get_records()) == 0
