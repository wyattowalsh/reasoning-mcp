"""Telemetry infrastructure for the Reasoning Router.

This module provides structured telemetry capture for routing decisions,
enabling measurement, analysis, and go/no-go decisions for ML tier investment.

Task 1.5.1.1: Define telemetry schema (RoutingTelemetry model)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from reasoning_mcp.router.models import RouterTier

if TYPE_CHECKING:
    from pathlib import Path

    from reasoning_mcp.router.models import ProblemAnalysis, RouteDecision

logger = logging.getLogger(__name__)


class RoutingTelemetry(BaseModel):
    """Telemetry schema for routing decisions.

    Captures all information needed to:
    - Measure routing accuracy and latency
    - Determine go/no-go criteria for ML tier investment
    - Enable continuous improvement through feedback learning

    Per spec Phase 1.5.1.1:
    - query_hash: SHA256 of problem text
    - method_selected: Method ID chosen
    - confidence: Router confidence (0-1)
    - tier_used: FAST, STANDARD, COMPLEX
    - latency_ms: Routing decision time
    - user_override: Did user select different method?
    - success_signal: Task completion indicator
    - timestamp: When the routing occurred
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    query_hash: str = Field(description="SHA256 hash of the problem text")
    session_id: str = Field(description="Unique session identifier")
    route_id: str = Field(description="Unique route identifier")

    # Routing decision
    method_selected: str = Field(description="Method ID chosen by router")
    pipeline_id: str | None = Field(default=None, description="Pipeline template ID if applicable")
    confidence: float = Field(ge=0.0, le=1.0, description="Router confidence in decision")
    tier_used: RouterTier = Field(description="Which tier made the decision")
    latency_ms: float = Field(ge=0.0, description="Time to make routing decision")

    # Problem analysis summary
    domain_detected: str = Field(description="Primary domain detected")
    intent_detected: str = Field(description="Intent detected")
    complexity_score: int = Field(ge=1, le=10, description="Complexity score 1-10")

    # Outcome signals (filled in later)
    user_override: bool | None = Field(default=None, description="Did user override method?")
    override_method: str | None = Field(default=None, description="Method user selected instead")
    success_signal: bool | None = Field(default=None, description="Task completion success")
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0, description="Quality rating")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None, description="When task completed")

    # Resource usage (filled in after execution)
    actual_tokens: int | None = Field(default=None, ge=0)
    actual_latency_ms: float | None = Field(default=None, ge=0.0)
    actual_thoughts: int | None = Field(default=None, ge=0)


def compute_query_hash(problem: str) -> str:
    """Compute SHA256 hash of problem text for telemetry.

    Args:
        problem: The problem text to hash

    Returns:
        Hex-encoded SHA256 hash string
    """
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()


def create_telemetry_from_route(
    problem: str,
    session_id: str,
    route_id: str,
    analysis: ProblemAnalysis,
    decision: RouteDecision,
) -> RoutingTelemetry:
    """Create a RoutingTelemetry record from routing results.

    Args:
        problem: Original problem text
        session_id: Unique session identifier
        route_id: Unique route identifier
        analysis: Problem analysis result
        decision: Route decision result

    Returns:
        RoutingTelemetry record ready for logging/storage
    """
    return RoutingTelemetry(
        query_hash=compute_query_hash(problem),
        session_id=session_id,
        route_id=route_id,
        method_selected=decision.method_id or decision.pipeline_id or "unknown",
        pipeline_id=decision.pipeline_id,
        confidence=decision.confidence,
        tier_used=decision.router_tier,
        latency_ms=decision.latency_ms,
        domain_detected=analysis.primary_domain.value,
        intent_detected=analysis.intent.value,
        complexity_score=analysis.complexity,
    )


class TelemetryLogger:
    """Structured logger for routing telemetry.

    Provides multiple output formats:
    - Structured Python logging
    - JSON export for analysis
    - Prometheus/StatsD metrics (via hooks)
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        json_export_path: Path | None = None,
        enable_metrics: bool = False,
    ):
        """Initialize the telemetry logger.

        Args:
            log_level: Python logging level for structured logs
            json_export_path: Optional path for JSON export file
            enable_metrics: Enable Prometheus/StatsD metrics hooks
        """
        self._log_level = log_level
        self._json_export_path = json_export_path
        self._enable_metrics = enable_metrics
        self._records: list[RoutingTelemetry] = []
        self._metrics_hooks: list[Any] = []

    def log_routing_decision(self, telemetry: RoutingTelemetry) -> None:
        """Log a routing decision with structured data.

        Args:
            telemetry: The telemetry record to log
        """
        # Store record
        self._records.append(telemetry)

        # Structured log
        logger.log(
            self._log_level,
            "Routing decision: method=%s tier=%s confidence=%.2f latency=%.1fms domain=%s",
            telemetry.method_selected,
            telemetry.tier_used.value,
            telemetry.confidence,
            telemetry.latency_ms,
            telemetry.domain_detected,
            extra={
                "telemetry": telemetry.model_dump(mode="json"),
                "event_type": "routing_decision",
            },
        )

        # Metrics hooks
        if self._enable_metrics:
            self._emit_metrics(telemetry)

    def log_outcome(
        self,
        route_id: str,
        success: bool,
        quality_score: float | None = None,
        actual_tokens: int | None = None,
        actual_latency_ms: float | None = None,
    ) -> None:
        """Log the outcome of a routing decision.

        Args:
            route_id: The route ID to update
            success: Whether the task succeeded
            quality_score: Optional quality rating 0-1
            actual_tokens: Actual tokens used
            actual_latency_ms: Actual execution latency
        """
        # Find and update the record
        for i, record in enumerate(self._records):
            if record.route_id == route_id:
                # Create updated record (immutable, so replace)
                updated = RoutingTelemetry(
                    **{
                        **record.model_dump(),
                        "success_signal": success,
                        "quality_score": quality_score,
                        "actual_tokens": actual_tokens,
                        "actual_latency_ms": actual_latency_ms,
                        "completed_at": datetime.now(),
                    }
                )
                self._records[i] = updated

                logger.log(
                    self._log_level,
                    "Route outcome: route_id=%s success=%s quality=%.2f",
                    route_id,
                    success,
                    quality_score or 0.0,
                    extra={
                        "telemetry": updated.model_dump(mode="json"),
                        "event_type": "route_outcome",
                    },
                )
                break

    def log_user_override(self, route_id: str, override_method: str) -> None:
        """Log when a user overrides the routing decision.

        Args:
            route_id: The route ID that was overridden
            override_method: The method the user selected instead
        """
        for i, record in enumerate(self._records):
            if record.route_id == route_id:
                updated = RoutingTelemetry(
                    **{
                        **record.model_dump(),
                        "user_override": True,
                        "override_method": override_method,
                    }
                )
                self._records[i] = updated

                logger.log(
                    self._log_level,
                    "User override: route_id=%s original=%s override=%s",
                    route_id,
                    record.method_selected,
                    override_method,
                    extra={
                        "telemetry": updated.model_dump(mode="json"),
                        "event_type": "user_override",
                    },
                )
                break

    def export_json(self, path: Path | None = None) -> str:
        """Export all telemetry records as JSON.

        Args:
            path: Optional path to write JSON file (uses configured path if None)

        Returns:
            JSON string of all records
        """
        export_path = path or self._json_export_path
        data = [r.model_dump(mode="json") for r in self._records]
        json_str = json.dumps(data, indent=2, default=str)

        if export_path:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(json_str)
            logger.info("Exported %d telemetry records to %s", len(self._records), export_path)

        return json_str

    def get_records(
        self,
        since: datetime | None = None,
        tier: RouterTier | None = None,
        with_outcomes: bool = False,
    ) -> list[RoutingTelemetry]:
        """Get telemetry records with optional filtering.

        Args:
            since: Only return records after this timestamp
            tier: Filter by router tier
            with_outcomes: Only return records that have outcome data

        Returns:
            Filtered list of telemetry records
        """
        records = self._records

        if since:
            records = [r for r in records if r.timestamp >= since]

        if tier:
            records = [r for r in records if r.tier_used == tier]

        if with_outcomes:
            records = [r for r in records if r.success_signal is not None]

        return records

    def compute_go_nogo_metrics(self) -> dict[str, Any]:
        """Compute metrics for go/no-go ML tier decision.

        Returns:
            Dictionary with:
            - misrouting_rate: Percentage of routes that were overridden
            - avg_confidence: Average confidence of routing decisions
            - user_override_rate: Rate of user overrides
            - sample_count: Total number of samples
            - ml_tier_justified: Boolean indicating if ML investment is justified
        """
        records_with_outcomes = [r for r in self._records if r.success_signal is not None]

        if len(records_with_outcomes) < 100:
            return {
                "misrouting_rate": None,
                "avg_confidence": None,
                "user_override_rate": None,
                "sample_count": len(records_with_outcomes),
                "ml_tier_justified": None,
                "reason": f"Insufficient samples ({len(records_with_outcomes)} < 100)",
            }

        # Calculate metrics
        overrides = [r for r in records_with_outcomes if r.user_override]
        failures = [r for r in records_with_outcomes if not r.success_signal]

        misrouting_rate = (len(overrides) + len(failures)) / len(records_with_outcomes)
        total = len(records_with_outcomes)
        avg_confidence = sum(r.confidence for r in records_with_outcomes) / total
        user_override_rate = len(overrides) / total

        # Go/no-go criteria from spec:
        # ML_TIER_JUSTIFIED = (
        #     misrouting_rate > 0.20 or
        #     avg_confidence < 0.75 or
        #     user_override_rate > 0.15
        # )
        ml_tier_justified = (
            misrouting_rate > 0.20 or avg_confidence < 0.75 or user_override_rate > 0.15
        )

        return {
            "misrouting_rate": misrouting_rate,
            "avg_confidence": avg_confidence,
            "user_override_rate": user_override_rate,
            "sample_count": len(records_with_outcomes),
            "ml_tier_justified": ml_tier_justified,
            "reason": self._get_justification_reason(
                misrouting_rate, avg_confidence, user_override_rate
            ),
        }

    def _get_justification_reason(
        self, misrouting_rate: float, avg_confidence: float, user_override_rate: float
    ) -> str:
        """Generate human-readable justification for go/no-go decision."""
        reasons = []

        if misrouting_rate > 0.20:
            reasons.append(f"misrouting rate {misrouting_rate:.1%} > 20%")
        if avg_confidence < 0.75:
            reasons.append(f"avg confidence {avg_confidence:.2f} < 0.75")
        if user_override_rate > 0.15:
            reasons.append(f"override rate {user_override_rate:.1%} > 15%")

        if reasons:
            return "ML tier justified: " + ", ".join(reasons)
        return "ML tier NOT justified: all metrics within acceptable bounds"

    def _emit_metrics(self, telemetry: RoutingTelemetry) -> None:
        """Emit metrics to configured hooks (Prometheus/StatsD).

        Override in subclass or configure hooks for production use.
        """
        for hook in self._metrics_hooks:
            try:
                hook(telemetry)
            except Exception as e:
                logger.warning("Metrics hook failed: %s", e)

    def register_metrics_hook(self, hook: Any) -> None:
        """Register a metrics emission hook.

        Args:
            hook: Callable that accepts RoutingTelemetry
        """
        self._metrics_hooks.append(hook)

    def clear(self) -> None:
        """Clear all stored telemetry records."""
        self._records.clear()


# Global telemetry logger instance
_telemetry_logger: TelemetryLogger | None = None


def get_telemetry_logger() -> TelemetryLogger:
    """Get the global telemetry logger instance."""
    global _telemetry_logger
    if _telemetry_logger is None:
        _telemetry_logger = TelemetryLogger()
    return _telemetry_logger


def configure_telemetry(
    log_level: int = logging.INFO,
    json_export_path: Path | None = None,
    enable_metrics: bool = False,
) -> TelemetryLogger:
    """Configure the global telemetry logger.

    Args:
        log_level: Python logging level
        json_export_path: Path for JSON export
        enable_metrics: Enable Prometheus/StatsD hooks

    Returns:
        Configured TelemetryLogger instance
    """
    global _telemetry_logger
    _telemetry_logger = TelemetryLogger(
        log_level=log_level,
        json_export_path=json_export_path,
        enable_metrics=enable_metrics,
    )
    return _telemetry_logger
