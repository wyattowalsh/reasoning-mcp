"""Configuration management for reasoning-mcp server.

This module provides the Settings class for managing server configuration
with support for environment variables and .env files.

Telemetry Configuration:
    Telemetry is opt-in and disabled by default. To enable telemetry in production:

    1. Set REASONING_MCP_ENABLE_TELEMETRY=true
    2. Set REASONING_MCP_TELEMETRY_OTLP_ENDPOINT to your collector URL
       Example: REASONING_MCP_TELEMETRY_OTLP_ENDPOINT=http://otel-collector:4317

    If telemetry is enabled but no endpoint is configured (empty string),
    telemetry will be automatically disabled with a warning.

    For local development with a collector running on localhost:
        REASONING_MCP_ENABLE_TELEMETRY=true
        REASONING_MCP_TELEMETRY_OTLP_ENDPOINT=http://localhost:4317

    For Kubernetes/container environments:
        REASONING_MCP_ENABLE_TELEMETRY=true
        REASONING_MCP_TELEMETRY_OTLP_ENDPOINT=http://otel-collector.observability.svc:4317
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reasoning_mcp.models.debug import TraceLevel
from reasoning_mcp.models.ensemble import VotingStrategy
from reasoning_mcp.streaming.backpressure import BackpressureStrategy

# Module-level logger for configuration warnings
_config_logger = logging.getLogger(__name__)


class StreamingConfig(BaseModel):
    """Configuration for streaming functionality."""

    enabled: bool = Field(default=True, description="Enable streaming events")
    default_buffer_size: int = Field(
        default=1000, ge=1, description="Default buffer size for streaming events"
    )
    default_backpressure: BackpressureStrategy = Field(
        default=BackpressureStrategy.BLOCK,
        description="Default backpressure strategy: block, drop_oldest, drop_newest, error",
    )
    sse_enabled: bool = Field(default=True, description="Enable SSE transport")
    websocket_enabled: bool = Field(default=True, description="Enable WebSocket transport")


class VerificationConfig(BaseModel):
    """Configuration for verification features."""

    enabled: bool = Field(default=True, description="Whether verification is enabled")
    extractor_type: str = Field(
        default="hybrid",
        description="Type of claim extractor: 'llm', 'rule_based', or 'hybrid'",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for claims",
    )
    check_hallucinations: bool = Field(
        default=True, description="Whether to check for hallucinations"
    )


class EnsembleConfigSettings(BaseModel):
    """Configuration settings for ensemble reasoning.

    Defines default parameters for ensemble execution including voting
    strategy, default member methods, and execution timeouts.
    """

    default_strategy: VotingStrategy = Field(
        default=VotingStrategy.MAJORITY,
        description="Default voting strategy for ensemble aggregation",
    )
    default_members: list[str] = Field(
        default_factory=lambda: ["chain_of_thought", "tree_of_thoughts", "self_reflection"],
        description="Default reasoning methods to include in ensemble",
    )
    timeout_ms: int = Field(
        default=30000,
        gt=0,
        description="Default timeout in milliseconds for ensemble execution",
    )


class DebugConfig(BaseModel):
    """Configuration for debugging and tracing."""

    enabled: bool = Field(default=False, description="Enable debug tracing")
    default_level: TraceLevel = Field(
        default=TraceLevel.STANDARD,
        description="Default trace verbosity level",
    )
    storage_type: Literal["memory", "file", "sqlite"] = Field(
        default="memory",
        description="Trace storage backend type",
    )
    storage_path: Path | None = Field(
        default=None,
        description="Path for file/sqlite storage",
    )
    auto_export: bool = Field(
        default=False,
        description="Automatically export traces after completion",
    )
    export_format: str = Field(
        default="json",
        description="Default export format (json, html, mermaid, otlp)",
    )


class APIKeyInfo(BaseModel):
    """Information about an API key.

    Attributes:
        key_hash: SHA-256 hash of the API key for secure storage
        user_id: User ID associated with this key
        description: Human-readable description of the key
        permissions: List of permissions (e.g., ["read", "write", "admin"])
        created_at: When the key was created
        last_used_at: When the key was last used (None if never used)
        revoked: Whether the key has been revoked
        revoked_at: When the key was revoked (None if not revoked)
        expires_at: When the key expires (None for no expiration)
    """

    key_hash: str = Field(description="SHA-256 hash of the API key")
    user_id: str = Field(description="User ID associated with this key")
    description: str = Field(default="", description="Human-readable description of the key")
    permissions: list[str] = Field(
        default_factory=lambda: ["read"], description="List of permissions granted to this key"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the key was created"
    )
    last_used_at: datetime | None = Field(default=None, description="When the key was last used")
    revoked: bool = Field(default=False, description="Whether the key has been revoked")
    revoked_at: datetime | None = Field(default=None, description="When the key was revoked")
    expires_at: datetime | None = Field(
        default=None, description="When the key expires (None for no expiration)"
    )

    def is_valid(self) -> bool:
        """Check if the key is currently valid.

        Returns:
            True if the key is not revoked and not expired
        """
        if self.revoked:
            return False
        return not (self.expires_at and datetime.now() > self.expires_at)


class Settings(BaseSettings):
    """Server configuration settings.

    Settings can be configured via environment variables with the
    REASONING_MCP_ prefix, or via a .env file.

    Example:
        REASONING_MCP_LOG_LEVEL=DEBUG
        REASONING_MCP_SESSION_TIMEOUT=7200
    """

    model_config = SettingsConfigDict(
        env_prefix="REASONING_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    server_name: str = Field(default="reasoning-mcp", description="Name of the MCP server")
    server_version: str = Field(default="0.1.0", description="Version of the server")

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )

    # Session settings
    session_timeout: int = Field(
        default=3600, ge=60, description="Session timeout in seconds (minimum 60)"
    )
    max_sessions: int = Field(
        default=100, ge=1, description="Maximum number of concurrent sessions"
    )
    session_cleanup_interval: int = Field(
        default=300, ge=60, description="Interval for session cleanup in seconds"
    )

    # Reasoning settings
    default_method: str = Field(
        default="chain_of_thought", description="Default reasoning method when not specified"
    )
    max_thoughts_per_session: int = Field(
        default=1000, ge=10, description="Maximum thoughts allowed per session"
    )
    max_branches_per_session: int = Field(
        default=50, ge=1, description="Maximum branches allowed per session"
    )

    # Pipeline settings
    max_pipeline_depth: int = Field(
        default=10, ge=1, le=100, description="Maximum nesting depth for pipelines"
    )
    pipeline_timeout: int = Field(
        default=300, ge=10, description="Pipeline execution timeout in seconds"
    )
    max_parallel_stages: int = Field(
        default=8, ge=1, le=32, description="Maximum parallel stages in a pipeline"
    )

    # Plugin settings
    plugins_dir: Path = Field(default=Path("plugins"), description="Directory for plugin discovery")
    enable_plugins: bool = Field(default=True, description="Whether to enable plugin loading")

    # Development settings
    enable_tracing: bool = Field(default=False, description="Enable detailed tracing")

    # Debug settings
    debug: DebugConfig = Field(
        default_factory=DebugConfig,
        description="Debug and tracing configuration",
    )

    # Streaming settings
    streaming: StreamingConfig = Field(
        default_factory=StreamingConfig,
        description="Streaming configuration for real-time events",
    )

    # Verification settings
    verification: VerificationConfig = Field(
        default_factory=VerificationConfig,
        description="Verification configuration for claim extraction and hallucination checking",
    )

    # Ensemble settings
    ensemble: EnsembleConfigSettings = Field(
        default_factory=EnsembleConfigSettings,
        description="Ensemble reasoning configuration for combining multiple methods",
    )

    # FastMCP v2 Middleware settings
    enable_middleware: bool = Field(
        default=True, description="Enable MCP middleware for logging and metrics"
    )
    middleware_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="DEBUG", description="Log level for middleware request/response logging"
    )
    enable_middleware_metrics: bool = Field(
        default=True, description="Enable metrics collection in middleware"
    )

    # FastMCP v2 Sampling settings
    enable_sampling: bool = Field(
        default=True, description="Enable LLM sampling support via ctx.sample()"
    )
    sampling_provider: Literal["openai", "anthropic", "auto"] = Field(
        default="auto", description="Sampling provider: openai, anthropic, or auto-detect"
    )
    sampling_model: str = Field(
        default="gpt-4o-mini", description="Default model for sampling operations"
    )
    sampling_max_tokens: int = Field(
        default=4096, ge=1, le=128000, description="Maximum tokens for sampling responses"
    )
    sampling_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for sampling responses"
    )

    # FastMCP v2 Elicitation settings
    enable_elicitation: bool = Field(
        default=True, description="Enable user elicitation support via ctx.elicit()"
    )
    elicitation_timeout: int = Field(
        default=300, ge=10, description="Timeout in seconds for elicitation responses"
    )

    # FastMCP v2 Background Tasks settings
    enable_background_tasks: bool = Field(
        default=True, description="Enable background task support for long-running operations"
    )
    task_backend: Literal["memory", "redis"] = Field(
        default="memory", description="Backend for background tasks: memory or redis"
    )
    redis_url: SecretStr | None = Field(
        default=None,
        description="Redis URL for background task persistence (optional). "
        "Uses SecretStr to prevent accidental logging of credentials.",
    )

    # FastMCP v2 Context State settings
    enable_context_state: bool = Field(
        default=True, description="Enable per-request context state management"
    )

    # FastMCP v2 Response Cache settings
    enable_cache: bool = Field(
        default=True, description="Enable response caching for expensive operations"
    )
    cache_default_ttl_seconds: int = Field(
        default=300, ge=1, le=86400, description="Default cache TTL in seconds (5 minutes)"
    )
    cache_max_entries: int = Field(
        default=1000, ge=10, le=100000, description="Maximum number of cache entries"
    )
    cache_routing: bool = Field(
        default=True, description="Enable caching for router.route() results"
    )
    cache_routing_ttl_seconds: int = Field(
        default=60, ge=1, le=3600, description="TTL for cached routing results (1 minute)"
    )
    cache_recommendations: bool = Field(
        default=True, description="Enable caching for method recommendations"
    )
    cache_recommendations_ttl_seconds: int = Field(
        default=300, ge=1, le=3600, description="TTL for cached recommendations (5 minutes)"
    )
    cache_analysis: bool = Field(
        default=True, description="Enable caching for problem analysis operations"
    )
    cache_analysis_ttl_seconds: int = Field(
        default=300, ge=1, le=3600, description="TTL for cached problem analysis (5 minutes)"
    )

    # Router settings
    enable_router: bool = Field(
        default=True, description="Enable intelligent routing for method selection"
    )
    router_default_tier: Literal["fast", "standard", "complex"] = Field(
        default="standard",
        description="Default routing tier: fast (<5ms), standard (~20ms), complex (~200ms)",
    )
    enable_ml_routing: bool = Field(
        default=True, description="Enable Tier 2 ML-based routing (classifiers, embeddings)"
    )
    enable_llm_routing: bool = Field(
        default=True, description="Enable Tier 3 LLM-based routing (complex analysis)"
    )

    # Router tier thresholds
    router_fast_threshold_ms: int = Field(
        default=5, ge=1, le=50, description="Maximum latency for fast tier in milliseconds"
    )
    router_standard_threshold_ms: int = Field(
        default=50, ge=10, le=500, description="Maximum latency for standard tier in milliseconds"
    )
    router_complex_threshold_ms: int = Field(
        default=500, ge=100, le=5000, description="Maximum latency for complex tier in milliseconds"
    )

    # Router resource defaults
    router_default_max_latency_ms: int = Field(
        default=30000, ge=100, description="Default max latency for routing budget"
    )
    router_default_max_tokens: int = Field(
        default=50000, ge=1000, description="Default max tokens for routing budget"
    )
    router_default_max_thoughts: int = Field(
        default=50, ge=1, description="Default max thoughts for routing budget"
    )

    # Router learning settings
    enable_route_learning: bool = Field(
        default=True, description="Enable learning from routing outcomes"
    )
    route_learning_mode: Literal["off", "observe", "incremental", "batch", "manual"] = Field(
        default="observe", description="Learning mode: off, observe, incremental, batch, or manual"
    )
    route_feedback_window: int = Field(
        default=1000, ge=100, description="Number of recent outcomes to consider for learning"
    )

    # Router storage settings
    router_storage_backend: Literal["memory", "sqlite", "postgres", "s3"] = Field(
        default="sqlite", description="Storage backend for router learning data"
    )
    router_sqlite_path: Path = Field(
        default=Path("~/.reasoning-mcp/router.db"),
        description="SQLite database path for router data",
    )
    router_postgres_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL for router data (if using postgres backend). "
        "Uses SecretStr to prevent accidental logging of credentials.",
    )
    router_s3_bucket: str | None = Field(
        default=None, description="S3 bucket for router data (if using s3 backend)"
    )
    router_s3_prefix: str = Field(
        default="reasoning-mcp/router/", description="S3 prefix for router data files"
    )
    router_max_outcomes_retained: int = Field(
        default=10000, ge=100, description="Maximum number of outcomes to retain"
    )
    router_outcome_retention_days: int = Field(
        default=90, ge=1, description="Number of days to retain outcome data"
    )

    # JWT Authentication settings (MCPaaS feature)
    jwt_enabled: bool = Field(
        default=False, description="Enable JWT Bearer token authentication (opt-in for MCPaaS)"
    )
    jwt_secret_key: SecretStr | None = Field(
        default=None,
        description=(
            "Secret key for JWT token signing and verification. "
            "Required if jwt_enabled=True. Set via REASONING_MCP_JWT_SECRET_KEY environment variable."
        ),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algorithm for JWT token signing (HS256, HS384, HS512, RS256, etc.)",
    )
    jwt_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=43200,  # Max 30 days
        description="JWT token expiration time in minutes",
    )
    jwt_issuer: str | None = Field(
        default=None, description="JWT issuer claim for token validation (optional)"
    )
    jwt_audience: str | None = Field(
        default=None, description="JWT audience claim for token validation (optional)"
    )

    @model_validator(mode="after")
    def validate_jwt_secret_when_enabled(self) -> Settings:
        """Validate that JWT secret is provided when JWT authentication is enabled.

        This is a critical security check to prevent the server from starting
        with JWT authentication enabled but no secret key configured, which
        would be a severe security vulnerability.

        Raises:
            ValueError: If jwt_enabled is True but jwt_secret_key is None or empty.
        """
        if self.jwt_enabled:
            if self.jwt_secret_key is None:
                raise ValueError(
                    "SECURITY ERROR: JWT authentication is enabled but no secret key is configured. "
                    "This is a critical security vulnerability. "
                    "Please set the REASONING_MCP_JWT_SECRET_KEY environment variable to a secure, "
                    "randomly generated secret key (at least 32 characters recommended)."
                )
            # Check if the secret value is empty after unwrapping
            secret_value = self.jwt_secret_key.get_secret_value()
            if not secret_value or not secret_value.strip():
                raise ValueError(
                    "SECURITY ERROR: JWT authentication is enabled but the secret key is empty. "
                    "This is a critical security vulnerability. "
                    "Please set the REASONING_MCP_JWT_SECRET_KEY environment variable to a secure, "
                    "randomly generated secret key (at least 32 characters recommended)."
                )
        return self

    @model_validator(mode="after")
    def validate_telemetry_configuration(self) -> Settings:
        """Validate telemetry configuration consistency.

        If telemetry is enabled but no OTLP endpoint is configured (empty string),
        logs a warning and automatically disables telemetry to prevent connection
        errors in container/cloud environments where localhost is not available.

        For production deployments, configure the endpoint via:
            REASONING_MCP_TELEMETRY_OTLP_ENDPOINT=http://otel-collector:4317

        Returns:
            Settings instance with potentially modified telemetry settings.
        """
        if self.enable_telemetry:
            # Check if OTLP exporter is selected but no endpoint is configured
            if self.telemetry_exporter == "otlp" and not self.telemetry_otlp_endpoint.strip():
                _config_logger.warning(
                    "Telemetry is enabled with OTLP exporter but no endpoint is configured. "
                    "Disabling telemetry to prevent connection errors. "
                    "To enable telemetry, set REASONING_MCP_TELEMETRY_OTLP_ENDPOINT to your "
                    "collector URL (e.g., http://otel-collector:4317 for Kubernetes, "
                    "or http://localhost:4317 for local development)."
                )
                # Use object.__setattr__ to bypass frozen model if needed
                object.__setattr__(self, "enable_telemetry", False)

            # Warn if using console exporter in what looks like production
            elif self.telemetry_exporter == "console":
                _config_logger.info(
                    "Telemetry is using console exporter. This is suitable for development "
                    "but not recommended for production. Consider using 'otlp' exporter with "
                    "an OpenTelemetry Collector for production deployments."
                )

        return self

    # API Key Authentication settings (MCPaaS feature)
    api_key_enabled: bool = Field(
        default=False, description="Enable API key authentication (opt-in for MCPaaS)"
    )
    api_key_header: str = Field(
        default="X-API-Key", description="HTTP header name for API key authentication"
    )

    # Rate Limiting settings (MCPaaS feature)
    enable_rate_limiting: bool = Field(
        default=False, description="Enable rate limiting middleware for API abuse prevention"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60, ge=1, le=10000, description="Maximum requests per minute per client"
    )
    rate_limit_requests_per_hour: int = Field(
        default=1000, ge=1, le=1000000, description="Maximum requests per hour per client"
    )
    rate_limit_burst_size: int = Field(
        default=10, ge=1, le=1000, description="Burst size for token bucket algorithm"
    )
    rate_limit_cleanup_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Interval in seconds for cleaning up old rate limit entries",
    )
    rate_limit_bypass_keys: list[str] = Field(
        default_factory=list, description="API keys or user IDs that bypass rate limiting"
    )
    rate_limit_identification_header: str = Field(
        default="X-API-Key", description="Header name for client identification (API key)"
    )

    # OpenTelemetry settings
    # Telemetry is opt-in and disabled by default. See module docstring for production setup.
    enable_telemetry: bool = Field(
        default=False,
        description=(
            "Enable OpenTelemetry distributed tracing (opt-in). "
            "Requires telemetry_otlp_endpoint to be configured for OTLP export."
        ),
    )
    telemetry_service_name: str = Field(
        default="reasoning-mcp",
        description="Service name for telemetry spans. Override for multi-service deployments.",
    )
    telemetry_exporter: Literal["console", "otlp", "jaeger", "zipkin", "none"] = Field(
        default="otlp",
        description=(
            "Telemetry exporter type. Use 'console' for development debugging, "
            "'otlp' for production with OpenTelemetry Collector, or 'none' to disable export."
        ),
    )
    telemetry_otlp_endpoint: str = Field(
        default="",
        description=(
            "OTLP collector endpoint for telemetry export. "
            "Empty string = disabled. "
            "For production, set to your collector URL (e.g., http://otel-collector:4317). "
            "For local development: http://localhost:4317"
        ),
    )
    telemetry_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Sampling rate for telemetry (0.0-1.0). "
            "Use 1.0 for development, lower values (0.1-0.5) for high-traffic production."
        ),
    )
    telemetry_export_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Timeout for telemetry export in milliseconds.",
    )

    # Session persistence settings (diskcache)
    enable_session_persistence: bool = Field(
        default=False, description="Enable persistent session storage with diskcache"
    )
    session_storage_backend: Literal["memory", "disk", "hybrid"] = Field(
        default="hybrid", description="Session storage backend: memory, disk, or hybrid"
    )
    session_cache_dir: Path = Field(
        default=Path("~/.reasoning-mcp/sessions"),
        description="Directory for persistent session storage",
    )
    session_cache_size_mb: int = Field(
        default=500, ge=10, le=10000, description="Maximum size of session cache in megabytes"
    )
    session_lazy_load_threshold_kb: int = Field(
        default=50, ge=1, description="Threshold in KB for lazy-loading large ThoughtGraphs"
    )
    session_auto_persist_interval: int = Field(
        default=60, ge=10, description="Interval in seconds for auto-persisting dirty sessions"
    )
    session_recovery_on_startup: bool = Field(
        default=True, description="Recover persisted sessions on server startup"
    )
    session_max_recovery_age_hours: int = Field(
        default=24, ge=1, description="Maximum age in hours for session recovery"
    )


# Global settings instance (lazy-loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        The global Settings instance, created on first access.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_settings(**overrides: Any) -> Settings:
    """Configure settings with overrides.

    Args:
        **overrides: Setting values to override.

    Returns:
        New Settings instance with overrides applied.
    """
    global _settings
    _settings = Settings(**overrides)
    return _settings


__all__ = [
    "APIKeyInfo",
    "DebugConfig",
    "EnsembleConfigSettings",
    "Settings",
    "StreamingConfig",
    "VerificationConfig",
    "configure_settings",
    "get_settings",
]
