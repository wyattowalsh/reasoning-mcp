"""Configuration management for reasoning-mcp server.

This module provides the Settings class for managing server configuration
with support for environment variables and .env files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    server_name: str = Field(
        default="reasoning-mcp",
        description="Name of the MCP server"
    )
    server_version: str = Field(
        default="0.1.0",
        description="Version of the server"
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )

    # Session settings
    session_timeout: int = Field(
        default=3600,
        ge=60,
        description="Session timeout in seconds (minimum 60)"
    )
    max_sessions: int = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent sessions"
    )
    session_cleanup_interval: int = Field(
        default=300,
        ge=60,
        description="Interval for session cleanup in seconds"
    )

    # Reasoning settings
    default_method: str = Field(
        default="chain_of_thought",
        description="Default reasoning method when not specified"
    )
    max_thoughts_per_session: int = Field(
        default=1000,
        ge=10,
        description="Maximum thoughts allowed per session"
    )
    max_branches_per_session: int = Field(
        default=50,
        ge=1,
        description="Maximum branches allowed per session"
    )

    # Pipeline settings
    max_pipeline_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum nesting depth for pipelines"
    )
    pipeline_timeout: int = Field(
        default=300,
        ge=10,
        description="Pipeline execution timeout in seconds"
    )
    max_parallel_stages: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum parallel stages in a pipeline"
    )

    # Plugin settings
    plugins_dir: Path = Field(
        default=Path("plugins"),
        description="Directory for plugin discovery"
    )
    enable_plugins: bool = Field(
        default=True,
        description="Whether to enable plugin loading"
    )

    # Development settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable detailed tracing"
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


def configure_settings(**overrides) -> Settings:
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
    "Settings",
    "get_settings",
    "configure_settings",
]
