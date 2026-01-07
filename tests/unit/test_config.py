"""Tests for reasoning_mcp.config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reasoning_mcp.config import Settings, configure_settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test Settings has correct default values."""
        settings = Settings()
        assert settings.server_name == "reasoning-mcp"
        assert settings.server_version == "0.1.0"
        assert settings.log_level == "INFO"
        assert settings.log_format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert settings.session_timeout == 3600
        assert settings.max_sessions == 100
        assert settings.session_cleanup_interval == 300
        assert settings.default_method == "chain_of_thought"
        assert settings.max_thoughts_per_session == 1000
        assert settings.max_branches_per_session == 50
        assert settings.max_pipeline_depth == 10
        assert settings.pipeline_timeout == 300
        assert settings.max_parallel_stages == 8
        assert settings.plugins_dir == Path("plugins")
        assert settings.enable_plugins is True
        assert settings.debug is False
        assert settings.enable_tracing is False

    def test_session_timeout_minimum(self):
        """Test session_timeout has minimum validation."""
        with pytest.raises(ValueError):
            Settings(session_timeout=30)  # Below minimum of 60

    def test_session_timeout_valid(self):
        """Test session_timeout accepts valid values."""
        settings = Settings(session_timeout=60)
        assert settings.session_timeout == 60

        settings = Settings(session_timeout=7200)
        assert settings.session_timeout == 7200

    def test_max_sessions_minimum(self):
        """Test max_sessions has minimum validation."""
        with pytest.raises(ValueError):
            Settings(max_sessions=0)  # Below minimum of 1

    def test_max_sessions_valid(self):
        """Test max_sessions accepts valid values."""
        settings = Settings(max_sessions=1)
        assert settings.max_sessions == 1

        settings = Settings(max_sessions=500)
        assert settings.max_sessions == 500

    def test_session_cleanup_interval_minimum(self):
        """Test session_cleanup_interval has minimum validation."""
        with pytest.raises(ValueError):
            Settings(session_cleanup_interval=30)  # Below minimum of 60

    def test_session_cleanup_interval_valid(self):
        """Test session_cleanup_interval accepts valid values."""
        settings = Settings(session_cleanup_interval=60)
        assert settings.session_cleanup_interval == 60

    def test_max_thoughts_per_session_minimum(self):
        """Test max_thoughts_per_session has minimum validation."""
        with pytest.raises(ValueError):
            Settings(max_thoughts_per_session=5)  # Below minimum of 10

    def test_max_thoughts_per_session_valid(self):
        """Test max_thoughts_per_session accepts valid values."""
        settings = Settings(max_thoughts_per_session=10)
        assert settings.max_thoughts_per_session == 10

        settings = Settings(max_thoughts_per_session=5000)
        assert settings.max_thoughts_per_session == 5000

    def test_max_branches_per_session_minimum(self):
        """Test max_branches_per_session has minimum validation."""
        with pytest.raises(ValueError):
            Settings(max_branches_per_session=0)  # Below minimum of 1

    def test_max_branches_per_session_valid(self):
        """Test max_branches_per_session accepts valid values."""
        settings = Settings(max_branches_per_session=1)
        assert settings.max_branches_per_session == 1

    def test_max_pipeline_depth_range(self):
        """Test max_pipeline_depth has range validation."""
        # Valid range
        settings = Settings(max_pipeline_depth=50)
        assert settings.max_pipeline_depth == 50

        # Below minimum
        with pytest.raises(ValueError):
            Settings(max_pipeline_depth=0)

        # Above maximum
        with pytest.raises(ValueError):
            Settings(max_pipeline_depth=101)

    def test_max_pipeline_depth_boundary_values(self):
        """Test max_pipeline_depth boundary values."""
        # Minimum valid value
        settings = Settings(max_pipeline_depth=1)
        assert settings.max_pipeline_depth == 1

        # Maximum valid value
        settings = Settings(max_pipeline_depth=100)
        assert settings.max_pipeline_depth == 100

    def test_pipeline_timeout_minimum(self):
        """Test pipeline_timeout has minimum validation."""
        with pytest.raises(ValueError):
            Settings(pipeline_timeout=5)  # Below minimum of 10

    def test_pipeline_timeout_valid(self):
        """Test pipeline_timeout accepts valid values."""
        settings = Settings(pipeline_timeout=10)
        assert settings.pipeline_timeout == 10

    def test_max_parallel_stages_range(self):
        """Test max_parallel_stages has range validation."""
        # Valid range
        settings = Settings(max_parallel_stages=16)
        assert settings.max_parallel_stages == 16

        # Below minimum
        with pytest.raises(ValueError):
            Settings(max_parallel_stages=0)

        # Above maximum
        with pytest.raises(ValueError):
            Settings(max_parallel_stages=33)

    def test_max_parallel_stages_boundary_values(self):
        """Test max_parallel_stages boundary values."""
        # Minimum valid value
        settings = Settings(max_parallel_stages=1)
        assert settings.max_parallel_stages == 1

        # Maximum valid value
        settings = Settings(max_parallel_stages=32)
        assert settings.max_parallel_stages == 32

    def test_log_level_literal(self):
        """Test log_level accepts only valid literals."""
        # Valid values
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Invalid value
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_plugins_dir_path(self):
        """Test plugins_dir is a Path."""
        settings = Settings()
        assert isinstance(settings.plugins_dir, Path)

    def test_plugins_dir_custom_path(self):
        """Test plugins_dir accepts custom paths."""
        custom_path = Path("/custom/plugins")
        settings = Settings(plugins_dir=custom_path)
        assert settings.plugins_dir == custom_path

        # Test string path is converted to Path
        settings = Settings(plugins_dir="custom/path")
        assert isinstance(settings.plugins_dir, Path)
        assert settings.plugins_dir == Path("custom/path")

    def test_boolean_fields(self):
        """Test boolean configuration fields."""
        # Test enable_plugins
        settings = Settings(enable_plugins=False)
        assert settings.enable_plugins is False

        # Test debug
        settings = Settings(debug=True)
        assert settings.debug is True

        # Test enable_tracing
        settings = Settings(enable_tracing=True)
        assert settings.enable_tracing is True

    def test_string_fields(self):
        """Test string configuration fields."""
        # Test server_name
        settings = Settings(server_name="custom-server")
        assert settings.server_name == "custom-server"

        # Test server_version
        settings = Settings(server_version="1.2.3")
        assert settings.server_version == "1.2.3"

        # Test default_method
        settings = Settings(default_method="tree_of_thoughts")
        assert settings.default_method == "tree_of_thoughts"

        # Test log_format
        custom_format = "%(levelname)s: %(message)s"
        settings = Settings(log_format=custom_format)
        assert settings.log_format == custom_format

    def test_environment_variable_prefix(self):
        """Test settings can be configured via environment variables."""
        # Set environment variables
        os.environ["REASONING_MCP_DEBUG"] = "true"
        os.environ["REASONING_MCP_LOG_LEVEL"] = "DEBUG"
        os.environ["REASONING_MCP_SESSION_TIMEOUT"] = "7200"
        os.environ["REASONING_MCP_MAX_SESSIONS"] = "200"

        try:
            settings = Settings()
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.session_timeout == 7200
            assert settings.max_sessions == 200
        finally:
            # Clean up
            del os.environ["REASONING_MCP_DEBUG"]
            del os.environ["REASONING_MCP_LOG_LEVEL"]
            del os.environ["REASONING_MCP_SESSION_TIMEOUT"]
            del os.environ["REASONING_MCP_MAX_SESSIONS"]

    def test_environment_variable_boolean_parsing(self):
        """Test boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("0", False),
        ]

        for env_value, expected in test_cases:
            os.environ["REASONING_MCP_ENABLE_PLUGINS"] = env_value
            try:
                settings = Settings()
                assert settings.enable_plugins is expected, f"Failed for {env_value}"
            finally:
                del os.environ["REASONING_MCP_ENABLE_PLUGINS"]

    def test_environment_variable_path_parsing(self):
        """Test Path environment variables are parsed correctly."""
        os.environ["REASONING_MCP_PLUGINS_DIR"] = "/custom/plugin/path"

        try:
            settings = Settings()
            assert isinstance(settings.plugins_dir, Path)
            assert settings.plugins_dir == Path("/custom/plugin/path")
        finally:
            del os.environ["REASONING_MCP_PLUGINS_DIR"]

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored (model_config extra='ignore')."""
        # This should not raise an error
        settings = Settings(unknown_field="value")
        assert not hasattr(settings, "unknown_field")

    def test_multiple_overrides(self):
        """Test creating Settings with multiple overrides."""
        settings = Settings(
            server_name="test-server",
            debug=True,
            log_level="DEBUG",
            session_timeout=1800,
            max_sessions=50,
            max_pipeline_depth=20,
            enable_plugins=False,
        )

        assert settings.server_name == "test-server"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.session_timeout == 1800
        assert settings.max_sessions == 50
        assert settings.max_pipeline_depth == 20
        assert settings.enable_plugins is False


class TestGetSettings:
    """Tests for get_settings function."""

    def setup_method(self):
        """Reset global settings before each test."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None

    def teardown_method(self):
        """Clean up global settings after each test."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None

    def test_returns_settings_instance(self):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_singleton_behavior(self):
        """Test get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_lazy_initialization(self):
        """Test settings are created only on first access."""
        import reasoning_mcp.config

        # Initially None
        assert reasoning_mcp.config._settings is None

        # Created on first access
        settings = get_settings()
        assert reasoning_mcp.config._settings is settings

        # Same instance on subsequent access
        assert get_settings() is settings


class TestConfigureSettings:
    """Tests for configure_settings function."""

    def setup_method(self):
        """Reset global settings before each test."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None

    def teardown_method(self):
        """Clean up global settings after each test."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None

    def test_applies_overrides(self):
        """Test configure_settings applies overrides."""
        settings = configure_settings(debug=True, log_level="DEBUG")
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_returns_new_instance(self):
        """Test configure_settings creates new instance."""
        original = get_settings()
        new = configure_settings(debug=True)

        # After configure_settings, get_settings should return the new one
        assert get_settings() is new
        assert get_settings() is not original

    def test_replaces_global_settings(self):
        """Test configure_settings replaces the global settings instance."""
        # Get initial settings
        settings1 = get_settings()
        settings1_id = id(settings1)

        # Configure with overrides
        settings2 = configure_settings(session_timeout=7200)
        settings2_id = id(settings2)

        # Should be different instances
        assert settings1_id != settings2_id

        # get_settings should return the new instance
        assert get_settings() is settings2
        assert get_settings().session_timeout == 7200

    def test_multiple_configure_calls(self):
        """Test multiple calls to configure_settings."""
        # First configuration
        settings1 = configure_settings(debug=True)
        assert settings1.debug is True
        assert settings1.log_level == "INFO"  # Default

        # Second configuration
        settings2 = configure_settings(log_level="ERROR")
        assert settings2.debug is False  # Reset to default
        assert settings2.log_level == "ERROR"

        # get_settings returns the latest
        assert get_settings() is settings2

    def test_configure_with_validation_error(self):
        """Test configure_settings raises validation errors."""
        with pytest.raises(ValueError):
            configure_settings(session_timeout=30)  # Below minimum

    def test_configure_multiple_overrides(self):
        """Test configure_settings with multiple overrides."""
        settings = configure_settings(
            server_name="configured-server",
            debug=True,
            log_level="WARNING",
            session_timeout=1800,
            max_pipeline_depth=50,
            enable_tracing=True,
        )

        assert settings.server_name == "configured-server"
        assert settings.debug is True
        assert settings.log_level == "WARNING"
        assert settings.session_timeout == 1800
        assert settings.max_pipeline_depth == 50
        assert settings.enable_tracing is True

        # Verify via get_settings
        global_settings = get_settings()
        assert global_settings is settings
        assert global_settings.server_name == "configured-server"


class TestSettingsIntegration:
    """Integration tests for Settings with environment variables."""

    def setup_method(self):
        """Store original environment and reset settings."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None
        self._original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment and clean up settings."""
        import reasoning_mcp.config
        reasoning_mcp.config._settings = None

        # Restore original environment
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_environment_overrides_defaults(self):
        """Test environment variables override default values."""
        os.environ["REASONING_MCP_SERVER_NAME"] = "env-server"
        os.environ["REASONING_MCP_DEBUG"] = "true"
        os.environ["REASONING_MCP_MAX_SESSIONS"] = "250"

        settings = Settings()
        assert settings.server_name == "env-server"
        assert settings.debug is True
        assert settings.max_sessions == 250

    def test_constructor_overrides_environment(self):
        """Test constructor arguments override environment variables."""
        os.environ["REASONING_MCP_DEBUG"] = "true"
        os.environ["REASONING_MCP_LOG_LEVEL"] = "ERROR"

        settings = Settings(debug=False, log_level="DEBUG")
        assert settings.debug is False
        assert settings.log_level == "DEBUG"

    def test_mixed_environment_and_defaults(self):
        """Test mixing environment variables with defaults."""
        os.environ["REASONING_MCP_DEBUG"] = "true"

        settings = Settings()
        assert settings.debug is True  # From environment
        assert settings.log_level == "INFO"  # Default
        assert settings.session_timeout == 3600  # Default
