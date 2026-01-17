"""Tests for streaming configuration."""

from reasoning_mcp.config import Settings, StreamingConfig
from reasoning_mcp.streaming.backpressure import BackpressureStrategy


class TestStreamingConfig:
    """Tests for StreamingConfig model."""

    def test_streaming_config_defaults(self):
        """Test StreamingConfig default values."""
        config = StreamingConfig()
        assert config.enabled is True
        assert config.default_buffer_size == 1000
        assert config.default_backpressure == BackpressureStrategy.BLOCK
        assert config.sse_enabled is True
        assert config.websocket_enabled is True

    def test_streaming_config_custom(self):
        """Test StreamingConfig with custom values."""
        config = StreamingConfig(
            enabled=False,
            default_buffer_size=500,
            default_backpressure=BackpressureStrategy.DROP_OLDEST,
            sse_enabled=True,
            websocket_enabled=False,
        )
        assert config.enabled is False
        assert config.default_buffer_size == 500
        assert config.default_backpressure == BackpressureStrategy.DROP_OLDEST
        assert config.websocket_enabled is False


class TestSettingsStreamingIntegration:
    """Tests for streaming configuration in Settings."""

    def test_settings_has_streaming_field(self):
        """Test that Settings has streaming field."""
        settings = Settings()
        assert hasattr(settings, "streaming")
        assert isinstance(settings.streaming, StreamingConfig)

    def test_settings_streaming_defaults(self):
        """Test streaming defaults in Settings."""
        settings = Settings()
        assert settings.streaming.enabled is True
        assert settings.streaming.default_buffer_size == 1000
