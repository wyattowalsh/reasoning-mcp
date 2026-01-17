"""Tests for cost tracking models.

This module provides comprehensive test coverage for the cost tracking models
in reasoning_mcp.models.cost, including:
- TokenCount model with computed total_tokens field
- Model validation and serialization
"""

import pytest
from pydantic import ValidationError

from reasoning_mcp.models.cost import TokenCount


class TestTokenCount:
    """Test suite for TokenCount model."""

    def test_token_count_creation(self) -> None:
        """Test creating TokenCount with input and output tokens."""
        # Arrange & Act
        tokens = TokenCount(input_tokens=100, output_tokens=50)

        # Assert
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50

    def test_total_tokens_computed_correctly(self) -> None:
        """Test that total_tokens is computed correctly."""
        # Arrange
        tokens = TokenCount(input_tokens=100, output_tokens=50)

        # Act & Assert
        assert tokens.total_tokens == 150

    def test_total_tokens_zero(self) -> None:
        """Test total_tokens with zero tokens."""
        # Arrange
        tokens = TokenCount(input_tokens=0, output_tokens=0)

        # Act & Assert
        assert tokens.total_tokens == 0

    def test_model_serialization(self) -> None:
        """Test that model serialization includes computed field."""
        # Arrange
        tokens = TokenCount(input_tokens=100, output_tokens=50)

        # Act
        data = tokens.model_dump()

        # Assert
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["total_tokens"] == 150

    def test_model_json_serialization(self) -> None:
        """Test that model JSON serialization includes computed field."""
        # Arrange
        tokens = TokenCount(input_tokens=100, output_tokens=50)

        # Act
        json_str = tokens.model_dump_json()

        # Assert
        assert '"input_tokens":100' in json_str
        assert '"output_tokens":50' in json_str
        assert '"total_tokens":150' in json_str

    def test_negative_input_tokens_rejected(self) -> None:
        """Test that negative input_tokens are rejected."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TokenCount(input_tokens=-1, output_tokens=50)

        # Verify error details
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("input_tokens",)
        assert "greater than or equal to 0" in errors[0]["msg"]

    def test_negative_output_tokens_rejected(self) -> None:
        """Test that negative output_tokens are rejected."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TokenCount(input_tokens=100, output_tokens=-1)

        # Verify error details
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("output_tokens",)
        assert "greater than or equal to 0" in errors[0]["msg"]

    def test_large_token_counts(self) -> None:
        """Test with large token counts."""
        # Arrange
        tokens = TokenCount(input_tokens=1_000_000, output_tokens=500_000)

        # Act & Assert
        assert tokens.total_tokens == 1_500_000

    def test_model_deserialization(self) -> None:
        """Test that model can be deserialized from dict."""
        # Arrange
        data = {"input_tokens": 100, "output_tokens": 50}

        # Act
        tokens = TokenCount(**data)

        # Assert
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50
        assert tokens.total_tokens == 150

    def test_model_parse_obj(self) -> None:
        """Test that model can be parsed from dict using model_validate."""
        # Arrange
        data = {"input_tokens": 200, "output_tokens": 100}

        # Act
        tokens = TokenCount.model_validate(data)

        # Assert
        assert tokens.input_tokens == 200
        assert tokens.output_tokens == 100
        assert tokens.total_tokens == 300

    def test_equality(self) -> None:
        """Test equality comparison between TokenCount instances."""
        # Arrange
        tokens1 = TokenCount(input_tokens=100, output_tokens=50)
        tokens2 = TokenCount(input_tokens=100, output_tokens=50)
        tokens3 = TokenCount(input_tokens=200, output_tokens=50)

        # Act & Assert
        assert tokens1 == tokens2
        assert tokens1 != tokens3

    def test_immutability(self) -> None:
        """Test that fields can be modified (Pydantic v2 allows mutation by default)."""
        # Arrange
        tokens = TokenCount(input_tokens=100, output_tokens=50)

        # Act - Pydantic v2 models are mutable by default
        tokens.input_tokens = 200

        # Assert
        assert tokens.input_tokens == 200
        assert tokens.total_tokens == 250  # Computed field updates

    def test_model_schema(self) -> None:
        """Test that model schema includes required fields.

        Note: Computed fields (like total_tokens) are not included in JSON schema
        by default in Pydantic v2 - they are runtime-only properties.
        """
        # Act
        schema = TokenCount.model_json_schema()

        # Assert
        assert "properties" in schema
        assert "input_tokens" in schema["properties"]
        assert "output_tokens" in schema["properties"]
        # Computed fields are not included in JSON schema by default
        assert "required" in schema
        assert set(schema["required"]) == {"input_tokens", "output_tokens"}

    def test_missing_required_field(self) -> None:
        """Test that missing required fields raise ValidationError."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            TokenCount(input_tokens=100)  # type: ignore

        # Verify error details
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("output_tokens",)
        assert errors[0]["type"] == "missing"
