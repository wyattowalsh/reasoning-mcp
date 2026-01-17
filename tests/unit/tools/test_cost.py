"""Unit tests for cost estimation tool."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reasoning_mcp.tools.cost import EstimateCostInput


class TestEstimateCostInput:
    """Tests for EstimateCostInput model."""

    def test_estimate_input(self):
        """Test that EstimateCostInput validates correctly."""
        # Arrange & Act
        input_data = EstimateCostInput(
            method="chain_of_thought",
            input_text="What is the meaning of life?",
        )

        # Assert
        assert input_data.method == "chain_of_thought"
        assert input_data.input_text == "What is the meaning of life?"
        assert input_data.model_id is None

    def test_estimate_input_with_model_id(self):
        """Test EstimateCostInput with explicit model_id."""
        # Arrange & Act
        input_data = EstimateCostInput(
            method="tree_of_thoughts",
            input_text="Solve this problem",
            model_id="claude-opus-4-5-20251101",
        )

        # Assert
        assert input_data.method == "tree_of_thoughts"
        assert input_data.input_text == "Solve this problem"
        assert input_data.model_id == "claude-opus-4-5-20251101"

    def test_estimate_input_missing_required_method(self):
        """Test that missing method raises ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            EstimateCostInput(
                input_text="Some text",
            )

        # Verify the error mentions the missing field
        assert "method" in str(exc_info.value)

    def test_estimate_input_missing_required_input_text(self):
        """Test that missing input_text raises ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            EstimateCostInput(
                method="chain_of_thought",
            )

        # Verify the error mentions the missing field
        assert "input_text" in str(exc_info.value)

    def test_estimate_input_empty_strings(self):
        """Test EstimateCostInput accepts empty strings."""
        # Arrange & Act
        input_data = EstimateCostInput(
            method="",
            input_text="",
        )

        # Assert - Empty strings are valid
        assert input_data.method == ""
        assert input_data.input_text == ""
        assert input_data.model_id is None

    def test_estimate_input_none_model_id_explicit(self):
        """Test EstimateCostInput with explicitly set None model_id."""
        # Arrange & Act
        input_data = EstimateCostInput(
            method="react",
            input_text="Debug this code",
            model_id=None,
        )

        # Assert
        assert input_data.method == "react"
        assert input_data.input_text == "Debug this code"
        assert input_data.model_id is None

    def test_estimate_input_auto_method(self):
        """Test EstimateCostInput with auto method selection."""
        # Arrange & Act
        input_data = EstimateCostInput(
            method="auto",
            input_text="Analyze this ethical dilemma",
            model_id="claude-sonnet-4-5-20250929",
        )

        # Assert
        assert input_data.method == "auto"
        assert input_data.input_text == "Analyze this ethical dilemma"
        assert input_data.model_id == "claude-sonnet-4-5-20250929"

    def test_estimate_input_long_text(self):
        """Test EstimateCostInput with long input text."""
        # Arrange
        long_text = "This is a very long problem description. " * 100

        # Act
        input_data = EstimateCostInput(
            method="mathematical_reasoning",
            input_text=long_text,
        )

        # Assert
        assert input_data.method == "mathematical_reasoning"
        assert input_data.input_text == long_text
        assert len(input_data.input_text) > 1000

    def test_estimate_input_special_characters(self):
        """Test EstimateCostInput with special characters in text."""
        # Arrange
        special_text = "Test with special chars: @#$%^&*()_+-=[]{}|;':\"<>,.?/~`"

        # Act
        input_data = EstimateCostInput(
            method="code_reasoning",
            input_text=special_text,
        )

        # Assert
        assert input_data.input_text == special_text

    def test_estimate_input_unicode_text(self):
        """Test EstimateCostInput with unicode characters."""
        # Arrange
        unicode_text = "Unicode test: 你好世界 🌍 émojis"

        # Act
        input_data = EstimateCostInput(
            method="chain_of_thought",
            input_text=unicode_text,
        )

        # Assert
        assert input_data.input_text == unicode_text

    def test_estimate_input_multiline_text(self):
        """Test EstimateCostInput with multiline input text."""
        # Arrange
        multiline_text = """This is a multi-line problem.
It has several lines.
Each line contains important information."""

        # Act
        input_data = EstimateCostInput(
            method="ethical_reasoning",
            input_text=multiline_text,
        )

        # Assert
        assert input_data.input_text == multiline_text
        assert "\n" in input_data.input_text

    def test_estimate_input_model_dump(self):
        """Test that EstimateCostInput can be serialized to dict."""
        # Arrange
        input_data = EstimateCostInput(
            method="tree_of_thoughts",
            input_text="Test problem",
            model_id="test-model",
        )

        # Act
        dumped = input_data.model_dump()

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["method"] == "tree_of_thoughts"
        assert dumped["input_text"] == "Test problem"
        assert dumped["model_id"] == "test-model"

    def test_estimate_input_model_dump_json(self):
        """Test that EstimateCostInput can be serialized to JSON."""
        # Arrange
        input_data = EstimateCostInput(
            method="chain_of_thought",
            input_text="Test problem",
        )

        # Act
        json_str = input_data.model_dump_json()

        # Assert
        assert isinstance(json_str, str)
        assert "chain_of_thought" in json_str
        assert "Test problem" in json_str

    def test_estimate_input_model_validate(self):
        """Test creating EstimateCostInput from dict using model_validate."""
        # Arrange
        data = {
            "method": "react",
            "input_text": "Debug this issue",
            "model_id": "claude-opus-4-5-20251101",
        }

        # Act
        input_data = EstimateCostInput.model_validate(data)

        # Assert
        assert input_data.method == "react"
        assert input_data.input_text == "Debug this issue"
        assert input_data.model_id == "claude-opus-4-5-20251101"

    def test_estimate_input_immutability(self):
        """Test that EstimateCostInput fields can be modified (not frozen)."""
        # Arrange
        input_data = EstimateCostInput(
            method="chain_of_thought",
            input_text="Original text",
        )

        # Act & Assert - Should NOT raise since BaseModel is mutable by default
        input_data.method = "tree_of_thoughts"
        assert input_data.method == "tree_of_thoughts"

    def test_estimate_input_extra_fields_ignored(self):
        """Test that extra fields are ignored by default in Pydantic v2."""
        # Arrange & Act - Extra fields don't raise, they're ignored
        input_data = EstimateCostInput(
            method="chain_of_thought",
            input_text="Test",
            extra_field="ignored",  # type: ignore[call-arg]
        )

        # Assert - Model is created successfully, extra field is ignored
        assert input_data.method == "chain_of_thought"
        assert input_data.input_text == "Test"
        assert not hasattr(input_data, "extra_field")
