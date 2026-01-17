"""Tests for validation utilities."""
from __future__ import annotations

import pytest

from reasoning_mcp.utils.validation import (
    MAX_INPUT_LENGTH,
    ValidationError,
    sanitize_prompt,
    validate_input_length,
    validate_metadata,
    validate_method_identifier,
)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_is_value_error(self) -> None:
        """Test ValidationError is a ValueError."""
        err = ValidationError("test error")
        assert isinstance(err, ValueError)
        assert str(err) == "test error"


class TestValidateInputLength:
    """Tests for validate_input_length function."""

    def test_valid_short_input(self) -> None:
        """Test short input passes validation."""
        result = validate_input_length("hello world")
        assert result == "hello world"

    def test_valid_at_max_length(self) -> None:
        """Test input at exactly max length passes."""
        text = "x" * MAX_INPUT_LENGTH
        result = validate_input_length(text)
        assert result == text

    def test_exceeds_max_length(self) -> None:
        """Test input exceeding max length raises error."""
        text = "x" * (MAX_INPUT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_input_length(text)
        assert "exceeds maximum length" in str(exc_info.value)
        assert str(MAX_INPUT_LENGTH) in str(exc_info.value)

    def test_custom_max_length(self) -> None:
        """Test custom max length is respected."""
        text = "hello world"
        result = validate_input_length(text, max_chars=100)
        assert result == text

        with pytest.raises(ValidationError):
            validate_input_length(text, max_chars=5)

    def test_custom_field_name_in_error(self) -> None:
        """Test custom field name appears in error message."""
        text = "x" * 1000
        with pytest.raises(ValidationError) as exc_info:
            validate_input_length(text, max_chars=100, field_name="prompt")
        assert "prompt" in str(exc_info.value)

    def test_empty_input_passes(self) -> None:
        """Test empty input passes validation."""
        result = validate_input_length("")
        assert result == ""


class TestSanitizePrompt:
    """Tests for sanitize_prompt function."""

    def test_removes_null_bytes(self) -> None:
        """Test null bytes are removed."""
        text = "hello\x00world"
        result = sanitize_prompt(text)
        assert result == "helloworld"
        assert "\x00" not in result

    def test_normalizes_unicode(self) -> None:
        """Test unicode is normalized to NFC."""
        # é can be represented as single char (NFC) or e + combining accent
        text = "cafe\u0301"  # e + combining accent
        result = sanitize_prompt(text)
        # Should be normalized to single character é
        assert result == "café"

    def test_preserves_normal_text(self) -> None:
        """Test normal text is preserved."""
        text = "Hello, World! 123"
        result = sanitize_prompt(text)
        assert result == text

    def test_preserves_newlines(self) -> None:
        """Test newlines are preserved."""
        text = "line1\nline2\nline3"
        result = sanitize_prompt(text)
        assert result == text

    def test_preserves_unicode_characters(self) -> None:
        """Test unicode characters are preserved."""
        text = "日本語テスト 🎉"
        result = sanitize_prompt(text)
        assert result == text

    def test_multiple_null_bytes(self) -> None:
        """Test multiple null bytes are all removed."""
        text = "\x00hello\x00\x00world\x00"
        result = sanitize_prompt(text)
        assert result == "helloworld"


class TestValidateMetadata:
    """Tests for validate_metadata function."""

    def test_none_returns_empty_dict(self) -> None:
        """Test None input returns empty dict."""
        result = validate_metadata(None)
        assert result == {}

    def test_empty_dict_passes(self) -> None:
        """Test empty dict passes validation."""
        result = validate_metadata({})
        assert result == {}

    def test_valid_metadata_passes(self) -> None:
        """Test valid metadata dict passes."""
        meta = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        result = validate_metadata(meta)
        assert result == meta

    def test_non_dict_raises(self) -> None:
        """Test non-dict input raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata("not a dict")  # type: ignore[arg-type]
        assert "must be a dict" in str(exc_info.value)

    def test_non_dict_list_raises(self) -> None:
        """Test list input raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata([1, 2, 3])  # type: ignore[arg-type]
        assert "must be a dict" in str(exc_info.value)

    def test_non_string_keys_raise(self) -> None:
        """Test non-string keys raise error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata({1: "value"})  # type: ignore[dict-item]
        assert "keys must be strings" in str(exc_info.value)

    def test_mixed_key_types_raise(self) -> None:
        """Test mixed key types raise error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_metadata({"valid": 1, 123: 2})  # type: ignore[dict-item]
        assert "keys must be strings" in str(exc_info.value)


class TestValidateMethodIdentifier:
    """Tests for validate_method_identifier function."""

    def test_valid_simple_identifier(self) -> None:
        """Test valid simple identifier passes."""
        result = validate_method_identifier("chain_of_thought")
        assert result == "chain_of_thought"

    def test_valid_with_numbers(self) -> None:
        """Test identifier with numbers passes."""
        result = validate_method_identifier("method1")
        assert result == "method1"

    def test_valid_with_hyphen(self) -> None:
        """Test identifier with hyphen passes."""
        result = validate_method_identifier("tree-of-thoughts")
        assert result == "tree-of-thoughts"

    def test_valid_mixed(self) -> None:
        """Test mixed identifier passes."""
        result = validate_method_identifier("my-method_v2")
        assert result == "my-method_v2"

    def test_empty_raises(self) -> None:
        """Test empty identifier raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier("")
        assert "cannot be empty" in str(exc_info.value)

    def test_non_string_raises(self) -> None:
        """Test non-string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier(123)  # type: ignore[arg-type]
        assert "must be a string" in str(exc_info.value)

    def test_starts_with_number_raises(self) -> None:
        """Test identifier starting with number raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier("1method")
        assert "must start with a letter" in str(exc_info.value)

    def test_starts_with_underscore_raises(self) -> None:
        """Test identifier starting with underscore raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier("_method")
        assert "must start with a letter" in str(exc_info.value)

    def test_starts_with_hyphen_raises(self) -> None:
        """Test identifier starting with hyphen raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier("-method")
        assert "must start with a letter" in str(exc_info.value)

    def test_contains_spaces_raises(self) -> None:
        """Test identifier with spaces raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_method_identifier("my method")
        assert "alphanumeric characters" in str(exc_info.value)

    def test_contains_special_chars_raises(self) -> None:
        """Test identifier with special characters raises error."""
        for char in ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "."]:
            with pytest.raises(ValidationError):
                validate_method_identifier(f"method{char}name")

    def test_uppercase_passes(self) -> None:
        """Test uppercase identifier passes."""
        result = validate_method_identifier("ChainOfThought")
        assert result == "ChainOfThought"

    def test_single_letter_passes(self) -> None:
        """Test single letter identifier passes."""
        result = validate_method_identifier("a")
        assert result == "a"
