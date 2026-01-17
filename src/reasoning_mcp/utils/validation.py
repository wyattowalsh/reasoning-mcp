"""Input validation utilities for reasoning-mcp."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

MAX_INPUT_LENGTH = 50000  # 50KB default max


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_input_length(
    text: str,
    max_chars: int = MAX_INPUT_LENGTH,
    field_name: str = "input",
) -> str:
    """Validate input text length.

    Args:
        text: Input text to validate
        max_chars: Maximum allowed characters
        field_name: Name of field for error messages

    Returns:
        The validated text

    Raises:
        ValidationError: If text exceeds max length
    """
    if len(text) > max_chars:
        raise ValidationError(
            f"{field_name} exceeds maximum length of {max_chars} characters "
            f"(got {len(text)})"
        )
    return text


def sanitize_prompt(text: str) -> str:
    """Basic sanitization for prompts.

    Removes potentially problematic patterns while preserving content.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    return text


def validate_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Validate metadata dictionary.

    Args:
        meta: Metadata dict or None

    Returns:
        Validated metadata (empty dict if None)

    Raises:
        ValidationError: If metadata contains invalid types
    """
    if meta is None:
        return {}

    if not isinstance(meta, dict):
        raise ValidationError(f"metadata must be a dict, got {type(meta).__name__}")

    # Validate all keys are strings
    for key in meta:
        if not isinstance(key, str):
            raise ValidationError(
                f"metadata keys must be strings, got {type(key).__name__}"
            )

    return meta


def validate_method_identifier(method_id: str) -> str:
    """Validate a method identifier string.

    Args:
        method_id: Method identifier to validate

    Returns:
        Validated method identifier

    Raises:
        ValidationError: If identifier is invalid
    """
    if not method_id:
        raise ValidationError("method_id cannot be empty")

    if not isinstance(method_id, str):
        raise ValidationError(
            f"method_id must be a string, got {type(method_id).__name__}"
        )

    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", method_id):
        raise ValidationError(
            f"method_id must start with a letter and contain only "
            f"alphanumeric characters, underscores, or hyphens: {method_id}"
        )

    return method_id
