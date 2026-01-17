"""User elicitation utilities for reasoning-mcp.

This module provides elicitation utilities that wrap FastMCP v2.14+'s ctx.elicit()
functionality for interactive user input during reasoning sessions.

FastMCP v2.14+ Elicitation Features:
- ctx.elicit(): Request structured input from users
- Pydantic models for type-safe user responses
- Rich form-like interactions with validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from fastmcp.server import Context
    from fastmcp.server.elicitation import (
        AcceptedElicitation,
        CancelledElicitation,
        DeclinedElicitation,
    )

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ElicitationAction(str, Enum):
    """Common elicitation action types."""

    CONFIRM = "confirm"
    SELECT = "select"
    INPUT = "input"
    RATE = "rate"
    FEEDBACK = "feedback"


@dataclass
class ElicitationConfig:
    """Configuration for elicitation operations.

    Attributes:
        required: Whether a response is required to continue
        default_on_timeout: Default value to use if user declines/cancels
        schema_defaults: Default values for schema fields (SEP-1034)
        timeout: Timeout in seconds for user response (optional)
    """

    required: bool = True
    default_on_timeout: Any = None
    schema_defaults: dict[str, Any] = field(default_factory=dict)
    timeout: int | None = None


class ElicitationError(Exception):
    """Base exception for elicitation-related errors.

    This exception is raised when elicitation operations fail, such as when
    user declines, cancels, or times out on a required elicitation request.
    """

    def __init__(self, message: str, action: ElicitationAction | None = None) -> None:
        """Initialize elicitation error.

        Args:
            message: Error description
            action: The elicitation action that failed (optional)
        """
        super().__init__(message)
        self.action = action


# Pre-built Pydantic models for common elicitation patterns


class ConfirmationResponse(BaseModel):
    """Response model for yes/no confirmations."""

    confirmed: bool = Field(description="Whether the user confirmed the action")
    reason: str | None = Field(default=None, description="Optional reason for the choice")


class SelectionResponse(BaseModel):
    """Response model for selection from options."""

    selected: str = Field(description="The selected option identifier")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="User's confidence in selection"
    )


class MultiSelectResponse(BaseModel):
    """Response model for multi-select from options (SEP-1330)."""

    selected: list[str] = Field(description="List of selected option identifiers")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="User's confidence in selections"
    )


class RatingResponse(BaseModel):
    """Response model for rating/scoring."""

    rating: int = Field(ge=1, le=10, description="Rating from 1-10")
    feedback: str | None = Field(default=None, description="Optional feedback text")


class FeedbackResponse(BaseModel):
    """Response model for open-ended feedback."""

    feedback: str = Field(description="User's feedback text")
    category: str | None = Field(default=None, description="Optional feedback category")


class ReasoningGuidanceResponse(BaseModel):
    """Response model for reasoning guidance/direction."""

    direction: str = Field(description="User's preferred direction for reasoning")
    focus_areas: list[str] = Field(default_factory=list, description="Specific areas to focus on")
    avoid_topics: list[str] = Field(
        default_factory=list, description="Topics to avoid or de-emphasize"
    )


def apply_schema_defaults(
    response_model: type[T],
    defaults: dict[str, Any],
) -> type[T]:
    """Apply default values to a Pydantic schema (SEP-1034).

    Creates a modified schema with default values for specified fields.
    This enables pre-populated form fields in elicitation UIs.

    Args:
        response_model: Original Pydantic model class
        defaults: Dict of field names to default values

    Returns:
        Modified model class with defaults applied

    Example:
        >>> DefaultedRating = apply_schema_defaults(
        ...     RatingResponse,
        ...     {"rating": 7, "feedback": "Good so far"}
        ... )
    """
    if not defaults:
        return response_model

    # Create field overrides with defaults
    field_definitions: dict[str, Any] = {}
    for field_name, field_info in response_model.model_fields.items():
        if field_name in defaults:
            # Create new field with default
            field_definitions[field_name] = (
                field_info.annotation,
                Field(default=defaults[field_name], description=field_info.description),
            )
        else:
            field_definitions[field_name] = (field_info.annotation, field_info)

    # Create new model with defaults
    return create_model(
        f"{response_model.__name__}WithDefaults",
        __base__=response_model,
        **{k: v for k, v in field_definitions.items() if k in defaults},
    )


def _is_accepted(
    result: AcceptedElicitation[Any] | DeclinedElicitation | CancelledElicitation,
) -> bool:
    """Check if an elicitation result was accepted."""
    return hasattr(result, "data")


async def elicit_confirmation(
    ctx: Context,
    message: str,
    *,
    config: ElicitationConfig | None = None,
) -> ConfirmationResponse:
    """Elicit a yes/no confirmation from the user.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The confirmation message to display
        config: Optional elicitation configuration

    Returns:
        ConfirmationResponse with the user's choice

    Example:
        >>> response = await elicit_confirmation(
        ...     ctx,
        ...     "The reasoning suggests a risky approach. Continue?",
        ... )
        >>> if response.confirmed:
        ...     # proceed with risky approach
    """
    if config is None:
        config = ElicitationConfig()

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(message, ConfirmationResponse)

        if _is_accepted(result):
            # Access the data attribute from AcceptedElicitation
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, ConfirmationResponse):
                return data
            # Convert dict to model if needed
            if isinstance(data, dict):
                return ConfirmationResponse.model_validate(data)
            return ConfirmationResponse(
                confirmed=bool(data.get("confirmed", False) if isinstance(data, dict) else False),
                reason="Parsed from response",
            )

        # User declined or cancelled
        if config.required:
            raise TimeoutError("User declined or cancelled the confirmation request")
        return ConfirmationResponse(
            confirmed=config.default_on_timeout or False,
            reason="User declined/cancelled - using default",
        )

    except AttributeError:
        logger.warning(
            "ctx.elicit() not available. Elicitation requires FastMCP v2.10+ "
            "with elicitation support enabled. Returning default."
        )
        return ConfirmationResponse(
            confirmed=config.default_on_timeout or True,
            reason="Elicitation not available - using default",
        )


async def elicit_selection(
    ctx: Context,
    message: str,
    options: list[dict[str, str]],
    *,
    config: ElicitationConfig | None = None,
) -> SelectionResponse:
    """Elicit a selection from predefined options.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The selection prompt message
        options: List of options with "id" and "label" keys
        config: Optional elicitation configuration

    Returns:
        SelectionResponse with the selected option

    Example:
        >>> options = [
        ...     {"id": "cot", "label": "Chain of Thought"},
        ...     {"id": "tot", "label": "Tree of Thoughts"},
        ...     {"id": "sc", "label": "Self-Consistency"},
        ... ]
        >>> response = await elicit_selection(
        ...     ctx,
        ...     "Which reasoning method would you prefer?",
        ...     options,
        ... )
        >>> selected_method = response.selected
    """
    if config is None:
        config = ElicitationConfig()

    # Build rich message with options
    options_text = "\n".join(f"  [{opt['id']}] {opt['label']}" for opt in options)
    full_message = f"{message}\n\nOptions:\n{options_text}"

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(full_message, SelectionResponse)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, SelectionResponse):
                response = data
            elif isinstance(data, dict):
                response = SelectionResponse.model_validate(data)
            else:
                default = config.default_on_timeout or options[0]["id"]
                return SelectionResponse(selected=default, confidence=0.5)

            # Validate selection is in options
            valid_ids = {opt["id"] for opt in options}
            if response.selected not in valid_ids:
                logger.warning(
                    f"Invalid selection '{response.selected}', defaulting to first option"
                )
                return SelectionResponse(selected=options[0]["id"], confidence=0.5)
            return response

        # User declined or cancelled
        default = config.default_on_timeout or options[0]["id"]
        return SelectionResponse(selected=default, confidence=0.5)

    except AttributeError:
        logger.warning("ctx.elicit() not available. Using first option as default.")
        default = config.default_on_timeout or options[0]["id"]
        return SelectionResponse(selected=default, confidence=0.5)


async def elicit_multi_select(
    ctx: Context,
    message: str,
    options: list[dict[str, str]],
    *,
    min_selections: int = 1,
    max_selections: int | None = None,
    config: ElicitationConfig | None = None,
) -> MultiSelectResponse:
    """Elicit multiple selections from predefined options (SEP-1330).

    FastMCP v2.14+ feature for multi-select enum schemas.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The selection prompt message
        options: List of options with "id" and "label" keys
        min_selections: Minimum number of selections required (default 1)
        max_selections: Maximum selections allowed (None for unlimited)
        config: Optional elicitation configuration

    Returns:
        MultiSelectResponse with list of selected options

    Example:
        >>> options = [
        ...     {"id": "accuracy", "label": "High Accuracy"},
        ...     {"id": "speed", "label": "Fast Response"},
        ...     {"id": "depth", "label": "Deep Analysis"},
        ... ]
        >>> response = await elicit_multi_select(
        ...     ctx,
        ...     "Which qualities are most important? (select multiple)",
        ...     options,
        ...     min_selections=1,
        ...     max_selections=2,
        ... )
    """
    if config is None:
        config = ElicitationConfig()

    # Build message with options and constraints
    options_text = "\n".join(f"  [{opt['id']}] {opt['label']}" for opt in options)
    constraints = f"Select {min_selections}"
    if max_selections:
        constraints += f" to {max_selections}"
    constraints += " option(s)."

    full_message = f"{message}\n\n{constraints}\n\nOptions:\n{options_text}"

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(full_message, MultiSelectResponse)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, MultiSelectResponse):
                response = data
            elif isinstance(data, dict):
                response = MultiSelectResponse.model_validate(data)
            else:
                # Return first option(s) as default
                default_count = min(min_selections, len(options))
                return MultiSelectResponse(
                    selected=[opt["id"] for opt in options[:default_count]],
                    confidence=0.5,
                )

            # Validate selections
            valid_ids = {opt["id"] for opt in options}
            valid_selections = [s for s in response.selected if s in valid_ids]

            if len(valid_selections) < min_selections:
                logger.warning("Too few valid selections, padding with defaults")
                needed = min_selections - len(valid_selections)
                for opt in options:
                    if opt["id"] not in valid_selections and needed > 0:
                        valid_selections.append(opt["id"])
                        needed -= 1

            if max_selections and len(valid_selections) > max_selections:
                valid_selections = valid_selections[:max_selections]

            return MultiSelectResponse(selected=valid_selections, confidence=response.confidence)

        # User declined or cancelled - return defaults
        default_count = min(min_selections, len(options))
        return MultiSelectResponse(
            selected=[opt["id"] for opt in options[:default_count]],
            confidence=0.5,
        )

    except AttributeError:
        logger.warning("ctx.elicit() not available. Using defaults.")
        default_count = min(min_selections, len(options))
        return MultiSelectResponse(
            selected=[opt["id"] for opt in options[:default_count]],
            confidence=0.5,
        )


async def elicit_rating(
    ctx: Context,
    message: str,
    *,
    default_rating: int | None = None,
    config: ElicitationConfig | None = None,
) -> RatingResponse:
    """Elicit a rating from the user.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The rating prompt message
        default_rating: Optional default rating (SEP-1034 schema default)
        config: Optional elicitation configuration

    Returns:
        RatingResponse with rating and optional feedback
    """
    if config is None:
        config = ElicitationConfig()

    full_message = f"{message}\n\nPlease rate from 1 (lowest) to 10 (highest)."

    # Apply schema defaults if provided (SEP-1034)
    schema: type[RatingResponse] = RatingResponse
    if default_rating is not None:
        schema = apply_schema_defaults(RatingResponse, {"rating": default_rating})
    elif config.schema_defaults:
        schema = apply_schema_defaults(RatingResponse, config.schema_defaults)

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(full_message, schema)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, RatingResponse):
                return data
            if isinstance(data, dict):
                return RatingResponse.model_validate(data)
            return RatingResponse(rating=config.default_on_timeout or 5)

        # User declined or cancelled
        if config.required:
            raise TimeoutError("User declined or cancelled the rating request")
        return RatingResponse(rating=config.default_on_timeout or 5)

    except AttributeError:
        logger.warning("ctx.elicit() not available. Using neutral rating.")
        return RatingResponse(rating=5, feedback="Elicitation not available")


async def elicit_feedback(
    ctx: Context,
    message: str,
    *,
    config: ElicitationConfig | None = None,
) -> FeedbackResponse:
    """Elicit open-ended feedback from the user.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The feedback prompt message
        config: Optional elicitation configuration

    Returns:
        FeedbackResponse with user's feedback text
    """
    if config is None:
        config = ElicitationConfig()

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(message, FeedbackResponse)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, FeedbackResponse):
                return data
            if isinstance(data, dict):
                return FeedbackResponse.model_validate(data)
            return FeedbackResponse(feedback="", category="invalid_response")

        # User declined or cancelled
        if config.required:
            raise TimeoutError("User declined or cancelled the feedback request")
        return FeedbackResponse(feedback="", category="declined")

    except AttributeError:
        logger.warning("ctx.elicit() not available.")
        return FeedbackResponse(feedback="", category="unavailable")


async def elicit_reasoning_guidance(
    ctx: Context,
    problem: str,
    current_state: str,
    *,
    config: ElicitationConfig | None = None,
) -> ReasoningGuidanceResponse:
    """Elicit guidance for the reasoning process from the user.

    This is useful when the reasoning method needs user direction to
    continue effectively.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        problem: The original problem being reasoned about
        current_state: Summary of current reasoning state
        config: Optional elicitation configuration

    Returns:
        ReasoningGuidanceResponse with direction and focus areas
    """
    if config is None:
        config = ElicitationConfig()

    message = (
        f"The reasoning process has reached a decision point.\n\n"
        f"Problem: {problem[:200]}{'...' if len(problem) > 200 else ''}\n\n"
        f"Current State: {current_state}\n\n"
        f"Please provide guidance on how to proceed:"
    )

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(message, ReasoningGuidanceResponse)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, ReasoningGuidanceResponse):
                return data
            if isinstance(data, dict):
                return ReasoningGuidanceResponse.model_validate(data)
            return ReasoningGuidanceResponse(direction="continue")

        # User declined or cancelled
        if config.required:
            raise TimeoutError("User declined or cancelled the guidance request")
        return ReasoningGuidanceResponse(direction="continue")

    except AttributeError:
        logger.warning("ctx.elicit() not available. Continuing with default direction.")
        return ReasoningGuidanceResponse(direction="continue")


async def elicit_custom(
    ctx: Context,
    message: str,
    response_model: type[T],
    *,
    config: ElicitationConfig | None = None,
) -> T | None:
    """Elicit a custom structured response from the user.

    Args:
        ctx: FastMCP Context with elicitation capabilities
        message: The elicitation prompt message
        response_model: Pydantic model class for the expected response
        config: Optional elicitation configuration

    Returns:
        Instance of response_model or None if declined/cancelled and not required

    Example:
        >>> class CustomInput(BaseModel):
        ...     value: str
        ...     priority: int
        ...
        >>> response = await elicit_custom(
        ...     ctx,
        ...     "Please provide your input:",
        ...     CustomInput,
        ... )
    """
    if config is None:
        config = ElicitationConfig()

    try:
        # FastMCP uses @overload for elicit - mypy may not resolve correctly
        result = await ctx.elicit(message, response_model)

        if _is_accepted(result):
            data = result.data  # type: ignore[union-attr]
            if isinstance(data, response_model):
                return data
            if isinstance(data, dict):
                return response_model.model_validate(data)
            return None

        # User declined or cancelled
        if config.required:
            raise TimeoutError("User declined or cancelled the custom elicitation request")
        return None

    except AttributeError:
        logger.warning("ctx.elicit() not available.")
        return None


__all__ = [
    "ElicitationAction",
    "ElicitationConfig",
    "ConfirmationResponse",
    "SelectionResponse",
    "MultiSelectResponse",
    "RatingResponse",
    "FeedbackResponse",
    "ReasoningGuidanceResponse",
    "apply_schema_defaults",
    "elicit_confirmation",
    "elicit_selection",
    "elicit_multi_select",
    "elicit_rating",
    "elicit_feedback",
    "elicit_reasoning_guidance",
    "elicit_custom",
]
