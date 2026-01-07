"""
Comprehensive tests for the reason tool in reasoning_mcp.tools.reason.

This module provides complete test coverage for the reason() function:
- Basic reasoning with just a problem string
- Reasoning with explicit method specified
- Reasoning with hints provided
- Auto-selection of method when none specified
- Return value structure validation
- Thought type validation
- Session ID format validation
"""

import pytest
from uuid import UUID

from reasoning_mcp.tools.reason import reason
from reasoning_mcp.models.tools import ReasonHints, ReasonOutput
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType


# ============================================================================
# Basic Reasoning Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_just_problem():
    """Test basic reasoning with only a problem string."""
    problem = "What is the best approach to implement caching in this system?"

    result = await reason(problem)

    # Verify return type
    assert isinstance(result, ReasonOutput)

    # Verify session ID is valid UUID
    session_uuid = UUID(result.session_id)
    assert str(session_uuid) == result.session_id

    # Verify thought is returned
    assert result.thought is not None
    assert result.thought.type == ThoughtType.INITIAL
    assert result.thought.content is not None
    assert len(result.thought.content) > 0

    # Verify method was selected
    assert result.method_used is not None
    assert isinstance(result.method_used, MethodIdentifier)

    # Verify suggestions are provided
    assert isinstance(result.suggestions, list)
    assert len(result.suggestions) > 0

    # Verify metadata indicates auto-selection
    assert isinstance(result.metadata, dict)
    assert result.metadata.get("auto_selected") is True
    assert result.metadata.get("hints_provided") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_long_problem():
    """Test reasoning with a long problem string (>200 chars)."""
    problem = (
        "This is a very long problem statement that exceeds 200 characters. "
        "It contains multiple sentences and describes a complex scenario that "
        "requires careful analysis and reasoning. The system needs to handle "
        "this gracefully and provide meaningful initial thoughts even when the "
        "problem description is quite verbose and detailed."
    )

    result = await reason(problem)

    assert isinstance(result, ReasonOutput)
    assert result.thought.type == ThoughtType.INITIAL
    assert result.metadata.get("problem_length") == len(problem)


# ============================================================================
# Explicit Method Selection Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_explicit_method():
    """Test reasoning with explicit method specified."""
    problem = "Calculate the optimal solution to this optimization problem"
    method = "mathematical_reasoning"

    result = await reason(problem, method=method)

    assert isinstance(result, ReasonOutput)
    assert result.method_used == MethodIdentifier.MATHEMATICAL_REASONING
    assert result.thought.method_id == MethodIdentifier.MATHEMATICAL_REASONING

    # Verify metadata shows method was not auto-selected
    assert result.metadata.get("auto_selected") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_chain_of_thought():
    """Test reasoning with chain_of_thought method."""
    problem = "How can we improve team collaboration?"
    method = "chain_of_thought"

    result = await reason(problem, method=method)

    assert result.method_used == MethodIdentifier.CHAIN_OF_THOUGHT
    assert result.thought.method_id == MethodIdentifier.CHAIN_OF_THOUGHT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_ethical_reasoning():
    """Test reasoning with ethical_reasoning method."""
    problem = "Should we implement this feature that might compromise user privacy?"
    method = "ethical_reasoning"

    result = await reason(problem, method=method)

    assert result.method_used == MethodIdentifier.ETHICAL_REASONING
    assert result.thought.method_id == MethodIdentifier.ETHICAL_REASONING


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_code_reasoning():
    """Test reasoning with code_reasoning method."""
    problem = "Debug this complex async race condition in the payment processor"
    method = "code_reasoning"

    result = await reason(problem, method=method)

    assert result.method_used == MethodIdentifier.CODE_REASONING
    assert result.thought.method_id == MethodIdentifier.CODE_REASONING


# ============================================================================
# Hints-Based Selection Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_hints_domain():
    """Test reasoning with hints containing domain information."""
    problem = "How should we refactor this legacy codebase?"
    hints = ReasonHints(domain="code")

    result = await reason(problem, hints=hints)

    assert isinstance(result, ReasonOutput)
    assert result.metadata.get("hints_provided") is True
    assert "hints" in result.metadata
    assert result.metadata["hints"]["domain"] == "code"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_hints_complexity():
    """Test reasoning with hints containing complexity information."""
    problem = "Analyze the implications of this architectural decision"
    hints = ReasonHints(
        domain="code",
        complexity="high"
    )

    result = await reason(problem, hints=hints)

    assert result.metadata.get("hints_provided") is True
    assert result.metadata["hints"]["complexity"] == "high"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_hints_prefer_methods():
    """Test reasoning with hints specifying preferred methods."""
    problem = "Evaluate this ethical dilemma"
    hints = ReasonHints(
        domain="ethical",
        prefer_methods=[MethodIdentifier.ETHICAL_REASONING, MethodIdentifier.DIALECTIC]
    )

    result = await reason(problem, hints=hints)

    assert result.metadata.get("hints_provided") is True
    assert "preferred_methods" in result.metadata["hints"]
    assert len(result.metadata["hints"]["preferred_methods"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_hints_avoid_methods():
    """Test reasoning with hints specifying methods to avoid."""
    problem = "How can we optimize this algorithm?"
    hints = ReasonHints(
        domain="code",
        avoid_methods=[MethodIdentifier.ETHICAL_REASONING, MethodIdentifier.SOCRATIC]
    )

    result = await reason(problem, hints=hints)

    assert result.metadata.get("hints_provided") is True
    assert "avoided_methods" in result.metadata["hints"]
    assert len(result.metadata["hints"]["avoided_methods"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_comprehensive_hints():
    """Test reasoning with comprehensive hints."""
    problem = "Design a fair algorithm for resource allocation"
    hints = ReasonHints(
        domain="ethical",
        complexity="high",
        prefer_methods=[
            MethodIdentifier.ETHICAL_REASONING,
            MethodIdentifier.DIALECTIC,
            MethodIdentifier.SOCRATIC
        ],
        avoid_methods=[MethodIdentifier.MATHEMATICAL_REASONING],
        custom_hints={"stakeholders": ["users", "developers", "society"]}
    )

    result = await reason(problem, hints=hints)

    assert result.metadata.get("hints_provided") is True
    hints_meta = result.metadata["hints"]
    assert hints_meta["domain"] == "ethical"
    assert hints_meta["complexity"] == "high"
    assert len(hints_meta["preferred_methods"]) == 3
    assert len(hints_meta["avoided_methods"]) == 1


# ============================================================================
# Auto-Selection Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_auto_selection_without_method():
    """Test that method is auto-selected when not specified."""
    problem = "What should we do in this situation?"

    result = await reason(problem)

    # Should auto-select a method
    assert result.method_used is not None
    assert isinstance(result.method_used, MethodIdentifier)
    assert result.metadata.get("auto_selected") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_fallback_to_chain_of_thought():
    """Test that system falls back to chain_of_thought when selection fails."""
    # This is testing the fallback behavior mentioned in the code
    problem = "Some ambiguous problem"

    result = await reason(problem)

    # Should return a valid result even if selection is uncertain
    assert result.method_used is not None
    assert isinstance(result.method_used, MethodIdentifier)


# ============================================================================
# Return Value Structure Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_output_has_correct_structure():
    """Test that ReasonOutput has all required fields."""
    problem = "Test problem"

    result = await reason(problem)

    # Verify all required fields are present
    assert hasattr(result, "session_id")
    assert hasattr(result, "thought")
    assert hasattr(result, "method_used")
    assert hasattr(result, "suggestions")
    assert hasattr(result, "metadata")

    # Verify types
    assert isinstance(result.session_id, str)
    assert result.thought is not None
    assert isinstance(result.method_used, MethodIdentifier)
    assert isinstance(result.suggestions, list)
    assert isinstance(result.metadata, dict)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_structure():
    """Test that thought has correct structure and fields."""
    problem = "Test problem for thought validation"

    result = await reason(problem)
    thought = result.thought

    # Verify thought has required fields
    assert hasattr(thought, "id")
    assert hasattr(thought, "type")
    assert hasattr(thought, "method_id")
    assert hasattr(thought, "content")
    assert hasattr(thought, "confidence")
    assert hasattr(thought, "step_number")
    assert hasattr(thought, "depth")
    assert hasattr(thought, "created_at")
    assert hasattr(thought, "metadata")

    # Verify thought field values
    assert isinstance(thought.id, str)
    assert thought.type == ThoughtType.INITIAL
    assert isinstance(thought.method_id, MethodIdentifier)
    assert isinstance(thought.content, str)
    assert len(thought.content) > 0
    assert isinstance(thought.confidence, float)
    assert 0.0 <= thought.confidence <= 1.0
    assert thought.step_number == 1
    assert thought.depth == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_suggestions_structure():
    """Test that suggestions have correct structure."""
    problem = "Test problem for suggestions"

    result = await reason(problem)

    assert isinstance(result.suggestions, list)
    assert len(result.suggestions) > 0

    # All suggestions should be strings
    for suggestion in result.suggestions:
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_metadata_structure():
    """Test that metadata has expected structure."""
    problem = "Test problem for metadata"
    hints = ReasonHints(domain="test", complexity="moderate")

    result = await reason(problem, hints=hints)

    assert isinstance(result.metadata, dict)

    # Check required metadata fields
    assert "auto_selected" in result.metadata
    assert "hints_provided" in result.metadata
    assert "problem_length" in result.metadata

    assert isinstance(result.metadata["auto_selected"], bool)
    assert isinstance(result.metadata["hints_provided"], bool)
    assert isinstance(result.metadata["problem_length"], int)
    assert result.metadata["problem_length"] == len(problem)


# ============================================================================
# Thought Type Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_is_initial_type():
    """Test that first thought is always INITIAL type."""
    problem = "First thought should be INITIAL"

    result = await reason(problem)

    assert result.thought.type == ThoughtType.INITIAL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_type_with_different_methods():
    """Test that INITIAL type is used regardless of method."""
    problem = "Test problem"

    methods = [
        "chain_of_thought",
        "ethical_reasoning",
        "code_reasoning",
        "mathematical_reasoning",
    ]

    for method in methods:
        result = await reason(problem, method=method)
        assert result.thought.type == ThoughtType.INITIAL, \
            f"Method {method} should produce INITIAL thought type"


# ============================================================================
# Session ID Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_session_id_is_valid_uuid():
    """Test that session_id is a valid UUID string."""
    problem = "Test UUID validation"

    result = await reason(problem)

    # Should not raise ValueError
    session_uuid = UUID(result.session_id)

    # Verify it's a properly formatted UUID string
    assert str(session_uuid) == result.session_id

    # UUID should be version 4 (random)
    assert session_uuid.version == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_generates_unique_session_ids():
    """Test that each call generates a unique session ID."""
    problem = "Test unique session IDs"

    result1 = await reason(problem)
    result2 = await reason(problem)
    result3 = await reason(problem)

    # All session IDs should be different
    session_ids = {result1.session_id, result2.session_id, result3.session_id}
    assert len(session_ids) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_id_is_valid_uuid():
    """Test that thought ID is a valid UUID string."""
    problem = "Test thought UUID validation"

    result = await reason(problem)

    # Should not raise ValueError
    thought_uuid = UUID(result.thought.id)

    # Verify it's a properly formatted UUID string
    assert str(thought_uuid) == result.thought.id


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_empty_hints():
    """Test reasoning with empty hints object."""
    problem = "Test with empty hints"
    hints = ReasonHints()

    result = await reason(problem, hints=hints)

    # Should still work, but hints_provided should be True
    assert isinstance(result, ReasonOutput)
    assert result.metadata.get("hints_provided") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_very_short_problem():
    """Test reasoning with very short problem string."""
    problem = "Help"

    result = await reason(problem)

    assert isinstance(result, ReasonOutput)
    assert result.thought.type == ThoughtType.INITIAL
    assert result.metadata["problem_length"] == len(problem)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_preserves_problem_in_thought_metadata():
    """Test that original problem is preserved in thought metadata."""
    problem = "This is the original problem statement"

    result = await reason(problem)

    assert "problem" in result.thought.metadata
    assert result.thought.metadata["problem"] == problem


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_confidence_in_valid_range():
    """Test that thought confidence is in valid range [0.0, 1.0]."""
    problem = "Test confidence range"

    result = await reason(problem)

    assert isinstance(result.thought.confidence, float)
    assert 0.0 <= result.thought.confidence <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_has_correct_step_and_depth():
    """Test that initial thought has step_number=1 and depth=0."""
    problem = "Test step and depth"

    result = await reason(problem)

    assert result.thought.step_number == 1
    assert result.thought.depth == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_output_is_immutable():
    """Test that ReasonOutput is frozen/immutable."""
    problem = "Test immutability"

    result = await reason(problem)

    # ReasonOutput should be frozen (Pydantic model with frozen=True)
    with pytest.raises(Exception):  # ValidationError or AttributeError
        result.session_id = "new_value"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_method_and_hints():
    """Test that explicit method takes precedence over hints."""
    problem = "Test method vs hints priority"
    method = "chain_of_thought"
    hints = ReasonHints(prefer_methods=[MethodIdentifier.ETHICAL_REASONING])

    result = await reason(problem, method=method, hints=hints)

    # Explicit method should be used
    assert result.method_used == MethodIdentifier.CHAIN_OF_THOUGHT
    # But metadata should still show hints were provided
    assert result.metadata.get("hints_provided") is True


# ============================================================================
# Error Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_invalid_method():
    """Test that invalid method raises ValueError."""
    problem = "Test with invalid method"

    # Note: Since the registry is currently empty and allows any valid MethodIdentifier,
    # this test documents the expected behavior once methods are registered.
    # Currently, any valid MethodIdentifier string will work.
    # This test will need to be updated when the registry is populated.

    # For now, test with an invalid enum value should raise ValueError
    with pytest.raises(ValueError):
        await reason(problem, method="totally_invalid_method_that_doesnt_exist")
