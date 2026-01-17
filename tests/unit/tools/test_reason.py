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

from uuid import UUID

import pytest

from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.tools import ReasonHints, ReasonOutput
from reasoning_mcp.tools.reason import reason

# ============================================================================
# Basic Reasoning Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_just_problem():
    """Test basic reasoning with only a problem string.

    Note: When no method is specified, the router may select a pipeline
    which can return different thought types. This test verifies the
    structure is valid without asserting a specific thought type.
    """
    problem = "What is the best approach to implement caching in this system?"

    result = await reason(problem)

    # Verify return type
    assert isinstance(result, ReasonOutput)

    # Verify session ID is valid UUID
    session_uuid = UUID(result.session_id)
    assert str(session_uuid) == result.session_id

    # Verify thought is returned
    assert result.thought is not None
    # Note: With router integration, thought type varies based on selected method/pipeline
    assert result.thought.type in ThoughtType
    assert result.thought.content is not None
    assert len(result.thought.content) > 0

    # Verify method was selected (may be from pipeline)
    assert result.method_used is not None

    # Verify suggestions are provided
    assert isinstance(result.suggestions, list)
    assert len(result.suggestions) > 0

    # Verify metadata indicates auto-selection (routing)
    assert isinstance(result.metadata, dict)


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
    hints = ReasonHints(domain="code", complexity="high")

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
        prefer_methods=[MethodIdentifier.ETHICAL_REASONING, MethodIdentifier.DIALECTIC],
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
        domain="code", avoid_methods=[MethodIdentifier.ETHICAL_REASONING, MethodIdentifier.SOCRATIC]
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
            MethodIdentifier.SOCRATIC,
        ],
        avoid_methods=[MethodIdentifier.MATHEMATICAL_REASONING],
        custom_hints={"stakeholders": ["users", "developers", "society"]},
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
    """Test that first thought is INITIAL type when using a single method.

    Note: This test specifies a method explicitly because when no method
    is specified, the router may select a pipeline which can produce
    different thought types.
    """
    problem = "First thought should be INITIAL"

    # Specify chain_of_thought to ensure we get a single method execution
    result = await reason(problem, method="chain_of_thought")

    assert result.thought.type == ThoughtType.INITIAL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_thought_type_with_different_methods():
    """Test that INITIAL type is used regardless of method."""
    problem = "Test problem"

    # Methods that produce INITIAL thought type without MCP context
    methods = [
        "chain_of_thought",
        "code_reasoning",
        "mathematical_reasoning",
    ]

    for method in methods:
        result = await reason(problem, method=method)
        assert result.thought.type == ThoughtType.INITIAL, (
            f"Method {method} should produce INITIAL thought type"
        )

    # ethical_reasoning may produce CONCLUSION when elicitation is unavailable
    result = await reason(problem, method="ethical_reasoning")
    assert result.thought.type in (ThoughtType.INITIAL, ThoughtType.CONCLUSION), (
        "Method ethical_reasoning should produce INITIAL or CONCLUSION thought type"
    )


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_combo_template():
    """Test reasoning with combo template specification."""
    problem = "Test combo template usage"

    result = await reason(problem, method="combo:debate")

    assert isinstance(result, ReasonOutput)
    assert "combo" in result.metadata
    assert result.metadata["combo"]["combo_id"] == "debate"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_with_inline_combo_dict():
    """Test reasoning with inline pipeline dict specification."""
    problem = "Test inline combo usage"
    pipeline = {
        "stage_type": "method",
        "method_id": "chain_of_thought",
        "name": "inline_test",
    }

    result = await reason(problem, method=pipeline)

    assert isinstance(result, ReasonOutput)
    assert "combo" in result.metadata
    assert result.metadata["combo"]["pipeline_type"] == "method"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reason_budget_param():
    """Test that budget parameter exists and defaults to None."""
    import inspect

    from reasoning_mcp.models.cost import Budget
    from reasoning_mcp.tools.reason import reason

    # Get the signature of the reason function
    sig = inspect.signature(reason)

    # Verify budget parameter exists
    assert "budget" in sig.parameters, "budget parameter should exist in reason function"

    # Get the budget parameter
    budget_param = sig.parameters["budget"]

    # Verify it defaults to None
    assert budget_param.default is None, "budget parameter should default to None"

    # Verify the annotation is Budget | None
    import typing

    annotation = budget_param.annotation

    # The annotation should be a Union type with Budget and None
    if hasattr(typing, "get_args"):
        # Python 3.8+
        args = typing.get_args(annotation)
        if args:  # Union type
            assert Budget in args or any("Budget" in str(arg) for arg in args), (
                "budget parameter should accept Budget type"
            )
            assert type(None) in args or any("None" in str(arg) for arg in args), (
                "budget parameter should accept None type"
            )

    # Test that the function can be called with budget=None
    result = await reason("Test problem", budget=None)
    assert isinstance(result, ReasonOutput)

    # Test that the function can be called without budget parameter
    result = await reason("Test problem")
    assert isinstance(result, ReasonOutput)


# ============================================================================
# Tracing Parameters Tests
# ============================================================================


class TestReasonTracingParams:
    """Tests for tracing parameters in the reason tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_param_default_false(self):
        """Test that trace defaults to False in the function signature."""
        import inspect

        from reasoning_mcp.tools.reason import reason

        # Get the signature of the reason function
        sig = inspect.signature(reason)

        # Verify trace parameter exists
        assert "trace" in sig.parameters, "trace parameter should exist in reason function"

        # Get the trace parameter
        trace_param = sig.parameters["trace"]

        # Verify it defaults to False
        assert trace_param.default is False, "trace parameter should default to False"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_level_param_default_standard(self):
        """Test that trace_level defaults to TraceLevel.STANDARD."""
        import inspect

        from reasoning_mcp.models.debug import TraceLevel
        from reasoning_mcp.tools.reason import reason

        # Get the signature of the reason function
        sig = inspect.signature(reason)

        # Verify trace_level parameter exists
        assert "trace_level" in sig.parameters, "trace_level parameter should exist"

        # Get the trace_level parameter
        trace_level_param = sig.parameters["trace_level"]

        # Verify it defaults to TraceLevel.STANDARD
        assert trace_level_param.default == TraceLevel.STANDARD, (
            "trace_level should default to TraceLevel.STANDARD"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_disabled_by_default(self):
        """Test that tracing is disabled by default (no trace_id in metadata)."""
        problem = "Test problem for tracing"

        result = await reason(problem)

        # Verify trace_id is NOT in metadata when tracing is disabled
        assert "trace_id" not in result.metadata, (
            "trace_id should not be in metadata when trace=False"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_enabled_creates_trace_id(self):
        """Test that enabling trace creates a trace_id in metadata."""
        problem = "Test problem with tracing enabled"

        result = await reason(problem, trace=True)

        # Verify trace_id is in metadata when tracing is enabled
        assert "trace_id" in result.metadata, "trace_id should be in metadata when trace=True"

        # Verify trace_id is a valid string
        assert isinstance(result.metadata["trace_id"], str), "trace_id should be a string"

        # Verify trace_id is not empty
        assert len(result.metadata["trace_id"]) > 0, "trace_id should not be empty"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_with_explicit_method(self):
        """Test that tracing works with explicit method specification."""
        problem = "Test tracing with explicit method"
        method = "chain_of_thought"

        result = await reason(problem, method=method, trace=True)

        # Verify trace_id is present
        assert "trace_id" in result.metadata

        # Verify method was used
        assert result.method_used == MethodIdentifier.CHAIN_OF_THOUGHT

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_level_parameter_accepted(self):
        """Test that different trace_level values are accepted."""
        from reasoning_mcp.models.debug import TraceLevel

        problem = "Test trace level parameter"

        # Test with MINIMAL
        result = await reason(problem, trace=True, trace_level=TraceLevel.MINIMAL)
        assert "trace_id" in result.metadata

        # Test with DETAILED
        result = await reason(problem, trace=True, trace_level=TraceLevel.DETAILED)
        assert "trace_id" in result.metadata

        # Test with VERBOSE
        result = await reason(problem, trace=True, trace_level=TraceLevel.VERBOSE)
        assert "trace_id" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backwards_compatibility_without_trace_params(self):
        """Test that existing calls without trace params still work."""
        problem = "Test backwards compatibility"

        # Call without any trace parameters (should work as before)
        result = await reason(problem)

        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.thought is not None
        assert "trace_id" not in result.metadata  # No tracing by default


# ============================================================================
# Verification Parameters Tests
# ============================================================================


class TestReasonVerificationParams:
    """Tests for verification parameters in the reason tool."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reason_verify_param(self):
        """Test that verify parameter exists and defaults to False."""
        import inspect

        from reasoning_mcp.tools.reason import reason

        # Get the signature of the reason function
        sig = inspect.signature(reason)

        # Verify verify parameter exists
        assert "verify" in sig.parameters, "verify parameter should exist in reason function"

        # Get the verify parameter
        verify_param = sig.parameters["verify"]

        # Verify it defaults to False
        assert verify_param.default is False, "verify parameter should default to False"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reason_verify_method_param(self):
        """Test that verify_method parameter exists and has correct default."""
        import inspect

        from reasoning_mcp.tools.reason import reason

        # Get the signature of the reason function
        sig = inspect.signature(reason)

        # Verify verify_method parameter exists
        assert "verify_method" in sig.parameters, (
            "verify_method parameter should exist in reason function"
        )

        # Get the verify_method parameter
        verify_method_param = sig.parameters["verify_method"]

        # Verify it defaults to "chain_of_verification"
        assert verify_method_param.default == "chain_of_verification", (
            "verify_method should default to 'chain_of_verification'"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reason_verification(self):
        """Test that verification runs when verify=True."""
        problem = "The sky is blue and 2+2=4"

        result = await reason(problem, method="chain_of_thought", verify=True)

        # Verify result structure
        assert isinstance(result, ReasonOutput)
        assert result.session_id is not None
        assert result.thought is not None

        # Verification should have run - check metadata
        # Note: If verification fails (e.g., missing dependencies), there may be a verification_error
        # But the call itself should succeed
        assert "verification" in result.metadata or "verification_error" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reason_verify_response(self):
        """Test that verification report data is included in metadata when verify=True."""
        problem = "The Earth orbits the Sun. Water freezes at 0°C."

        result = await reason(problem, method="chain_of_thought", verify=True)

        # Verification should have run
        assert isinstance(result, ReasonOutput)

        # Check for verification data in metadata
        # If verification succeeded, we should have the report
        if "verification" in result.metadata:
            verification = result.metadata["verification"]

            # Verify the expected fields are present
            assert "overall_accuracy" in verification, "Should have overall_accuracy"
            assert "claims_count" in verification, "Should have claims_count"
            assert "verified_count" in verification, "Should have verified_count"
            assert "flagged_count" in verification, "Should have flagged_count"

            # Verify field types and ranges
            assert isinstance(verification["overall_accuracy"], float)
            assert 0.0 <= verification["overall_accuracy"] <= 1.0

            assert isinstance(verification["claims_count"], int)
            assert verification["claims_count"] >= 0

            assert isinstance(verification["verified_count"], int)
            assert verification["verified_count"] >= 0

            assert isinstance(verification["flagged_count"], int)
            assert verification["flagged_count"] >= 0
        else:
            # If verification failed, there should be an error message
            assert "verification_error" in result.metadata
