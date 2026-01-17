"""Tests for cost estimation module."""

from reasoning_mcp.cost.estimator import METHOD_TOKEN_MULTIPLIERS


def test_method_multipliers():
    """Test that METHOD_TOKEN_MULTIPLIERS exists and has expected keys."""
    # Verify the dictionary exists
    assert METHOD_TOKEN_MULTIPLIERS is not None
    assert isinstance(METHOD_TOKEN_MULTIPLIERS, dict)

    # Expected keys based on the specification
    expected_keys = {
        "chain_of_thought",
        "tree_of_thoughts",
        "react",
        "self_consistency",
        "mcts",
        "graph_of_thoughts",
        "step_back",
        "least_to_most",
        "self_ask",
        "decomposed",
        "sequential",
        "default",
    }

    # Verify all expected keys are present
    actual_keys = set(METHOD_TOKEN_MULTIPLIERS.keys())
    assert expected_keys.issubset(actual_keys), (
        f"Missing expected keys: {expected_keys - actual_keys}"
    )

    # Verify all values are positive floats
    for method, multiplier in METHOD_TOKEN_MULTIPLIERS.items():
        assert isinstance(multiplier, (int, float)), (
            f"Multiplier for {method} is not numeric: {type(multiplier)}"
        )
        assert multiplier > 0, f"Multiplier for {method} must be positive: {multiplier}"

    # Verify specific expected values from the spec
    assert METHOD_TOKEN_MULTIPLIERS["chain_of_thought"] == 2.5
    assert METHOD_TOKEN_MULTIPLIERS["tree_of_thoughts"] == 4.0
    assert METHOD_TOKEN_MULTIPLIERS["react"] == 3.0
    assert METHOD_TOKEN_MULTIPLIERS["self_consistency"] == 5.0
    assert METHOD_TOKEN_MULTIPLIERS["mcts"] == 6.0
    assert METHOD_TOKEN_MULTIPLIERS["graph_of_thoughts"] == 4.5
    assert METHOD_TOKEN_MULTIPLIERS["step_back"] == 2.0
    assert METHOD_TOKEN_MULTIPLIERS["least_to_most"] == 3.0
    assert METHOD_TOKEN_MULTIPLIERS["self_ask"] == 2.5
    assert METHOD_TOKEN_MULTIPLIERS["decomposed"] == 3.5
    assert METHOD_TOKEN_MULTIPLIERS["sequential"] == 1.5
    assert METHOD_TOKEN_MULTIPLIERS["default"] == 2.0


def test_method_multipliers_default_exists():
    """Test that a default multiplier exists for unknown methods."""
    assert "default" in METHOD_TOKEN_MULTIPLIERS
    assert METHOD_TOKEN_MULTIPLIERS["default"] > 0


def test_method_multipliers_ordering():
    """Test that multipliers follow expected ordering based on complexity."""
    # MCTS should have highest multiplier (most complex)
    assert METHOD_TOKEN_MULTIPLIERS["mcts"] >= METHOD_TOKEN_MULTIPLIERS["self_consistency"]

    # Self-consistency should be high (multiple samples)
    assert (
        METHOD_TOKEN_MULTIPLIERS["self_consistency"] > METHOD_TOKEN_MULTIPLIERS["tree_of_thoughts"]
    )

    # Sequential should have lowest multiplier (simplest)
    assert METHOD_TOKEN_MULTIPLIERS["sequential"] < METHOD_TOKEN_MULTIPLIERS["default"]
