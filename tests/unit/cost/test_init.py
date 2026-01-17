"""Tests for cost module initialization."""


def test_exports():
    """Test that the cost module can be imported successfully."""
    import reasoning_mcp.cost

    # Verify module has the expected __all__ attribute
    assert hasattr(reasoning_mcp.cost, "__all__")
    assert isinstance(reasoning_mcp.cost.__all__, list)

    # Verify module docstring exists
    assert reasoning_mcp.cost.__doc__ is not None
    assert "cost prediction" in reasoning_mcp.cost.__doc__.lower()
