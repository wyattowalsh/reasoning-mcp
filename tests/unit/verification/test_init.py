"""Tests for verification module initialization."""


def test_exports():
    """Test that the verification module can be imported successfully."""
    import reasoning_mcp.verification

    # Verify module has the expected __all__ attribute
    assert hasattr(reasoning_mcp.verification, "__all__")
    assert isinstance(reasoning_mcp.verification.__all__, list)

    # Verify module docstring exists
    assert reasoning_mcp.verification.__doc__ is not None
    assert "verification" in reasoning_mcp.verification.__doc__.lower()
