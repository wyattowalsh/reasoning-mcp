"""Tests for the debug package initialization."""


def test_debug_package_import():
    """Test that the debug package can be imported."""
    from reasoning_mcp import debug

    assert debug is not None


def test_exports():
    """Test that __all__ exports are defined and accessible."""
    from reasoning_mcp import debug

    # Verify __all__ exists and is a list
    assert hasattr(debug, "__all__")
    assert isinstance(debug.__all__, list)

    # Verify all exported names are accessible
    for name in debug.__all__:
        assert hasattr(debug, name), f"Exported name '{name}' not found in debug module"


def test_module_docstring():
    """Test that the module has a proper docstring."""
    from reasoning_mcp import debug

    assert debug.__doc__ is not None
    assert len(debug.__doc__) > 0
    assert "debug" in debug.__doc__.lower()
