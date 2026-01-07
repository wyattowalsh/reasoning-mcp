# Native Methods Unit Tests

This directory contains unit tests for native reasoning methods implemented in the `reasoning_mcp.methods.native` package.

## Test Files

### test_code.py
Comprehensive test suite for the `CodeReasoning` method.

**Coverage**: 66+ test cases across 9 test classes
- Initialization and properties
- Basic execution and analysis
- Code analysis phases (structure, patterns, issues, flow, suggestions)
- Configuration options (language hints, focus areas)
- Continue reasoning (continuations and branches)
- Bug detection (division by zero, undefined variables, anti-patterns)
- Code structure analysis (functions, classes, imports)
- Multi-language support (Python, JavaScript, TypeScript, Java, C++, Rust, Go)
- Edge cases (empty code, pseudocode, incomplete code, special characters)

**Implementation**: `src/reasoning_mcp/methods/native/code.py`

## Running Tests

### Run all native method tests:
```bash
pytest tests/unit/methods/native/ -v
```

### Run specific test file:
```bash
pytest tests/unit/methods/native/test_code.py -v
```

### Run with coverage:
```bash
pytest tests/unit/methods/native/test_code.py --cov=reasoning_mcp.methods.native --cov-report=term-missing
```

### Run specific test class:
```bash
pytest tests/unit/methods/native/test_code.py::TestBugDetection -v
```

### Run specific test:
```bash
pytest tests/unit/methods/native/test_code.py::TestCodeReasoningInitialization::test_identifier -v
```

## Test Structure

All tests follow a consistent pattern:

1. **Fixtures**: Shared test fixtures for method instances and sessions
2. **Test Classes**: Organized by functionality area
3. **Async Tests**: All execution tests are async using `@pytest.mark.asyncio`
4. **Assertions**: Clear, focused assertions on behavior and output
5. **Documentation**: Every test has a clear docstring

### Example Test Structure:
```python
@pytest.fixture
def code_method():
    """Create a CodeReasoning instance for testing."""
    return CodeReasoning()

@pytest.fixture
def session():
    """Create a test session."""
    return Session().start()

class TestSomeFunctionality:
    """Tests for specific functionality."""

    @pytest.mark.asyncio
    async def test_specific_behavior(self, code_method, session):
        """Test description."""
        # Arrange
        input_data = "..."

        # Act
        result = await code_method.execute(session, input_data)

        # Assert
        assert result is not None
        assert expected_condition
```

## Adding New Tests

When adding new native method tests:

1. Create a new test file: `test_<method_name>.py`
2. Follow the existing pattern from `test_code.py`
3. Include fixtures for method instance and session
4. Organize tests into logical classes
5. Test all public methods and properties
6. Include edge cases and error conditions
7. Aim for 90%+ code coverage
8. Add documentation strings to all tests

## Test Dependencies

- `pytest>=8.0.0`
- `pytest-asyncio>=0.24.0` (for async test support)
- `pytest-cov>=5.0.0` (for coverage reporting)

## Coverage Goals

Each test file should aim for:
- **Line Coverage**: 90%+
- **Branch Coverage**: 85%+
- **Function Coverage**: 100%

## See Also

- [Test Coverage Summary](./TEST_COVERAGE_SUMMARY.md) - Detailed coverage breakdown for CodeReasoning tests
- [Base Method Tests](../test_base.py) - Tests for the base ReasoningMethod protocol
