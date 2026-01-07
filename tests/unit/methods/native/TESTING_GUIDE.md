# Testing Guide for CodeReasoning Method

This guide provides examples and patterns used in the CodeReasoning test suite.

## Table of Contents
1. [Test Fixtures](#test-fixtures)
2. [Basic Property Tests](#basic-property-tests)
3. [Async Execution Tests](#async-execution-tests)
4. [Configuration Testing](#configuration-testing)
5. [Bug Detection Testing](#bug-detection-testing)
6. [Edge Case Testing](#edge-case-testing)
7. [Continuation Testing](#continuation-testing)

## Test Fixtures

### Method Instance Fixture
```python
@pytest.fixture
def code_method():
    """Create a CodeReasoning instance for testing."""
    return CodeReasoning()
```

### Session Fixture
```python
@pytest.fixture
def session():
    """Create a test session."""
    return Session().start()
```

## Basic Property Tests

### Testing Properties
```python
def test_identifier(self, code_method):
    """Test that identifier returns the correct method identifier."""
    assert code_method.identifier == str(MethodIdentifier.CODE_REASONING)
    assert code_method.identifier == "code_reasoning"

def test_category(self, code_method):
    """Test that category returns HIGH_VALUE."""
    assert code_method.category == str(MethodCategory.HIGH_VALUE)
    assert code_method.category == "high_value"
```

### Testing Initialization
```python
@pytest.mark.asyncio
async def test_initialize(self, code_method):
    """Test initialize method completes without error."""
    await code_method.initialize()
    # Initialization is a no-op but should complete successfully

@pytest.mark.asyncio
async def test_health_check(self, code_method):
    """Test health_check returns True."""
    result = await code_method.health_check()
    assert result is True
```

## Async Execution Tests

### Basic Execution
```python
@pytest.mark.asyncio
async def test_execute_simple_python_code(self, code_method, session):
    """Test executing analysis on simple Python code."""
    code = """
def add(a, b):
    return a + b
"""
    thought = await code_method.execute(session, code)

    assert thought is not None
    assert thought.type == ThoughtType.INITIAL
    assert thought.method_id == MethodIdentifier.CODE_REASONING
    assert isinstance(thought.content, str)
    assert "python" in thought.content.lower()
```

### Testing ThoughtNode Structure
```python
@pytest.mark.asyncio
async def test_execute_returns_thought_node(self, code_method, session):
    """Test execute returns a properly structured ThoughtNode."""
    code = "def hello(): return 'world'"
    thought = await code_method.execute(session, code)

    # Verify ThoughtNode properties
    assert hasattr(thought, 'id')
    assert hasattr(thought, 'type')
    assert hasattr(thought, 'content')
    assert hasattr(thought, 'confidence')
    assert hasattr(thought, 'metadata')

    # Verify metadata
    assert 'language' in thought.metadata
    assert 'has_error' in thought.metadata
    assert 'focus' in thought.metadata
```

## Configuration Testing

### Language Hints
```python
@pytest.mark.asyncio
async def test_language_hint_python(self, code_method, session):
    """Test language hint for Python."""
    code = "print('hello')"
    thought = await code_method.execute(
        session, code, context={"language": "python"}
    )

    assert thought.metadata['language'] == "python"
    assert "python" in thought.content.lower()
```

### Auto-Detection
```python
@pytest.mark.asyncio
async def test_language_auto_detection_javascript(self, code_method, session):
    """Test automatic language detection for JavaScript."""
    code = """
function myFunc() {
    const x = 42;
    return x;
}
"""
    thought = await code_method.execute(session, code)

    assert thought.metadata['language'] == "javascript"
```

### Focus Areas
```python
@pytest.mark.asyncio
async def test_focus_security(self, code_method, session):
    """Test focus option for security."""
    code = "user_input = input()"
    thought = await code_method.execute(
        session, code, context={"focus": "security"}
    )

    assert thought.metadata['focus'] == "security"
    assert "security" in thought.content.lower()
```

### Error Messages
```python
@pytest.mark.asyncio
async def test_error_message_in_context(self, code_method, session):
    """Test providing error message in context."""
    code = "def func(): return undefined_var"
    error = "NameError: name 'undefined_var' is not defined"

    thought = await code_method.execute(
        session, code, context={"error_message": error}
    )

    assert thought.metadata['has_error'] is True
    assert "Error Context" in thought.content
```

## Bug Detection Testing

### Division by Zero
```python
@pytest.mark.asyncio
async def test_detect_division_by_zero(self, code_method, session):
    """Test detection of potential division by zero."""
    code = "result = numerator / 0"
    thought = await code_method.execute(session, code)

    assert "division" in thought.content.lower() or "zero" in thought.content.lower()
```

### Anti-Patterns
```python
@pytest.mark.asyncio
async def test_detect_mutable_default_argument(self, code_method, session):
    """Test detection of mutable default argument anti-pattern."""
    code = """
def append_to_list(item, items=[]):
    items.append(item)
    return items
"""
    thought = await code_method.execute(session, code)

    content_lower = thought.content.lower()
    assert "mutable" in content_lower or "default" in content_lower
```

### Security Issues
```python
@pytest.mark.asyncio
async def test_detect_eval_usage(self, code_method, session):
    """Test detection of eval() usage in Python."""
    code = """
user_input = "print('hello')"
eval(user_input)
"""
    thought = await code_method.execute(session, code, context={"language": "python"})

    content_lower = thought.content.lower()
    assert "eval" in content_lower or "security" in content_lower
```

## Edge Case Testing

### Empty Input
```python
@pytest.mark.asyncio
async def test_empty_code(self, code_method, session):
    """Test analysis of empty code."""
    code = ""
    thought = await code_method.execute(session, code)

    assert thought is not None
    assert isinstance(thought.content, str)
```

### Special Characters
```python
@pytest.mark.asyncio
async def test_special_characters_in_code(self, code_method, session):
    """Test analysis of code with special characters."""
    code = """
def process():
    emoji = "🔥🚀💻"
    unicode = "你好世界"
    return f"{emoji} {unicode}"
"""
    thought = await code_method.execute(session, code)

    assert thought is not None
    assert thought.metadata['language'] == "python"
```

### Pseudocode
```python
@pytest.mark.asyncio
async def test_pseudocode(self, code_method, session):
    """Test analysis of pseudocode."""
    code = """
ALGORITHM Sort(array)
    FOR each element in array
        COMPARE with next element
        IF greater THEN
            SWAP elements
        END IF
    END FOR
END ALGORITHM
"""
    thought = await code_method.execute(session, code)

    assert thought is not None
    assert thought.metadata['language'] == "unknown"
```

## Continuation Testing

### Standard Continuation
```python
@pytest.mark.asyncio
async def test_continue_reasoning_continuation(self, code_method, session):
    """Test continuing reasoning as a continuation."""
    # Create initial thought
    initial_code = "def add(a, b): return a + b"
    initial = await code_method.execute(session, initial_code)

    # Continue reasoning
    continuation = await code_method.continue_reasoning(
        session, initial, guidance="Explore edge cases"
    )

    assert continuation is not None
    assert continuation.type == ThoughtType.CONTINUATION
    assert continuation.parent_id == initial.id
    assert continuation.step_number == initial.step_number + 1
    assert continuation.depth == initial.depth + 1
```

### Branch Creation
```python
@pytest.mark.asyncio
async def test_continue_reasoning_branch(self, code_method, session):
    """Test continuing reasoning as a branch."""
    initial_code = "def process(): pass"
    initial = await code_method.execute(session, initial_code)

    # Create branch
    branch = await code_method.continue_reasoning(
        session, initial, guidance="branch: explore performance"
    )

    assert branch is not None
    assert branch.type == ThoughtType.BRANCH
    assert branch.parent_id == initial.id
    assert branch.branch_id is not None
```

### Focus-Specific Continuation
```python
@pytest.mark.asyncio
async def test_continue_reasoning_focus_performance(self, code_method, session):
    """Test continue_reasoning with performance focus."""
    initial_code = "for i in range(1000): print(i)"
    initial = await code_method.execute(session, initial_code)

    continuation = await code_method.continue_reasoning(
        session, initial, guidance="focus on performance"
    )

    assert continuation is not None
    assert continuation.metadata['focus'] == "performance"
    assert "performance" in continuation.content.lower()
```

## Best Practices

### 1. Clear Test Names
Use descriptive test names that explain what is being tested:
- ✅ `test_detect_division_by_zero`
- ❌ `test_bug1`

### 2. Focused Assertions
Each test should verify one specific behavior:
```python
# Good - focused
assert thought.metadata['language'] == "python"

# Also good - related assertions
assert thought is not None
assert thought.type == ThoughtType.INITIAL
```

### 3. Descriptive Code Samples
Use realistic code samples that clearly demonstrate the test case:
```python
# Good - clear intent
code = """
def divide(a, b):
    return a / b  # Bug: no zero check
"""

# Less clear
code = "x = a / b"
```

### 4. Document Expected Behavior
Include comments explaining non-obvious test logic:
```python
# Should preserve branch_id from initial (which is None)
assert continuation.branch_id == initial.branch_id
```

### 5. Test Both Success and Failure Paths
```python
# Test success case
async def test_no_issues_detected(self, code_method, session):
    code = "def add(a, b): return a + b"
    thought = await code_method.execute(session, code)
    assert thought.quality_score < 0.8

# Test failure case
async def test_detect_division_by_zero(self, code_method, session):
    code = "result = x / 0"
    thought = await code_method.execute(session, code)
    assert "division" in thought.content.lower()
```

## Running Specific Tests

```bash
# Run all tests in a class
pytest tests/unit/methods/native/test_code.py::TestBugDetection -v

# Run a specific test
pytest tests/unit/methods/native/test_code.py::TestBugDetection::test_detect_division_by_zero -v

# Run tests matching a pattern
pytest tests/unit/methods/native/test_code.py -k "language" -v

# Run with verbose output
pytest tests/unit/methods/native/test_code.py -vv

# Run with coverage
pytest tests/unit/methods/native/test_code.py --cov=reasoning_mcp.methods.native.code --cov-report=term-missing
```

## Common Patterns

### Pattern 1: Property Verification
```python
def test_property(self, code_method):
    """Test property returns expected value."""
    assert code_method.property_name == expected_value
```

### Pattern 2: Async Execution
```python
@pytest.mark.asyncio
async def test_execution(self, code_method, session):
    """Test execution produces expected output."""
    result = await code_method.execute(session, input_data)
    assert result.property == expected_value
```

### Pattern 3: Content Verification
```python
@pytest.mark.asyncio
async def test_content(self, code_method, session):
    """Test output contains expected content."""
    result = await code_method.execute(session, input_data)
    assert "expected_text" in result.content.lower()
```

### Pattern 4: Metadata Verification
```python
@pytest.mark.asyncio
async def test_metadata(self, code_method, session):
    """Test metadata contains expected values."""
    result = await code_method.execute(session, input_data)
    assert result.metadata['key'] == expected_value
```
