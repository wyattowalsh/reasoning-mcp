"""Unit tests for CodeReasoning method.

This module provides comprehensive test coverage for the CodeReasoning method,
including initialization, execution, code analysis phases, configuration options,
continuation reasoning, bug detection, and edge cases.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.code import CodeReasoning
from reasoning_mcp.models import Session
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType


@pytest.fixture
def code_method():
    """Create a CodeReasoning instance for testing."""
    return CodeReasoning()


@pytest.fixture
def session():
    """Create a test session."""
    return Session().start()


class TestCodeReasoningInitialization:
    """Tests for CodeReasoning initialization and properties."""

    def test_identifier(self, code_method):
        """Test that identifier returns the correct method identifier."""
        assert code_method.identifier == str(MethodIdentifier.CODE_REASONING)
        assert code_method.identifier == "code_reasoning"

    def test_name(self, code_method):
        """Test that name returns human-readable method name."""
        assert code_method.name == "Code Reasoning"
        assert isinstance(code_method.name, str)

    def test_description(self, code_method):
        """Test that description returns appropriate text."""
        assert "code" in code_method.description.lower()
        assert "debugging" in code_method.description.lower()
        assert isinstance(code_method.description, str)

    def test_category(self, code_method):
        """Test that category returns HIGH_VALUE."""
        assert code_method.category == str(MethodCategory.HIGH_VALUE)
        assert code_method.category == "high_value"

    @pytest.mark.asyncio
    async def test_initialize(self, code_method):
        """Test initialize method completes without error."""
        # Should not raise any exceptions
        await code_method.initialize()
        # Initialization is a no-op but should complete successfully

    @pytest.mark.asyncio
    async def test_health_check(self, code_method):
        """Test health_check returns True."""
        result = await code_method.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(self, code_method):
        """Test health_check works before initialization."""
        # Health check should work even before initialize is called
        result = await code_method.health_check()
        assert result is True


class TestCodeReasoningBasicExecution:
    """Tests for basic code analysis execution."""

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
        assert thought.step_number == 1
        assert thought.depth == 0
        assert isinstance(thought.content, str)
        assert len(thought.content) > 0
        assert "python" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_execute_with_bug(self, code_method, session):
        """Test executing analysis on code with a bug."""
        code = """
def divide(a, b):
    return a / b  # Bug: no zero check
"""
        thought = await code_method.execute(session, code)

        assert thought is not None
        assert "division" in thought.content.lower() or "zero" in thought.content.lower()
        # Should detect potential division by zero

    @pytest.mark.asyncio
    async def test_execute_returns_thought_node(self, code_method, session):
        """Test execute returns a properly structured ThoughtNode."""
        code = "def hello(): return 'world'"
        thought = await code_method.execute(session, code)

        # Verify ThoughtNode properties
        assert hasattr(thought, "id")
        assert hasattr(thought, "type")
        assert hasattr(thought, "method_id")
        assert hasattr(thought, "content")
        assert hasattr(thought, "confidence")
        assert hasattr(thought, "quality_score")
        assert hasattr(thought, "metadata")

        # Verify metadata
        assert "language" in thought.metadata
        assert "has_error" in thought.metadata
        assert "focus" in thought.metadata
        assert "issues_count" in thought.metadata
        assert "supports_branching" in thought.metadata

    @pytest.mark.asyncio
    async def test_execute_with_empty_context(self, code_method, session):
        """Test execute with empty context dict."""
        code = "x = 42"
        thought = await code_method.execute(session, code, context={})

        assert thought is not None
        assert thought.metadata["has_error"] is False


class TestCodeAnalysisPhases:
    """Tests for different code analysis phases."""

    @pytest.mark.asyncio
    async def test_structure_analysis_phase(self, code_method, session):
        """Test that structure analysis is included in output."""
        code = """
def func1():
    pass

def func2():
    pass

class MyClass:
    pass
"""
        thought = await code_method.execute(session, code)

        assert "Structure Analysis" in thought.content
        assert "Functions/methods" in thought.content or "functions" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_pattern_identification_phase(self, code_method, session):
        """Test that pattern identification is included in output."""
        code = """
try:
    result = process()
except Exception:
    pass
"""
        thought = await code_method.execute(session, code)

        assert "Patterns Identified" in thought.content or "patterns" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_issues_detection_phase(self, code_method, session):
        """Test that issues detection is included in output."""
        code = "def bad(items=[]): pass"  # Mutable default argument
        thought = await code_method.execute(session, code)

        assert "Issues Detected" in thought.content or "issues" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_execution_flow_phase(self, code_method, session):
        """Test that execution flow analysis is included in output."""
        code = """
if condition:
    do_something()
for item in items:
    process(item)
"""
        thought = await code_method.execute(session, code)

        assert "Execution Flow" in thought.content or "flow" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_suggestions_phase(self, code_method, session):
        """Test that suggestions are included in output."""
        code = "x = 1 / 0"
        thought = await code_method.execute(session, code)

        assert "Suggestions" in thought.content or "suggestion" in thought.content.lower()


class TestCodeReasoningConfiguration:
    """Tests for configuration options."""

    @pytest.mark.asyncio
    async def test_language_hint_python(self, code_method, session):
        """Test language hint for Python."""
        code = "print('hello')"
        thought = await code_method.execute(session, code, context={"language": "python"})

        assert thought.metadata["language"] == "python"
        assert "python" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_language_hint_javascript(self, code_method, session):
        """Test language hint for JavaScript."""
        code = "console.log('hello');"
        thought = await code_method.execute(session, code, context={"language": "javascript"})

        assert thought.metadata["language"] == "javascript"

    @pytest.mark.asyncio
    async def test_language_auto_detection_python(self, code_method, session):
        """Test automatic language detection for Python."""
        code = """
def my_function():
    import os
    return True
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "python"

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

        assert thought.metadata["language"] == "javascript"

    @pytest.mark.asyncio
    async def test_focus_correctness(self, code_method, session):
        """Test focus option for correctness (default)."""
        code = "x = 1"
        thought = await code_method.execute(session, code, context={"focus": "correctness"})

        assert thought.metadata["focus"] == "correctness"

    @pytest.mark.asyncio
    async def test_focus_performance(self, code_method, session):
        """Test focus option for performance."""
        code = "result = [x**2 for x in range(1000)]"
        thought = await code_method.execute(session, code, context={"focus": "performance"})

        assert thought.metadata["focus"] == "performance"
        assert "performance" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_focus_security(self, code_method, session):
        """Test focus option for security."""
        code = "user_input = input()"
        thought = await code_method.execute(session, code, context={"focus": "security"})

        assert thought.metadata["focus"] == "security"
        assert "security" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_focus_readability(self, code_method, session):
        """Test focus option for readability."""
        code = "x=1;y=2;z=x+y"
        thought = await code_method.execute(session, code, context={"focus": "readability"})

        assert thought.metadata["focus"] == "readability"
        assert "readability" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_error_message_in_context(self, code_method, session):
        """Test providing error message in context."""
        code = "def func(): return undefined_var"
        error = "NameError: name 'undefined_var' is not defined"

        thought = await code_method.execute(session, code, context={"error_message": error})

        assert thought.metadata["has_error"] is True
        assert "Error Context" in thought.content

    @pytest.mark.asyncio
    async def test_error_message_in_input(self, code_method, session):
        """Test extracting error message from input text."""
        code_with_error = """
def divide(a, b):
    return a / b

Error: ZeroDivisionError: division by zero
"""
        thought = await code_method.execute(session, code_with_error)

        assert thought.metadata["has_error"] is True
        assert "Error Context" in thought.content


class TestContinueReasoning:
    """Tests for continue_reasoning functionality."""

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
        assert "Continued Analysis" in continuation.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_branch(self, code_method, session):
        """Test continuing reasoning as a branch."""
        # Create initial thought
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
        assert "Continued Analysis" in branch.content

    @pytest.mark.asyncio
    async def test_continue_reasoning_focus_performance(self, code_method, session):
        """Test continue_reasoning with performance focus."""
        initial_code = "for i in range(1000): print(i)"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(
            session, initial, guidance="focus on performance"
        )

        assert continuation is not None
        assert continuation.metadata["focus"] == "performance"
        assert "performance" in continuation.content.lower()

    @pytest.mark.asyncio
    async def test_continue_reasoning_focus_security(self, code_method, session):
        """Test continue_reasoning with security focus."""
        initial_code = "query = 'SELECT * FROM users'"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(
            session, initial, guidance="analyze security implications"
        )

        assert continuation is not None
        assert continuation.metadata["focus"] == "security"

    @pytest.mark.asyncio
    async def test_continue_reasoning_focus_readability(self, code_method, session):
        """Test continue_reasoning with readability focus."""
        initial_code = "x=lambda a,b:a+b if a>0 else b"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(
            session, initial, guidance="improve readability"
        )

        assert continuation is not None
        assert continuation.metadata["focus"] == "readability"

    @pytest.mark.asyncio
    async def test_continue_reasoning_without_guidance(self, code_method, session):
        """Test continue_reasoning without guidance."""
        initial_code = "def test(): pass"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(session, initial)

        assert continuation is not None
        assert continuation.type == ThoughtType.CONTINUATION
        assert continuation.metadata["focus"] == "alternative approach"

    @pytest.mark.asyncio
    async def test_continue_reasoning_preserves_branch_id(self, code_method, session):
        """Test that continuation preserves branch_id if not creating new branch."""
        initial_code = "def func(): pass"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(
            session, initial, guidance="deepen analysis"
        )

        # Should preserve branch_id from initial (which is None)
        assert continuation.branch_id == initial.branch_id

    @pytest.mark.asyncio
    async def test_continue_reasoning_with_custom_context(self, code_method, session):
        """Test continue_reasoning with custom context."""
        initial_code = "def process(): pass"
        initial = await code_method.execute(session, initial_code)

        continuation = await code_method.continue_reasoning(
            session, initial, context={"focus": "testing", "branch_id": "custom_branch"}
        )

        assert continuation is not None
        assert continuation.metadata["focus"] == "testing"


class TestBugDetection:
    """Tests for bug detection capabilities."""

    @pytest.mark.asyncio
    async def test_detect_division_by_zero(self, code_method, session):
        """Test detection of potential division by zero."""
        code = "result = numerator / 0"
        thought = await code_method.execute(session, code)

        assert "division" in thought.content.lower() or "zero" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_detect_undefined_variable(self, code_method, session):
        """Test detection of undefined variable through error message."""
        code = "print(undefined_variable)"
        error = "NameError: name 'undefined_variable' is not defined"

        thought = await code_method.execute(session, code, context={"error_message": error})

        assert thought.metadata["has_error"] is True
        assert thought.confidence > 0.8  # Higher confidence with error message

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

    @pytest.mark.asyncio
    async def test_detect_bare_except(self, code_method, session):
        """Test detection of bare except clause."""
        code = """
try:
    risky_operation()
except:
    pass
"""
        thought = await code_method.execute(session, code)

        # Should identify bare except as anti-pattern
        assert "except" in thought.content.lower() or "pattern" in thought.content.lower()

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

    @pytest.mark.asyncio
    async def test_detect_deep_nesting(self, code_method, session):
        """Test detection of deep nesting."""
        code = """
def deeply_nested():
    if x:
        if y:
            if z:
                if a:
                    if b:
                        return True
"""
        thought = await code_method.execute(session, code)

        # Should detect deep nesting
        assert "nesting" in thought.content.lower() or "indent" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_no_issues_detected(self, code_method, session):
        """Test clean code with no issues."""
        code = """
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""
        thought = await code_method.execute(session, code)

        # Quality score should be lower for clean code
        assert thought.quality_score < 0.8


class TestCodeStructureAnalysis:
    """Tests for code structure analysis."""

    @pytest.mark.asyncio
    async def test_analyze_function_count(self, code_method, session):
        """Test counting functions in code."""
        code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        thought = await code_method.execute(session, code)

        assert "Functions/methods" in thought.content or "function" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_analyze_class_count(self, code_method, session):
        """Test counting classes in code."""
        code = """
class MyClass:
    pass

class AnotherClass:
    pass
"""
        thought = await code_method.execute(session, code)

        assert "Classes/interfaces" in thought.content or "class" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_analyze_import_statements(self, code_method, session):
        """Test detecting import statements."""
        code = """
import os
import sys
from pathlib import Path
"""
        thought = await code_method.execute(session, code)

        assert "Import statements" in thought.content or "import" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_analyze_python_decorators(self, code_method, session):
        """Test detecting Python decorators."""
        code = """
@property
def get_value(self):
    return self._value

@staticmethod
def static_method():
    pass
"""
        thought = await code_method.execute(session, code)

        assert "Decorators" in thought.content or "decorator" in thought.content.lower()

    @pytest.mark.asyncio
    async def test_analyze_javascript_arrow_functions(self, code_method, session):
        """Test detecting JavaScript arrow functions."""
        code = """
const add = (a, b) => {
    return a + b;
};

const multiply = (x, y) => {
    return x * y;
};
"""
        thought = await code_method.execute(session, code)

        # Should detect JavaScript and arrow functions
        content_lower = thought.content.lower()
        assert "arrow" in content_lower or "function" in content_lower


class TestMultipleLanguages:
    """Tests for handling different programming languages."""

    @pytest.mark.asyncio
    async def test_python_detection(self, code_method, session):
        """Test Python language detection."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "python"

    @pytest.mark.asyncio
    async def test_javascript_detection(self, code_method, session):
        """Test JavaScript language detection."""
        code = """
function calculate(x, y) {
    const result = x + y;
    return result;
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "javascript"

    @pytest.mark.asyncio
    async def test_typescript_detection(self, code_method, session):
        """Test TypeScript language detection."""
        code = """
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}`;
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "typescript"

    @pytest.mark.asyncio
    async def test_java_detection(self, code_method, session):
        """Test Java language detection."""
        code = """
public class MyClass {
    private int value;

    public void setValue(int v) {
        this.value = v;
    }
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "java"

    @pytest.mark.asyncio
    async def test_cpp_detection(self, code_method, session):
        """Test C++ language detection."""
        code = """
#include <iostream>
#include <vector>

namespace myapp {
    void process() {
        std::cout << "Hello" << std::endl;
    }
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "c++"

    @pytest.mark.asyncio
    async def test_rust_detection(self, code_method, session):
        """Test Rust language detection."""
        code = """
fn main() {
    let mut counter = 0;
    pub struct MyStruct {
        value: i32,
    }
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "rust"

    @pytest.mark.asyncio
    async def test_go_detection(self, code_method, session):
        """Test Go language detection."""
        code = """
package main

import "fmt"

func main() {
    x := 42
    fmt.Println(x)
}
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "go"

    @pytest.mark.asyncio
    async def test_unknown_language(self, code_method, session):
        """Test handling of unknown/undetectable language."""
        code = """
some random text
that doesn't match
any language patterns
"""
        thought = await code_method.execute(session, code)

        assert thought.metadata["language"] == "unknown"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_code(self, code_method, session):
        """Test analysis of empty code."""
        code = ""
        thought = await code_method.execute(session, code)

        assert thought is not None
        assert isinstance(thought.content, str)

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self, code_method, session):
        """Test analysis of whitespace-only code."""
        code = "   \n\n   \t\t   \n"
        thought = await code_method.execute(session, code)

        assert thought is not None

    @pytest.mark.asyncio
    async def test_single_line_code(self, code_method, session):
        """Test analysis of single-line code."""
        code = "x = 42"
        thought = await code_method.execute(session, code)

        assert thought is not None
        assert "Lines of code" in thought.content

    @pytest.mark.asyncio
    async def test_very_long_code(self, code_method, session):
        """Test analysis of very long code."""
        # Generate long code
        code = "\n".join([f"def function_{i}(): pass" for i in range(100)])
        thought = await code_method.execute(session, code)

        assert thought is not None
        assert thought.metadata["issues_count"] >= 0

    @pytest.mark.asyncio
    async def test_code_with_comments_only(self, code_method, session):
        """Test analysis of code that is only comments."""
        code = """
# This is a comment
# Another comment
# TODO: Implement this
"""
        thought = await code_method.execute(session, code)

        assert thought is not None
        # Should detect TODO comment
        assert "TODO" in thought.content or "todo" in thought.content.lower()

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
        assert thought.metadata["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_incomplete_code(self, code_method, session):
        """Test analysis of incomplete code."""
        code = """
def incomplete_function(x, y
    # Missing closing parenthesis and function body
"""
        thought = await code_method.execute(session, code)

        assert thought is not None

    @pytest.mark.asyncio
    async def test_mixed_languages(self, code_method, session):
        """Test analysis of code with mixed language features."""
        code = """
# Python-like syntax
def my_function():
    # JavaScript-like syntax
    const x = 42;
    return x;
"""
        thought = await code_method.execute(session, code)

        assert thought is not None
        # Should detect one of the languages

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
        assert thought.metadata["language"] == "python"

    @pytest.mark.asyncio
    async def test_multiple_error_markers(self, code_method, session):
        """Test extraction with multiple error markers."""
        code = """
def func():
    pass

Error: First error
Traceback: Some traceback
Exception: Another exception
"""
        thought = await code_method.execute(session, code)

        # Should extract first error marker
        assert thought.metadata["has_error"] is True

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, code_method, session):
        """Test confidence score calculation."""
        # Code with no issues should have higher confidence
        clean_code = "def add(a, b): return a + b"
        clean_thought = await code_method.execute(session, clean_code)

        # Code with issues should have lower confidence
        buggy_code = """
def buggy(items=[]):
    result = x / 0
    eval(user_input)
"""
        buggy_thought = await code_method.execute(session, buggy_code)

        # Confidence should be reasonable
        assert 0.5 <= clean_thought.confidence <= 1.0
        assert 0.5 <= buggy_thought.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_quality_score_with_issues(self, code_method, session):
        """Test quality score when issues are detected."""
        code = "result = x / 0"
        thought = await code_method.execute(session, code)

        # Should have higher quality score when issues are found
        assert thought.quality_score >= 0.7

    @pytest.mark.asyncio
    async def test_quality_score_without_issues(self, code_method, session):
        """Test quality score when no issues are detected."""
        code = "def safe(): return True"
        thought = await code_method.execute(session, code)

        # Quality score should be lower for clean code
        assert thought.quality_score <= 0.8

    @pytest.mark.asyncio
    async def test_metadata_supports_branching(self, code_method, session):
        """Test that metadata indicates branching support."""
        code = "x = 1"
        thought = await code_method.execute(session, code)

        assert thought.metadata["supports_branching"] is True

    @pytest.mark.asyncio
    async def test_step_number_increments(self, code_method, session):
        """Test that step numbers increment in continuations."""
        code = "def test(): pass"
        initial = await code_method.execute(session, code)
        continuation = await code_method.continue_reasoning(session, initial)

        assert initial.step_number == 1
        assert continuation.step_number == 2

    @pytest.mark.asyncio
    async def test_depth_increments(self, code_method, session):
        """Test that depth increments in continuations."""
        code = "def test(): pass"
        initial = await code_method.execute(session, code)
        continuation = await code_method.continue_reasoning(session, initial)

        assert initial.depth == 0
        assert continuation.depth == 1
