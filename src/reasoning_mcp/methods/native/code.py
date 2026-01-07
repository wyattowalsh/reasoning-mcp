"""Code reasoning method implementation.

This module provides specialized reasoning for code analysis, debugging, and development.
It analyzes code structure, identifies patterns and potential issues, traces execution flow,
and suggests fixes or improvements.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session

from reasoning_mcp.methods.base import MethodMetadata


class CodeReasoning:
    """Code-specialized reasoning method for analysis and debugging.

    This method implements structured code reasoning with the following phases:
    1. Code structure analysis - Understand the code organization
    2. Pattern identification - Detect common patterns and anti-patterns
    3. Issue detection - Find potential bugs, errors, or problems
    4. Execution tracing - Follow the logical flow (conceptual)
    5. Fix suggestion - Propose improvements or fixes

    The method supports branching for exploring different fix approaches and has
    medium-high complexity (5-7) suitable for various code analysis scenarios.

    Examples:
        >>> method = CodeReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> thought = await method.execute(
        ...     session,
        ...     "def add(a, b):\\n    return a + b + c  # Bug: undefined 'c'"
        ... )
        >>> assert thought.type == ThoughtType.INITIAL
    """

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.CODE_REASONING)

    @property
    def name(self) -> str:
        """Return human-readable method name."""
        return "Code Reasoning"

    @property
    def description(self) -> str:
        """Return method description."""
        return "Specialized reasoning for code analysis, debugging, and development"

    @property
    def category(self) -> str:
        """Return method category."""
        return str(MethodCategory.HIGH_VALUE)

    async def initialize(self) -> None:
        """Initialize the code reasoning method.

        Performs any necessary setup. For code reasoning, this is primarily
        a no-op but follows the protocol requirement.
        """
        # No initialization required for this method
        pass

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute code reasoning on the input.

        This method analyzes code through structured phases:
        1. Parse and understand code structure
        2. Identify patterns (good and bad)
        3. Detect potential issues
        4. Trace execution flow
        5. Suggest improvements/fixes

        Args:
            session: The current reasoning session
            input_text: Code to analyze (with optional error message)
            context: Optional context with hints like:
                - language: Programming language (auto-detected if not provided)
                - error_message: Error or exception text
                - focus: What to focus on (e.g., "performance", "security")

        Returns:
            A ThoughtNode containing the analysis results

        Examples:
            >>> method = CodeReasoning()
            >>> session = Session().start()
            >>> result = await method.execute(
            ...     session,
            ...     "def divide(a, b):\\n    return a / b"
            ... )
            >>> assert "division" in result.content.lower()
        """
        context = context or {}

        # Extract error message if present in input
        code, error_message = self._extract_code_and_error(input_text)

        # Override with context if provided
        if "error_message" in context:
            error_message = context["error_message"]

        # Detect programming language
        language = context.get("language") or self._detect_language(code)

        # Get analysis focus
        focus = context.get("focus", "correctness")

        # Phase 1: Analyze code structure
        structure_analysis = self._analyze_structure(code, language)

        # Phase 2: Identify patterns
        patterns = self._identify_patterns(code, language)

        # Phase 3: Detect issues
        issues = self._detect_issues(code, language, error_message)

        # Phase 4: Trace execution flow (conceptual)
        flow_trace = self._trace_execution_flow(code, language)

        # Phase 5: Suggest fixes or improvements
        suggestions = self._suggest_fixes(code, issues, focus, language)

        # Build comprehensive analysis content
        analysis_parts = [
            f"# Code Analysis ({language})",
            "",
            "## Structure Analysis",
            structure_analysis,
            "",
            "## Patterns Identified",
            patterns,
            "",
            "## Issues Detected",
            issues,
            "",
            "## Execution Flow",
            flow_trace,
            "",
            "## Suggestions",
            suggestions,
        ]

        if error_message:
            analysis_parts.insert(
                2,
                f"## Error Context\n```\n{error_message}\n```\n",
            )

        content = "\n".join(analysis_parts)

        # Calculate confidence based on issue detection
        confidence = self._calculate_confidence(issues, error_message)

        # Determine if issues were found for quality score
        quality_score = 0.9 if "No critical issues" not in issues else 0.7

        # Create the thought node
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CODE_REASONING,
            content=content,
            confidence=confidence,
            quality_score=quality_score,
            step_number=1,
            depth=0,
            metadata={
                "language": language,
                "has_error": error_message is not None,
                "focus": focus,
                "issues_count": len(self._count_issues(issues)),
                "supports_branching": True,
            },
        )

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        This allows for deeper analysis or exploration of alternative approaches.
        Can be used to explore different fix strategies as branches.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation (e.g., "explore performance fix")
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the analysis
        """
        context = context or {}

        # Determine continuation type
        if guidance and "branch" in guidance.lower():
            thought_type = ThoughtType.BRANCH
            branch_id = context.get("branch_id", f"branch_{uuid4().hex[:8]}")
        else:
            thought_type = ThoughtType.CONTINUATION
            branch_id = previous_thought.branch_id

        # Extract focus from guidance or context
        focus = context.get("focus", "alternative approach")
        if guidance:
            if "performance" in guidance.lower():
                focus = "performance"
            elif "security" in guidance.lower():
                focus = "security"
            elif "readability" in guidance.lower():
                focus = "readability"

        # Build continuation content
        content_parts = [
            f"# Continued Analysis: {focus.title()}",
            "",
            f"Building on previous analysis (thought {previous_thought.step_number})...",
            "",
        ]

        if guidance:
            content_parts.extend([
                f"Guidance: {guidance}",
                "",
            ])

        content_parts.extend([
            f"## Focus Area: {focus.title()}",
            f"This continuation explores {focus}-specific improvements and considerations.",
            "",
            "## Additional Insights",
            "Further analysis would examine:",
            f"- {focus.title()}-specific patterns",
            "- Trade-offs with other approaches",
            "- Implementation complexity",
            "- Testing requirements",
        ])

        content = "\n".join(content_parts)

        # Create continuation thought
        thought = ThoughtNode(
            id=str(uuid4()),
            type=thought_type,
            method_id=MethodIdentifier.CODE_REASONING,
            content=content,
            parent_id=previous_thought.id,
            branch_id=branch_id,
            confidence=0.8,
            quality_score=0.85,
            step_number=previous_thought.step_number + 1,
            depth=previous_thought.depth + 1,
            metadata={
                "focus": focus,
                "continuation_type": "branch" if thought_type == ThoughtType.BRANCH else "continuation",
            },
        )

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if healthy (always True for this native method)
        """
        return True

    # Helper methods for analysis

    def _extract_code_and_error(self, input_text: str) -> tuple[str, str | None]:
        """Extract code and optional error message from input.

        Args:
            input_text: The input text which may contain code and error

        Returns:
            Tuple of (code, error_message)
        """
        # Look for error patterns (case-insensitive)
        error_markers = [
            "error:",
            "exception:",
            "traceback:",
            "error message:",
            "---error---",
        ]

        error_message = None
        code = input_text
        lower_text = input_text.lower()

        for marker in error_markers:
            pos = lower_text.find(marker)
            if pos != -1:
                # Split at the actual position found in the text
                code = input_text[:pos].strip()
                error_message = input_text[pos + len(marker):].strip()
                break

        return code, error_message

    def _detect_language(self, code: str) -> str:
        """Detect the programming language from code.

        Args:
            code: The code to analyze

        Returns:
            Detected language name
        """
        # Pattern-based detection - order matters!
        # More specific languages must be checked before general ones:
        # - TypeScript before JavaScript (both use function/const/let)
        # - C++ before Java (both use public/private/void)
        # - Rust before JavaScript (both use let)
        # - Go before Python (both use import)
        detection_order = [
            ("typescript", [r":\s*(string|number|boolean|any|void)\b", r"\binterface\s+\w+\s*{", r"\btype\s+\w+\s*="]),
            ("c++", [r"#include\s*<", r"\bnamespace\s+", r"std::", r"\btemplate\s*<", r"::\w+"]),
            ("rust", [r"\bfn\s+\w+", r"\blet\s+mut\s+", r"\bimpl\s+", r"\b->\s*\w+", r"&mut\s+", r"&\w+"]),
            ("go", [r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+", r"\bimport\s+\(", r":=", r"\bfunc\s*\("]),
            ("java", [r"\bpublic\s+class\s+", r"\bprivate\s+\w+\s+\w+", r"\bprotected\s+", r"\bvoid\s+\w+\s*\("]),
            ("javascript", [r"\bfunction\s+\w+", r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bvar\s+", r"=>\s*[{(]"]),
            ("python", [r"^\s*def\s+\w+\s*\(", r"^\s*class\s+\w+", r"^\s*from\s+\w+\s+import", r"^\s*import\s+\w+"]),
        ]

        for language, lang_patterns in detection_order:
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return language

        return "unknown"

    def _analyze_structure(self, code: str, language: str) -> str:
        """Analyze the code structure.

        Args:
            code: The code to analyze
            language: Programming language

        Returns:
            Structure analysis description
        """
        lines = code.split("\n")
        line_count = len([line for line in lines if line.strip()])

        # Count common structural elements
        functions = len(re.findall(r"\b(def|function|func|fn|void|public|private)\s+\w+", code))
        classes = len(re.findall(r"\b(class|interface|struct)\s+\w+", code))
        imports = len(re.findall(r"\b(import|from|#include|using)\s+", code))

        parts = [
            f"- **Lines of code**: {line_count}",
            f"- **Functions/methods**: {functions}",
            f"- **Classes/interfaces**: {classes}",
            f"- **Import statements**: {imports}",
        ]

        # Add language-specific observations
        if language == "python":
            decorators = len(re.findall(r"^\s*@\w+", code, re.MULTILINE))
            if decorators > 0:
                parts.append(f"- **Decorators**: {decorators}")
        elif language in ("javascript", "typescript"):
            arrow_funcs = len(re.findall(r"=>\s*{", code))
            if arrow_funcs > 0:
                parts.append(f"- **Arrow functions**: {arrow_funcs}")

        return "\n".join(parts)

    def _identify_patterns(self, code: str, language: str) -> str:
        """Identify code patterns (good and bad).

        Args:
            code: The code to analyze
            language: Programming language

        Returns:
            Pattern analysis description
        """
        patterns_found = []

        # Check for common patterns
        if re.search(r"\btry\s*{|\btry:", code):
            patterns_found.append("✓ Error handling (try-catch/except)")

        if re.search(r"\bif\s+.*\bnot\s+|\bif\s+!|unless", code):
            patterns_found.append("⚠ Negative conditionals (consider positive logic)")

        if re.search(r"\w+\s*=\s*\w+\s*\?\s*\w+\s*:\s*\w+", code):
            patterns_found.append("✓ Ternary operators")

        if re.search(r"\b(map|filter|reduce)\s*\(", code):
            patterns_found.append("✓ Functional programming patterns")

        # Check for anti-patterns
        if re.search(r"\bexcept\s*:|catch\s*\(\s*\)", code):
            patterns_found.append("⚠ Bare except/catch (anti-pattern)")

        if language == "python" and re.search(r"\beval\s*\(", code):
            patterns_found.append("⚠ Use of eval() (security risk)")

        if re.search(r"\b(var|let)\s+\w+\s*=\s*\w+\s*=\s*", code):
            patterns_found.append("⚠ Chained assignments (readability concern)")

        # Nesting depth check
        max_indent = max((len(line) - len(line.lstrip()) for line in code.split("\n")), default=0)
        if max_indent > 16:
            patterns_found.append(f"⚠ Deep nesting detected (max indent: {max_indent} spaces)")

        if not patterns_found:
            return "No significant patterns detected in this code sample."

        return "\n".join(f"- {p}" for p in patterns_found)

    def _detect_issues(self, code: str, language: str, error_message: str | None) -> str:
        """Detect potential issues in the code.

        Args:
            code: The code to analyze
            language: Programming language
            error_message: Optional error message to help identify issues

        Returns:
            Issues analysis description
        """
        issues = []

        # Check for undefined variables (simple heuristic)
        if error_message and "undefined" in error_message.lower():
            var_match = re.search(r"['\"](\\w+)['\"]", error_message)
            if var_match:
                var_name = var_match.group(1)
                issues.append(f"🔴 **Undefined variable**: `{var_name}` is referenced but not defined")

        # Check for division by zero potential
        if re.search(r"/\s*0(?!\d)|/\s*\w+(?=\s*#.*zero)", code):
            issues.append("🟡 **Potential division by zero**: Check denominator validation")

        # Check for null/None reference potential
        if re.search(r"\.\w+(?!\s*\()|(?:null|None|nil)\.\w+", code):
            issues.append("🟡 **Potential null reference**: Consider null/None checks before access")

        # Language-specific checks
        if language == "python":
            # Mutable default arguments
            if re.search(r"def\s+\w+\([^)]*=\s*(\[\]|\{\})", code):
                issues.append("🟡 **Mutable default argument**: Use None and initialize in function body")

        if language in ("javascript", "typescript"):
            # == instead of ===
            if re.search(r"[^=!]==[^=]", code):
                issues.append("🟡 **Use === instead of ==**: Avoid type coercion issues")

        # Check for magic numbers
        magic_numbers = re.findall(r"\b\d{2,}\b", code)
        if len(magic_numbers) > 3:
            issues.append(f"🟡 **Magic numbers**: Consider using named constants ({len(magic_numbers)} found)")

        # Check for TODO/FIXME comments
        todos = re.findall(r"#\s*(TODO|FIXME|HACK|XXX)", code, re.IGNORECASE)
        if todos:
            issues.append(f"ℹ️ **Unresolved comments**: {len(todos)} TODO/FIXME markers found")

        if not issues:
            return "✅ No critical issues detected. Code appears structurally sound."

        return "\n".join(f"- {issue}" for issue in issues)

    def _trace_execution_flow(self, code: str, language: str) -> str:
        """Trace conceptual execution flow.

        Args:
            code: The code to analyze
            language: Programming language

        Returns:
            Execution flow description
        """
        flow_steps = []

        # Identify entry points
        if language == "python":
            if "if __name__" in code:
                flow_steps.append("1. Entry: `if __name__ == '__main__':` block")
        elif language in ("javascript", "typescript"):
            if re.search(r"\b(async\s+)?function\s+main\s*\(", code):
                flow_steps.append("1. Entry: `main()` function")

        # Identify function calls
        func_calls = re.findall(r"\b(\w+)\s*\(", code)
        if func_calls:
            unique_calls = list(dict.fromkeys(func_calls))[:5]  # First 5 unique
            flow_steps.append(f"2. Function calls: {', '.join(f'`{c}()`' for c in unique_calls)}")

        # Identify control flow
        has_if = bool(re.search(r"\bif\s+", code))
        has_loop = bool(re.search(r"\b(for|while)\s+", code))
        has_try = bool(re.search(r"\b(try|catch|except)\s*[:{]", code))

        control_flow = []
        if has_if:
            control_flow.append("conditional branching")
        if has_loop:
            control_flow.append("iteration")
        if has_try:
            control_flow.append("error handling")

        if control_flow:
            flow_steps.append(f"3. Control flow: {', '.join(control_flow)}")

        # Identify return points
        returns = len(re.findall(r"\breturn\b", code))
        if returns > 0:
            flow_steps.append(f"4. Exit points: {returns} return statement(s)")

        if not flow_steps:
            return "Execution flow is linear with no complex control structures."

        return "\n".join(flow_steps)

    def _suggest_fixes(self, code: str, issues: str, focus: str, language: str) -> str:
        """Suggest fixes or improvements.

        Args:
            code: The code to analyze
            issues: Detected issues
            focus: Analysis focus
            language: Programming language

        Returns:
            Fix suggestions
        """
        suggestions = []

        # Extract issues that need fixes
        if "Undefined variable" in issues:
            suggestions.append(
                "**Fix undefined variable**: Ensure all variables are declared before use. "
                "Check for typos or missing imports."
            )

        if "division by zero" in issues.lower():
            suggestions.append(
                "**Add validation**: Check denominator before division:\n"
                "```python\n"
                "if denominator != 0:\n"
                "    result = numerator / denominator\n"
                "else:\n"
                "    # Handle zero case\n"
                "```"
            )

        if "Mutable default argument" in issues:
            suggestions.append(
                "**Fix mutable defaults**: Use None as default and initialize in function:\n"
                "```python\n"
                "def func(items=None):\n"
                "    if items is None:\n"
                "        items = []\n"
                "```"
            )

        if "=== instead of ==" in issues:
            suggestions.append(
                "**Use strict equality**: Replace `==` with `===` and `!=` with `!==` "
                "to avoid type coercion issues."
            )

        # Focus-specific suggestions
        if focus == "performance":
            suggestions.append(
                "**Performance considerations**:\n"
                "- Profile code to identify bottlenecks\n"
                "- Consider using built-in functions (often optimized)\n"
                "- Evaluate algorithm complexity (time and space)"
            )
        elif focus == "security":
            suggestions.append(
                "**Security considerations**:\n"
                "- Validate and sanitize all inputs\n"
                "- Avoid eval() and similar dynamic code execution\n"
                "- Use parameterized queries for database operations"
            )
        elif focus == "readability":
            suggestions.append(
                "**Readability improvements**:\n"
                "- Add docstrings/comments for complex logic\n"
                "- Use descriptive variable names\n"
                "- Break down large functions into smaller ones"
            )

        # General suggestions
        if "No critical issues" in issues:
            suggestions.append(
                "**Code looks good!** Consider:\n"
                "- Adding unit tests if not present\n"
                "- Documenting edge cases\n"
                "- Running a linter for style consistency"
            )

        if not suggestions:
            suggestions.append(
                "Review the detected issues above and apply appropriate fixes. "
                "Consider adding tests to validate the changes."
            )

        return "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    def _calculate_confidence(self, issues: str, error_message: str | None) -> float:
        """Calculate confidence score based on analysis.

        Args:
            issues: Detected issues text
            error_message: Optional error message

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with base confidence
        confidence = 0.85

        # Lower confidence if many issues found
        critical_issues = issues.count("🔴")
        warning_issues = issues.count("🟡")

        confidence -= critical_issues * 0.05
        confidence -= warning_issues * 0.02

        # Increase confidence if we have error message (more context)
        if error_message:
            confidence += 0.1

        # Clamp to valid range
        return max(0.5, min(1.0, confidence))

    def _count_issues(self, issues: str) -> list[str]:
        """Count issues from the issues text.

        Args:
            issues: Issues text

        Returns:
            List of issue markers found
        """
        markers = []
        for marker in ["🔴", "🟡", "ℹ️"]:
            markers.extend([marker] * issues.count(marker))
        return markers


# Method metadata for registry
CODE_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CODE_REASONING,
    name="Code Reasoning",
    description="Specialized reasoning for code analysis, debugging, and development. "
    "Analyzes code structure, identifies patterns, detects issues, traces execution, "
    "and suggests improvements.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset({
        "code",
        "debugging",
        "analysis",
        "programming",
        "development",
        "bug-detection",
        "refactoring",
    }),
    complexity=6,  # Medium-high complexity
    supports_branching=True,  # Can branch for different fix approaches
    supports_revision=True,  # Can revise analysis
    requires_context=False,  # Works without additional context
    min_thoughts=1,
    max_thoughts=10,  # Can generate multiple thoughts for deep analysis
    avg_tokens_per_thought=600,
    best_for=(
        "code analysis",
        "bug detection",
        "debugging",
        "code review",
        "refactoring suggestions",
        "security analysis",
        "performance optimization",
    ),
    not_recommended_for=(
        "creative writing",
        "ethical dilemmas",
        "mathematical proofs",
        "general reasoning",
    ),
)
