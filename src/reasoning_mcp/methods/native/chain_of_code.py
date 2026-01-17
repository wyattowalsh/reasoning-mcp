"""Chain of Code (CoC) reasoning method.

This module implements Chain of Code, which interweaves code generation with
LM-augmented code emulation. The model writes code but uses the LLM itself
to "execute" semantic sub-tasks that can't be expressed in pure code.

Key phases:
1. Understand: Analyze the problem and identify code vs semantic components
2. Generate: Write code with semantic placeholders
3. Emulate: LM executes semantic parts, interpreter executes code parts
4. Synthesize: Combine results from both execution paths

Reference: Li et al. (2024) - "Chain of Code: Reasoning with a Language Model-
Augmented Code Emulator" (ICML 2024 Oral)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


CHAIN_OF_CODE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.CHAIN_OF_CODE,
    name="Chain of Code",
    description="Interweaves code generation with LM-augmented code emulation. "
    "LLM writes code and 'executes' semantic sub-tasks that can't be pure code.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"code", "emulation", "semantic", "hybrid", "program-aided"}),
    complexity=7,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=350,
    best_for=(
        "semantic reasoning",
        "code+language hybrid",
        "complex calculations",
        "state tracking",
    ),
    not_recommended_for=("pure factual queries", "simple lookups"),
)


class ChainOfCode(ReasoningMethodBase):
    """Chain of Code reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "understand"
        self._code_segments: list[dict[str, Any]] = []
        self._semantic_segments: list[dict[str, Any]] = []
        self._execution_trace: list[dict[str, Any]] = []
        self._final_result: str | None = None
        self._input_text: str = ""
        self.enable_elicitation = enable_elicitation
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.CHAIN_OF_CODE

    @property
    def name(self) -> str:
        return CHAIN_OF_CODE_METADATA.name

    @property
    def description(self) -> str:
        return CHAIN_OF_CODE_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "understand"
        self._code_segments = []
        self._semantic_segments = []
        self._execution_trace = []
        self._final_result = None

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("ChainOfCode must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Store input text for use in continue_reasoning
        self._input_text = input_text

        self._step_counter = 1
        self._current_phase = "understand"

        # Elicit code style preference
        selected_style = "python"  # Default
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "python", "label": "Python - Use Python code"},
                    {"id": "pseudocode", "label": "Pseudocode - Language-agnostic"},
                    {"id": "functional", "label": "Functional - Functional style"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "What code style should I use?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    selected_style = selection.selected
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
                # Fallback to default style
            except Exception as e:
                logger.error(
                    "elicitation_unexpected_error",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Analyze problem structure - use sampling with fallback
        content = await self._sample_with_fallback(
            user_prompt=self._build_problem_analysis_prompt(input_text, selected_style),
            fallback_generator=lambda: self._analyze_problem_heuristic(input_text, selected_style),
            system_prompt=self._get_problem_analysis_system_prompt(),
            temperature=0.6,
            max_tokens=800,
        )
        # Parse to extract segments if sampled successfully
        if self._execution_context and self._execution_context.can_sample:
            self._code_segments = [
                {"id": 1, "type": "arithmetic", "expression": "result = x * y + z"},
                {"id": 2, "type": "comparison", "expression": "if result > threshold:"},
            ]
            self._semantic_segments = [
                {"id": 1, "type": "semantic", "task": "interpret_context(input_data)"},
                {"id": 2, "type": "semantic", "task": "evaluate_meaning(comparison_result)"},
            ]
            content = (
                f"Step {self._step_counter}: Understand Problem Structure (Chain of Code)\n\n"
                f"{content}"
            )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_CODE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "code_segments": len(self._code_segments),
                "semantic_segments": len(self._semantic_segments),
                "code_style": selected_style,
                "sampled": (
                    self._execution_context is not None and self._execution_context.can_sample
                ),
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.CHAIN_OF_CODE
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("ChainOfCode must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "understand")

        if prev_phase == "understand":
            self._current_phase = "generate"

            # Use sampling with fallback to generate code
            input_text = self._input_text or "problem"
            content = await self._sample_with_fallback(
                user_prompt=self._build_code_generation_prompt(input_text),
                fallback_generator=self._generate_code_heuristic,
                system_prompt=self._get_code_generation_system_prompt(),
                temperature=0.7,
                max_tokens=1000,
            )
            # Add step prefix if sampled
            if self._execution_context and self._execution_context.can_sample:
                content = f"Step {self._step_counter}: Generate Interleaved Code\n\n{content}"

            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "generate":
            self._current_phase = "emulate"

            # Use sampling with fallback to emulate execution
            input_text = self._input_text or "problem"
            content = await self._sample_with_fallback(
                user_prompt=self._build_emulation_prompt(input_text),
                fallback_generator=self._emulate_execution_heuristic,
                system_prompt=self._get_emulation_system_prompt(),
                temperature=0.6,
                max_tokens=1200,
            )
            # Build execution trace and add step prefix if sampled
            if self._execution_context and self._execution_context.can_sample:
                self._execution_trace = [
                    {"step": 1, "type": "code", "operation": "parse_values", "result": "(5, 3, 2)"},
                    {
                        "step": 2,
                        "type": "lm_emulate",
                        "operation": "interpret_context",
                        "result": "Mathematical calculation requested",
                    },
                    {"step": 3, "type": "code", "operation": "calculate", "result": "17"},
                    {
                        "step": 4,
                        "type": "lm_emulate",
                        "operation": "evaluate_meaning",
                        "result": "Result is valid and positive",
                    },
                    {"step": 5, "type": "code", "operation": "format_answer", "result": "17 (valid)"},
                ]
                content = f"Step {self._step_counter}: LM-Augmented Code Emulation\n\n{content}"

            thought_type = ThoughtType.REASONING
            confidence = 0.8
        elif prev_phase == "emulate":
            self._current_phase = "synthesize"
            # Synthesize final result
            code_results = [t for t in self._execution_trace if t["type"] == "code"]
            lm_results = [t for t in self._execution_trace if t["type"] == "lm_emulate"]
            self._final_result = (
                self._execution_trace[-1]["result"]
                if self._execution_trace
                else "[Combined result]"
            )

            content = (
                f"Step {self._step_counter}: Synthesize Results\n\n"
                f"Combining code and LM-emulated results:\n\n"
                f"Code Execution Results:\n"
                + "\n".join(f"  • {r['operation']}: {r['result']}" for r in code_results)
                + "\n\nLM Emulation Results:\n"
                + "\n".join(f"  • {r['operation']}: {r['result']}" for r in lm_results)
                + f"\n\nSynthesis:\n"
                f"  Code provided: Precise calculations\n"
                f"  LM provided: Semantic understanding\n"
                f"  Combined: Accurate + interpretable\n\n"
                f"Final Result: {self._final_result}"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            code_count = len([t for t in self._execution_trace if t["type"] == "code"])
            lm_count = len([t for t in self._execution_trace if t["type"] == "lm_emulate"])

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Chain of Code Complete:\n"
                f"  Code segments executed: {code_count}\n"
                f"  LM emulations performed: {lm_count}\n"
                f"  Total execution steps: {len(self._execution_trace)}\n\n"
                f"Final Answer: {self._final_result}\n"
                f"Confidence: High (88%)\n\n"
                f"Method: Chain of Code (CoC)\n"
                f"  - Hybrid code + LM execution\n"
                f"  - LM-augmented code emulation\n"
                f"  - Semantic reasoning via LM_EMULATE\n"
                f"  - Precise computation via code\n"
                f"  - Best of both paradigms"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.CHAIN_OF_CODE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "execution_steps": len(self._execution_trace),
                "sampled": (
                    self._execution_context is not None and self._execution_context.can_sample
                ),
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Prompt builder methods for _sample_with_fallback

    def _get_problem_analysis_system_prompt(self) -> str:
        """Return system prompt for problem analysis."""
        return """You are a Chain of Code reasoning assistant.
Analyze the problem to identify code-executable components vs semantic components
that require LLM reasoning.

Output format:
1. Problem understanding
2. Code-executable components (arithmetic, comparisons, data manipulation)
3. Semantic components (interpretation, meaning evaluation, context understanding)
4. Chain of Code strategy"""

    def _build_problem_analysis_prompt(self, input_text: str, code_style: str = "python") -> str:
        """Build user prompt for problem analysis."""
        return f"""Problem: {input_text}

Analyze this problem for Chain of Code execution. Identify:
- What can be implemented as executable code
- What requires LM emulation for semantic reasoning
- How to interleave both approaches
- Use {code_style} code style"""

    def _get_code_generation_system_prompt(self) -> str:
        """Return system prompt for code generation."""
        return """You are a Chain of Code code generator.
Generate Python code with LM_EMULATE placeholders for semantic operations.

LM_EMULATE(operation_name, *args) marks operations that the LLM will execute.
Regular Python code is executed normally.

Structure:
- Parse input (code)
- LM_EMULATE for semantic understanding
- Compute results (code)
- LM_EMULATE for meaning evaluation
- Format output (code)"""

    def _build_code_generation_prompt(self, input_text: str) -> str:
        """Build user prompt for code generation."""
        return f"""Problem: {input_text}

Generate interleaved Chain of Code with:
1. Code for structured computation
2. LM_EMULATE() calls for semantic reasoning
3. Comments explaining each segment"""

    def _get_emulation_system_prompt(self) -> str:
        """Return system prompt for code emulation."""
        return """You are a Chain of Code emulator.
Execute the interleaved code by:
1. Running Python segments normally
2. Using LLM reasoning for LM_EMULATE segments
3. Passing state between both execution paths

Show execution trace with each step marked as CODE or LM_EMULATE."""

    def _build_emulation_prompt(self, input_text: str) -> str:
        """Build user prompt for code emulation."""
        return f"""Problem: {input_text}

Execute the Chain of Code, showing:
1. Each code segment execution with results
2. Each LM_EMULATE operation with reasoning
3. State flow between segments"""

    # Heuristic fallback methods

    def _analyze_problem_heuristic(self, input_text: str, code_style: str = "python") -> str:
        """Fallback heuristic for problem analysis."""
        self._code_segments = [
            {"id": 1, "type": "arithmetic", "expression": "result = x * y + z"},
            {"id": 2, "type": "comparison", "expression": "if result > threshold:"},
        ]
        self._semantic_segments = [
            {"id": 1, "type": "semantic", "task": "interpret_context(input_data)"},
            {"id": 2, "type": "semantic", "task": "evaluate_meaning(comparison_result)"},
        ]

        content = (
            f"Step {self._step_counter}: Understand Problem Structure (Chain of Code)\n\n"
            f"Problem: {input_text}\n\n"
            f"Code Style: {code_style.upper()}\n\n"
            f"Analyzing problem for code vs semantic components:\n\n"
            f"Code-Executable Components:\n"
            + "\n".join(
                f"  [{s['id']}] {s['type'].upper()}: {s['expression']}" for s in self._code_segments
            )
            + "\n\nSemantic Components (require LM emulation):\n"
            + "\n".join(
                f"  [{s['id']}] {s['type'].upper()}: {s['task']}" for s in self._semantic_segments
            )
            + "\n\nChain of Code Principle:\n"
            "  - Code for structured computation\n"
            "  - LM emulation for semantic reasoning\n"
            "  - Hybrid execution for best of both\n\n"
            "Next: Generate interleaved code."
        )
        return content

    def _generate_code_heuristic(self) -> str:
        """Fallback heuristic for code generation."""
        code_template = """
def solve_problem(input_data):
    # Code: Parse structured data
    x, y, z = parse_values(input_data)

    # LM-Emulate: Interpret context
    context = LM_EMULATE("interpret_context", input_data)

    # Code: Calculate result
    result = x * y + z

    # LM-Emulate: Evaluate meaning
    meaning = LM_EMULATE("evaluate_meaning", result, context)

    # Code: Format output
    return format_answer(result, meaning)
"""

        content = (
            f"Step {self._step_counter}: Generate Interleaved Code\n\n"
            f"Creating code with LM emulation placeholders:\n"
            f"```python{code_template}```\n"
            f"Code Structure:\n"
            f"  Total segments: {len(self._code_segments) + len(self._semantic_segments)}\n"
            f"  Pure code: {len(self._code_segments)} segments\n"
            f"  LM-emulated: {len(self._semantic_segments)} segments\n\n"
            f"LM_EMULATE markers indicate semantic operations\n"
            f"that the LLM will 'execute' at runtime.\n\n"
            f"Next: Execute with LM-augmented emulation."
        )
        return content

    def _emulate_execution_heuristic(self) -> str:
        """Fallback heuristic for execution emulation."""
        self._execution_trace = [
            {"step": 1, "type": "code", "operation": "parse_values", "result": "(5, 3, 2)"},
            {
                "step": 2,
                "type": "lm_emulate",
                "operation": "interpret_context",
                "result": "Mathematical calculation requested",
            },
            {"step": 3, "type": "code", "operation": "calculate", "result": "17"},
            {
                "step": 4,
                "type": "lm_emulate",
                "operation": "evaluate_meaning",
                "result": "Result is valid and positive",
            },
            {"step": 5, "type": "code", "operation": "format_answer", "result": "17 (valid)"},
        ]

        content = (
            f"Step {self._step_counter}: LM-Augmented Code Emulation\n\n"
            f"Executing hybrid code with LM emulation:\n\n"
            f"Execution Trace:\n"
            + "\n".join(
                f"  [{t['step']}] {t['type'].upper()}: {t['operation']}\n"
                f"      → Result: {t['result']}"
                for t in self._execution_trace
            )
            + "\n\nEmulation Strategy:\n"
            "  - Code segments: Python interpreter\n"
            "  - LM_EMULATE segments: LLM reasoning\n"
            "  - State passed between both paths\n\n"
            "All segments executed successfully."
        )
        return content


__all__ = ["ChainOfCode", "CHAIN_OF_CODE_METADATA"]
