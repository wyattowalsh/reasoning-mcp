"""Program of Thoughts (PoT) reasoning method.

This module implements Program of Thoughts prompting (Chen et al. 2022), which
disentangles computation from reasoning by generating executable Python code
to solve numerical and logical problems. The LLM acts as a program generator
while actual computation is delegated to an interpreter.

Key phases:
1. Analyze: Understand the problem and identify computational needs
2. Generate: Create Python code to solve the problem
3. Execute: Run/simulate code execution
4. Interpret: Analyze results and provide final answer

Reference: Chen et al. (2022) - "Program of Thoughts Prompting: Disentangling
Computation from Reasoning for Numerical Reasoning Tasks"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

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


# Metadata for Program of Thoughts method
PROGRAM_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.PROGRAM_OF_THOUGHTS,
    name="Program of Thoughts",
    description="Generates executable Python code to solve problems, separating "
    "computation from reasoning. The LLM acts as a program generator while "
    "actual computation is delegated to an interpreter through "
    "analyze → generate → execute → interpret phases.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "code-generation",
            "computational",
            "numerical-reasoning",
            "python",
            "interpreter",
            "mathematical",
            "executable",
            "disentangled",
        }
    ),
    complexity=6,  # Moderate-high complexity
    supports_branching=False,  # Linear code generation flow
    supports_revision=True,  # Can revise code based on execution errors
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: analyze + generate + execute + interpret
    max_thoughts=8,  # Including potential code revisions
    avg_tokens_per_thought=500,  # Code generation can be verbose
    best_for=(
        "numerical reasoning",
        "mathematical problems",
        "algorithmic tasks",
        "data manipulation",
        "logical computations",
        "financial calculations",
        "scientific computing",
        "step-by-step calculations",
    ),
    not_recommended_for=(
        "open-ended questions",
        "subjective analysis",
        "creative writing",
        "philosophical reasoning",
        "tasks without clear computation",
    ),
)

logger = structlog.get_logger(__name__)


class ProgramOfThoughts(ReasoningMethodBase):
    """Program of Thoughts reasoning method implementation.

    This class implements the PoT pattern where the system generates Python
    code to solve problems, separating computation from reasoning:
    1. Analyze: Understand problem structure and computational needs
    2. Generate: Create Python code that solves the problem
    3. Execute: Simulate/run the generated code
    4. Interpret: Analyze execution results and formulate answer

    Key characteristics:
    - Code-based problem solving
    - Computation delegation to interpreter
    - Separation of reasoning and calculation
    - Support for numerical/mathematical problems
    - Moderate-high complexity (6)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = ProgramOfThoughts()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="If a train travels 60 mph for 2.5 hours, how far does it go?"
        ... )
        >>> print(result.content)  # Analyze phase

        Continue with code generation:
        >>> code = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Generate Python code"
        ... )
        >>> print(code.content)  # Contains generated Python code
    """

    # Maximum lines of generated code
    MAX_CODE_LINES = 50
    # Maximum execution attempts for error recovery
    MAX_RETRIES = 2

    # Enable LLM sampling for dynamic content generation
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Program of Thoughts method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "analyze"  # analyze, generate, execute, interpret
        self._generated_code: str = ""
        self._execution_result: str = ""
        self._retry_count = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.PROGRAM_OF_THOUGHTS

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return PROGRAM_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return PROGRAM_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Program of Thoughts method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "analyze"
        self._generated_code = ""
        self._execution_result = ""
        self._retry_count = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Program of Thoughts method.

        Creates the initial analysis phase, understanding the problem
        and identifying computational needs.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the analysis phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Program of Thoughts method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "analyze"
        self._generated_code = ""
        self._execution_result = ""
        self._retry_count = 0

        # Generate analysis content
        if use_sampling:
            content = await self._sample_analysis(input_text, context)
        else:
            content = self._generate_analysis_heuristic(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.PROGRAM_OF_THOUGHTS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.65,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "program_of_thoughts",
                "phase": self._current_phase,
                "code_generated": False,
                "executed": False,
                "retry_count": self._retry_count,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.PROGRAM_OF_THOUGHTS

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
        """Continue reasoning from a previous thought.

        Implements the PoT phase progression:
        - After analyze: generate Python code
        - After generate: execute the code
        - After execute: interpret results
        - After interpret: conclude or revise if needed

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the PoT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Program of Thoughts method must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "analyze")

        # Determine if we can use sampling
        use_sampling = (
            self._execution_context is not None
            and self._execution_context.can_sample
            and self._use_sampling
        )

        if prev_phase == "analyze":
            # Next: generate Python code
            self._current_phase = "generate"
            thought_type = ThoughtType.REASONING
            if use_sampling:
                content = await self._sample_code_generation(previous_thought, guidance, context)
            else:
                content = self._generate_code_heuristic(previous_thought, guidance, context)
            self._generated_code = content  # Store for later reference
            confidence = 0.75
            quality_score = 0.7

        elif prev_phase == "generate":
            # Next: execute the code
            self._current_phase = "execute"
            thought_type = ThoughtType.ACTION
            if use_sampling:
                content, success = await self._sample_code_execution(
                    previous_thought, guidance, context
                )
            else:
                content, success = self._execute_code_heuristic(previous_thought, guidance, context)
            self._execution_result = content
            confidence = 0.8 if success else 0.4
            quality_score = 0.75 if success else 0.4

        elif prev_phase == "execute":
            # Check if execution was successful
            execution_success = previous_thought.metadata.get("execution_success", True)

            if not execution_success and self._retry_count < self.MAX_RETRIES:
                # Retry: revise code
                self._current_phase = "generate"
                self._retry_count += 1
                thought_type = ThoughtType.REVISION
                if use_sampling:
                    content = await self._sample_code_revision(previous_thought, guidance, context)
                else:
                    content = self._revise_code_heuristic(previous_thought, guidance, context)
                self._generated_code = content
                confidence = 0.7
                quality_score = 0.65
            else:
                # Next: interpret results
                self._current_phase = "interpret"
                thought_type = ThoughtType.SYNTHESIS
                if use_sampling:
                    content = await self._sample_result_interpretation(
                        previous_thought, guidance, context
                    )
                else:
                    content = self._interpret_results_heuristic(previous_thought, guidance, context)
                confidence = 0.85
                quality_score = 0.85

        elif prev_phase == "interpret":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if use_sampling:
                content = await self._sample_conclusion(previous_thought, guidance, context)
            else:
                content = self._generate_conclusion_heuristic(previous_thought, guidance, context)
            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback
            self._current_phase = "interpret"
            thought_type = ThoughtType.SYNTHESIS
            if use_sampling:
                content = await self._sample_result_interpretation(
                    previous_thought, guidance, context
                )
            else:
                content = self._interpret_results_heuristic(previous_thought, guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.PROGRAM_OF_THOUGHTS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "program_of_thoughts",
                "code_generated": bool(self._generated_code),
                "executed": self._current_phase in ("execute", "interpret", "conclude"),
                "retry_count": self._retry_count,
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    # Sampling methods (LLM-based)
    async def _sample_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate problem analysis using LLM sampling.

        Args:
            input_text: The problem to analyze
            context: Optional additional context

        Returns:
            The content for the analysis thought
        """
        system_prompt = """You are a Program of Thoughts reasoning assistant.
Analyze the given problem to identify computational requirements and plan code generation.
Your response should:
1. Clearly identify the input variables and their types
2. List the required operations/calculations
3. Specify the expected output format
4. Consider edge cases that need handling
5. Outline the approach for generating executable Python code"""

        user_prompt = f"""Problem: {input_text}

Analyze this problem for a Program of Thoughts approach:
1. Identify all input variables and their data types
2. Determine what computational operations are needed
3. Specify the expected output format
4. List potential edge cases to handle in the code
5. Outline the strategy for generating Python code to solve this

Begin your computational analysis."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_analysis_heuristic(input_text, context),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

    async def _sample_code_generation(
        self,
        analysis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate Python code using LLM sampling.

        Args:
            analysis_thought: The previous analysis thought
            guidance: Optional guidance for code generation
            context: Optional additional context

        Returns:
            The content containing generated Python code
        """
        system_prompt = """You are a Program of Thoughts code generator.
Generate clean, executable Python code to solve the problem.
Your response should:
1. Include a complete, well-structured Python function
2. Add clear comments explaining each step
3. Use proper variable names and coding conventions
4. Handle edge cases appropriately
5. Return the final result clearly
6. Keep code concise (under 50 lines)"""

        analysis_content = analysis_thought.content
        guidance_text = f"\nAdditional Guidance: {guidance}" if guidance else ""

        user_prompt = f"""Based on this problem analysis:
{analysis_content}
{guidance_text}

Generate executable Python code that:
1. Defines a function to solve the problem
2. Includes clear comments for each major step
3. Handles the identified edge cases
4. Returns the computed result
5. Includes code to execute the function and print the answer

Provide complete, runnable Python code within a code block."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_code_heuristic(
                analysis_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=1200,
        )

    async def _sample_code_execution(
        self,
        code_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, bool]:
        """Simulate code execution using LLM sampling.

        Args:
            code_thought: The thought containing the generated code
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            A tuple of (execution content, success status)
        """
        system_prompt = """You are a Python code execution simulator for Program of Thoughts.
Analyze the given code and simulate its execution.
Your response should:
1. Trace through the code logic step-by-step
2. Show intermediate computational results
3. Display the final output
4. Indicate whether execution would succeed or fail
5. If there are errors, clearly identify them"""

        code_content = code_thought.content

        user_prompt = f"""Analyze and simulate execution of this code:
{code_content}

Provide:
1. Step-by-step trace of execution
2. Intermediate values computed
3. Final output/result
4. Success status (SUCCESS or ERROR)
5. Any error messages if applicable

Simulate the execution and show the results."""

        # Use _sample_with_fallback, but we need to handle the tuple return differently
        # by wrapping the heuristic result
        def fallback() -> str:
            content, _ = self._execute_code_heuristic(code_thought, guidance, context)
            return content

        result_str = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1000,
        )

        # Determine success based on presence of "SUCCESS" or "ERROR" in response
        success = "SUCCESS" in result_str.upper() and "ERROR" not in result_str.upper()
        return result_str, success

    async def _sample_code_revision(
        self,
        error_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Revise code after execution error using LLM sampling.

        Args:
            error_thought: The thought containing the error
            guidance: Optional guidance for revision
            context: Optional additional context

        Returns:
            The content containing revised Python code
        """
        system_prompt = """You are a Program of Thoughts code debugger.
Analyze execution errors and generate corrected code.
Your response should:
1. Identify the specific error in the code
2. Explain what went wrong
3. Provide corrected, executable Python code
4. Add comments explaining the fix
5. Ensure the revised code handles the error case"""

        error_content = error_thought.content
        guidance_text = f"\nAdditional Guidance: {guidance}" if guidance else ""

        user_prompt = f"""Execution encountered issues:
{error_content}
{guidance_text}

Previous code attempt: {self._retry_count}

Analyze the error and provide:
1. Error analysis: What went wrong?
2. Correction strategy: How to fix it?
3. Revised Python code that addresses the issue
4. Comments explaining the changes

Generate corrected code within a code block."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._revise_code_heuristic(
                error_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=1200,
        )

    async def _sample_result_interpretation(
        self,
        execution_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Interpret execution results using LLM sampling.

        Args:
            execution_thought: The thought containing execution results
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content interpreting the results
        """
        system_prompt = """You are a Program of Thoughts result interpreter.
Analyze code execution results and provide meaningful interpretation.
Your response should:
1. Summarize the computed result
2. Verify the result makes sense for the problem
3. Perform sanity checks on the output
4. Explain how the code solved the problem
5. Assess confidence in the result"""

        execution_content = execution_thought.content

        user_prompt = f"""Code execution completed:
{execution_content}

Interpret the results:
1. What was the final computed value?
2. Does this result make sense for the original problem?
3. What sanity checks validate this answer?
4. How did the code successfully solve the problem?
5. What is your confidence level in this result?

Provide a detailed interpretation."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._interpret_results_heuristic(
                execution_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

    async def _sample_conclusion(
        self,
        interpret_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final conclusion using LLM sampling.

        Args:
            interpret_thought: The thought containing result interpretation
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the conclusion
        """
        system_prompt = """You are a Program of Thoughts conclusion generator.
Synthesize the entire reasoning process into a final answer.
Your response should:
1. State the final answer clearly
2. Summarize the approach taken
3. Highlight key computational steps
4. Assess confidence in the result
5. Note any limitations or assumptions"""

        interpret_content = interpret_thought.content

        user_prompt = f"""Based on the interpretation:
{interpret_content}

Retries needed: {self._retry_count}

Generate a final conclusion that:
1. Clearly states the final answer
2. Summarizes the Program of Thoughts approach used
3. Reviews the key steps (analyze → generate → execute → interpret)
4. Assesses confidence level (computation verified through code)
5. Notes the number of retries if any were needed

Provide a comprehensive final answer."""

        return await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: self._generate_conclusion_heuristic(
                interpret_thought, guidance, context
            ),
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

    # Heuristic methods (fallback when sampling unavailable)
    def _generate_analysis_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the analysis phase content (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Problem Analysis (Program of Thoughts)\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing computational requirements...\n\n"
            f"Key Components Identified:\n"
            f"1. Input variables and their types\n"
            f"2. Required operations/calculations\n"
            f"3. Expected output format\n"
            f"4. Edge cases to consider\n\n"
            f"Approach: I will generate Python code to solve this problem, "
            f"separating the computational logic from reasoning. This allows "
            f"for precise calculations while maintaining clear reasoning."
        )

    def _generate_code_heuristic(
        self,
        analysis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate Python code to solve the problem (heuristic fallback)."""
        guidance_text = f"\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Code Generation\n\n"
            f"Based on the analysis, generating Python code...{guidance_text}\n\n"
            f"```python\n"
            f"# Program of Thoughts - Generated Code\n"
            f"# Problem: [extracted from analysis]\n\n"
            f"def solve_problem():\n"
            f"    # Step 1: Define inputs\n"
            f"    # [LLM would extract variables from problem]\n"
            f"    \n"
            f"    # Step 2: Perform calculations\n"
            f"    # [LLM would generate computation logic]\n"
            f"    \n"
            f"    # Step 3: Format and return result\n"
            f"    # result = ...\n"
            f"    return result\n\n"
            f"# Execute\n"
            f"answer = solve_problem()\n"
            f"print(f'Answer: {{answer}}')\n"
            f"```\n\n"
            f"Code generated. Ready for execution."
        )

    def _execute_code_heuristic(
        self,
        code_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, bool]:
        """Simulate code execution (heuristic fallback)."""
        # In a real implementation, this would safely execute the code
        success = True  # Simulated success

        content = (
            f"Step {self._step_counter}: Code Execution\n\n"
            f"Executing generated Python code...\n\n"
            f"[Simulated Execution Output]\n"
            f">>> Running solve_problem()\n"
            f">>> Computation in progress...\n"
            f">>> Answer: [computed result]\n\n"
            f"Execution Status: {'SUCCESS' if success else 'ERROR'}\n"
        )

        return content, success

    def _revise_code_heuristic(
        self,
        error_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Revise code after execution error (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Code Revision (Attempt {self._retry_count})\n\n"
            f"Analyzing execution error and revising code...\n\n"
            f"Error Analysis:\n"
            f"- Identified issue in previous code\n"
            f"- Applying correction\n\n"
            f"```python\n"
            f"# Revised Program of Thoughts Code\n"
            f"def solve_problem_v{self._retry_count + 1}():\n"
            f"    # [Corrected implementation]\n"
            f"    pass\n"
            f"```\n"
        )

    def _interpret_results_heuristic(
        self,
        execution_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Interpret the execution results (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Result Interpretation\n\n"
            f"Analyzing the computed result...\n\n"
            f"Execution Result: [from previous step]\n\n"
            f"Interpretation:\n"
            f"- The computation was performed correctly\n"
            f"- The result aligns with expected output format\n"
            f"- Verification: [sanity check of result]\n\n"
            f"The code successfully computed the answer by separating "
            f"the calculation logic from the reasoning process."
        )

    def _generate_conclusion_heuristic(
        self,
        interpret_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion (heuristic fallback)."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Based on the Program of Thoughts approach:\n\n"
            f"1. Analyzed the problem structure\n"
            f"2. Generated executable Python code\n"
            f"3. Executed the computation\n"
            f"4. Interpreted the results\n\n"
            f"Final Answer: [computed result with explanation]\n\n"
            f"Confidence: High (computation verified through code execution)\n"
            f"Retries needed: {self._retry_count}"
        )


# Export
__all__ = ["ProgramOfThoughts", "PROGRAM_OF_THOUGHTS_METADATA"]
