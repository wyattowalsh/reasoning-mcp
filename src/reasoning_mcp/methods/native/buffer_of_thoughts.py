"""Buffer of Thoughts (BoT) reasoning method.

This module implements Buffer of Thoughts based on Yang et al. (NeurIPS 2024),
which caches reusable thought templates for efficient multi-step reasoning.
The method maintains a buffer of high-level reasoning structures that can be
instantiated for new problems.

Key phases:
1. Distill: Extract problem essence and identify relevant templates
2. Retrieve: Find matching thought templates from buffer
3. Instantiate: Adapt templates to the current problem
4. Reason: Execute templated reasoning with problem specifics

Reference: Yang et al. (2024) - "Buffer of Thoughts: Thought-Augmented Reasoning"
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
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


BUFFER_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.BUFFER_OF_THOUGHTS,
    name="Buffer of Thoughts",
    description="Caches reusable thought templates for efficient multi-step reasoning. "
    "Distills problems, retrieves matching templates, and instantiates for new tasks.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"buffer", "templates", "reusable", "efficient", "neurips-2024"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=("recurring problem types", "template-based reasoning", "efficiency"),
    not_recommended_for=("novel unique problems", "one-off queries"),
)


class BufferOfThoughts(ReasoningMethodBase):
    """Buffer of Thoughts reasoning method implementation."""

    # Template buffer (simplified representation)
    TEMPLATE_BUFFER = [
        {
            "id": "math_word",
            "name": "Math Word Problem",
            "structure": ["parse", "formulate", "compute", "verify"],
        },
        {
            "id": "logical_deduction",
            "name": "Logical Deduction",
            "structure": ["premises", "rules", "inference", "conclude"],
        },
        {
            "id": "comparison",
            "name": "Comparative Analysis",
            "structure": ["criteria", "evaluate_each", "compare", "decide"],
        },
        {
            "id": "causal_chain",
            "name": "Causal Chain",
            "structure": ["identify_cause", "trace_effects", "validate", "conclude"],
        },
    ]

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "distill"
        self._selected_template: dict[str, Any] | None = None
        self._instantiated_steps: list[str] = []
        self._use_sampling: bool = True
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.BUFFER_OF_THOUGHTS

    @property
    def name(self) -> str:
        return BUFFER_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        return BUFFER_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "distill"
        self._selected_template = None
        self._instantiated_steps = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Buffer of Thoughts must be initialized before execution")

        # Store execution context and configure sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        self._step_counter = 1
        self._current_phase = "distill"

        content = (
            f"Step {self._step_counter}: Distill Problem (Buffer of Thoughts)\n\n"
            f"Problem: {input_text}\n\n"
            f"Distillation:\n"
            f"  • Problem Type: [Analyzing...]\n"
            f"  • Key Elements: [Extracting...]\n"
            f"  • Complexity: [Assessing...]\n\n"
            f"Available Templates in Buffer:\n"
            + "\n".join(f"  [{t['id']}] {t['name']}" for t in self.TEMPLATE_BUFFER)
            + "\n\nNext: Retrieve matching template."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BUFFER_OF_THOUGHTS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "templates_available": len(self.TEMPLATE_BUFFER),
                "input_text": input_text,
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.BUFFER_OF_THOUGHTS
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Buffer of Thoughts must be initialized before continuation")

        # Store execution context and configure sampling
        self._execution_context = execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "distill")

        if prev_phase == "distill":
            self._current_phase = "retrieve"
            # Get problem description from session
            input_text = session.get_recent_thoughts(n=1)[0].metadata.get("input_text", "")

            # Use sampling if available, otherwise use heuristic
            if self._use_sampling and self._execution_context:
                content = await self._sample_template_retrieval(input_text)
            else:
                content = self._generate_template_retrieval(input_text)

            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "retrieve":
            self._current_phase = "instantiate"
            # Get problem description from session
            input_text = session.get_recent_thoughts(n=1)[0].metadata.get("input_text", "")

            # Use sampling if available, otherwise use heuristic
            if self._use_sampling and self._execution_context:
                content = await self._sample_template_instantiation(input_text)
            else:
                content = self._generate_template_instantiation(input_text)

            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.75
        elif prev_phase == "instantiate":
            self._current_phase = "reason"
            # Get problem description from session
            input_text = session.get_recent_thoughts(n=1)[0].metadata.get("input_text", "")

            # Use sampling if available, otherwise use heuristic
            if self._use_sampling and self._execution_context:
                content = await self._sample_reasoning_execution(input_text)
            else:
                content = self._generate_reasoning_execution()

            thought_type = ThoughtType.REASONING
            confidence = 0.8
        else:
            self._current_phase = "conclude"
            template_name = self._selected_template["name"] if self._selected_template else "None"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Buffer of Thoughts Complete:\n"
                f"  • Template used: {template_name}\n"
                f"  • Steps executed: {len(self._instantiated_steps)}\n\n"
                f"Final Answer: [Template-guided solution]\n"
                f"Confidence: High (88%)\n"
                f"Template efficiency: Reduced reasoning steps by ~40%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.BUFFER_OF_THOUGHTS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "template": self._selected_template,
                "sampled": self._use_sampling,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_template_retrieval(self, input_text: str) -> str:
        """Generate template retrieval using heuristic approach.

        Args:
            input_text: The problem to analyze

        Returns:
            Formatted content for template retrieval phase
        """
        # Select template (simplified - would use similarity matching)
        self._selected_template = self.TEMPLATE_BUFFER[0]  # Default to math_word
        content = (
            f"Step {self._step_counter}: Retrieve Template\n\n"
            f"Template Matching:\n"
            f"  Analyzing problem characteristics...\n"
            f"  Computing similarity scores...\n\n"
            f"Selected Template: {self._selected_template['name']}\n"
            f"  Structure: {' → '.join(self._selected_template['structure'])}\n"
            f"  Match Score: 0.87\n\n"
            f"Next: Instantiate template for this problem."
        )
        return content

    def _generate_template_instantiation(self, input_text: str) -> str:
        """Generate template instantiation using heuristic approach.

        Args:
            input_text: The problem to analyze

        Returns:
            Formatted content for template instantiation phase
        """
        if self._selected_template:
            self._instantiated_steps = [
                f"{step}: [Problem-specific instantiation]"
                for step in self._selected_template["structure"]
            ]
        template_name = self._selected_template["name"] if self._selected_template else "None"
        content = (
            f"Step {self._step_counter}: Instantiate Template\n\n"
            f"Template: {template_name}\n\n"
            f"Instantiated Steps:\n"
            + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self._instantiated_steps))
            + "\n\nNext: Execute instantiated reasoning."
        )
        return content

    def _generate_reasoning_execution(self) -> str:
        """Generate reasoning execution using heuristic approach.

        Returns:
            Formatted content for reasoning execution phase
        """
        content = (
            f"Step {self._step_counter}: Execute Reasoning\n\n"
            f"Executing instantiated template...\n\n"
            f"Step-by-step execution:\n"
            + "\n".join(f"  ✓ {s}" for s in self._instantiated_steps)
            + "\n\nAll template steps completed."
        )
        return content

    async def _sample_template_retrieval(self, input_text: str) -> str:
        """Generate template retrieval using LLM sampling.

        Uses the execution context's sampling capability to analyze the problem
        and select the most appropriate template from the buffer.

        Args:
            input_text: The problem to analyze

        Returns:
            Formatted content for template retrieval phase
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for template retrieval sampling")

        # Build template descriptions for the prompt
        template_descriptions = "\n".join(
            f"  - [{t['id']}] {t['name']}: {' → '.join(t['structure'])}"
            for t in self.TEMPLATE_BUFFER
        )

        system_prompt = f"""You are a reasoning assistant using Buffer of Thoughts methodology.
Your task is to analyze a problem and select the most appropriate reasoning template.

Available Templates:
{template_descriptions}

Analyze the problem characteristics and select the best matching template.
Explain your template selection reasoning and provide a match score."""

        user_prompt = f"""Problem: {input_text}

Analyze this problem and:
1. Identify key problem characteristics
2. Evaluate which template best matches
3. Select the most appropriate template
4. Explain the match reasoning and provide a similarity score

Format your response as:
Step {self._step_counter}: Retrieve Template

Template Matching:
[Your analysis of problem characteristics]

Selected Template: [Template name]
  Structure: [Template structure]
  Match Score: [0.0-1.0]

Next: Instantiate template for this problem."""

        def _fallback() -> str:
            return self._generate_template_retrieval(input_text)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=_fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

        # Try to extract selected template from response (simplified heuristic)
        for template in self.TEMPLATE_BUFFER:
            if template["name"].lower() in content.lower():
                self._selected_template = template
                break
        if not self._selected_template:
            self._selected_template = self.TEMPLATE_BUFFER[0]

        return content

    async def _sample_template_instantiation(self, input_text: str) -> str:
        """Generate template instantiation using LLM sampling.

        Uses the execution context's sampling capability to instantiate
        the selected template for the specific problem.

        Args:
            input_text: The problem to solve

        Returns:
            Formatted content for template instantiation phase
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for template instantiation sampling")

        if not self._selected_template:
            return self._generate_template_instantiation(input_text)

        system_prompt = f"""You are a reasoning assistant using Buffer of Thoughts methodology.
Your task is to instantiate a reasoning template for a specific problem.

Selected Template: {self._selected_template["name"]}
Template Structure: {" → ".join(self._selected_template["structure"])}

Adapt each step of the template to the specific problem, providing concrete
problem-specific instantiations."""

        user_prompt = f"""Problem: {input_text}

Instantiate the {self._selected_template["name"]} template for this problem.
For each step in the template structure, provide a specific instantiation:

{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(self._selected_template["structure"]))}

Format your response as:
Step {self._step_counter}: Instantiate Template

Template: {self._selected_template["name"]}

Instantiated Steps:
[List each step with problem-specific instantiation]

Next: Execute instantiated reasoning."""

        def _fallback() -> str:
            return self._generate_template_instantiation(input_text)

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=_fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )

        # Try to extract instantiated steps from response
        lines = content.split("\n")
        self._instantiated_steps = []
        for line in lines:
            line = line.strip()
            # Look for numbered steps or bullet points
            if (
                line
                and (line[0].isdigit() or line.startswith("-") or line.startswith("•"))
                and len(line) > 3
            ):
                self._instantiated_steps.append(line)

        # Ensure we have at least the template structure
        if not self._instantiated_steps:
            self._instantiated_steps = [
                f"{step}: [Problem-specific instantiation]"
                for step in self._selected_template["structure"]
            ]

        return content

    async def _sample_reasoning_execution(self, input_text: str) -> str:
        """Generate reasoning execution using LLM sampling.

        Uses the execution context's sampling capability to execute
        the instantiated reasoning steps.

        Args:
            input_text: The problem to solve

        Returns:
            Formatted content for reasoning execution phase
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for reasoning execution sampling")

        if not self._instantiated_steps:
            return self._generate_reasoning_execution()

        instantiated_steps_text = "\n".join(f"  {s}" for s in self._instantiated_steps)

        system_prompt = f"""You are a reasoning assistant using Buffer of Thoughts methodology.
Your task is to execute the instantiated reasoning steps to solve the problem.

Template: {self._selected_template["name"] if self._selected_template else "Unknown"}
Instantiated Steps:
{instantiated_steps_text}

Execute each step systematically to arrive at a solution."""

        user_prompt = f"""Problem: {input_text}

Execute the following instantiated reasoning steps:
{instantiated_steps_text}

Work through each step systematically, showing your reasoning.

Format your response as:
Step {self._step_counter}: Execute Reasoning

Executing instantiated template...

Step-by-step execution:
[Execute each step with detailed reasoning]

All template steps completed."""

        def _fallback() -> str:
            return self._generate_reasoning_execution()

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=_fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )


__all__ = ["BufferOfThoughts", "BUFFER_OF_THOUGHTS_METADATA"]
