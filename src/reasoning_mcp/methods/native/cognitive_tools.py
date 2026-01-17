"""Cognitive Tools reasoning method.

This module implements Cognitive Tools (Ebouky et al. 2025), an advanced reasoning method
that employs modular cognitive operations in an agentic framework. The method provides
four core cognitive tools - analogical, deductive, abductive, and inductive reasoning -
and dynamically selects and applies the most appropriate tool(s) based on the problem type.

The method proceeds through five phases:
1. Tool selection: Identify which cognitive tool(s) to use
2. Application: Apply the selected cognitive operation
3. Evaluation: Assess the result of the cognitive operation
4. Combination: If multiple tools were used, combine insights
5. Conclusion: Synthesize final answer

Based on Ebouky et al. (2025, NeurIPS): "Cognitive Tools: Modular Cognitive Operations
in Agentic Frameworks"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_feedback,
    elicit_selection,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for Cognitive Tools method
COGNITIVE_TOOLS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.COGNITIVE_TOOLS,
    name="Cognitive Tools",
    description="Modular cognitive operations (analogical, deductive, abductive, inductive) "
    "in agentic framework. Dynamically selects and applies appropriate cognitive tools "
    "based on problem type, then evaluates and combines insights.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "modular",
            "cognitive-operations",
            "agentic",
            "multi-tool",
            "adaptive",
            "analogical",
            "deductive",
            "abductive",
            "inductive",
            "tool-selection",
        }
    ),
    complexity=8,  # Advanced complexity - multiple cognitive operations
    supports_branching=True,  # Can explore multiple tools in parallel
    supports_revision=True,  # Can revise tool selection and application
    requires_context=False,  # No special context needed
    min_thoughts=5,  # At least: select + apply + evaluate + combine + conclude
    max_thoughts=20,  # Multiple tools with evaluation and combination
    avg_tokens_per_thought=500,  # Moderate to high - includes cognitive analysis
    best_for=(
        "complex reasoning problems",
        "multi-faceted analysis",
        "problems requiring multiple perspectives",
        "analogical reasoning tasks",
        "logical deduction problems",
        "hypothesis generation",
        "pattern recognition",
        "causal inference",
        "scientific reasoning",
    ),
    not_recommended_for=(
        "simple factual queries",
        "time-critical decisions",
        "single-perspective problems",
        "purely computational tasks",
    ),
)

logger = structlog.get_logger(__name__)


class CognitiveTools(ReasoningMethodBase):
    """Cognitive Tools reasoning method implementation.

    This class implements the Cognitive Tools pattern (Ebouky et al. 2025) where
    the system acts as an agent with access to modular cognitive operations:

    1. **Analogical**: Find similar patterns/problems and transfer solutions
    2. **Deductive**: Apply logical rules to derive conclusions from premises
    3. **Abductive**: Infer the best explanation from observations
    4. **Inductive**: Generalize patterns from specific instances

    The method dynamically selects the most appropriate tool(s) based on problem
    characteristics, applies them, evaluates results, and combines insights.

    Key characteristics:
    - Modular cognitive operations
    - Dynamic tool selection
    - Multi-tool combination capability
    - Evaluation and refinement
    - Advanced complexity (8)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = CognitiveTools()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Why do birds migrate south for winter?"
        ... )
        >>> print(result.content)  # Tool selection phase

        Continue with tool application:
        >>> application = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Apply selected cognitive tools"
        ... )
        >>> print(application.type)  # ThoughtType.REASONING

        Continue with evaluation:
        >>> evaluation = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=application,
        ...     guidance="Evaluate results"
        ... )
        >>> print(evaluation.type)  # ThoughtType.VERIFICATION

        Combine insights (if multiple tools):
        >>> combination = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=evaluation,
        ...     guidance="Combine insights"
        ... )
        >>> print(combination.type)  # ThoughtType.SYNTHESIS

        Conclude:
        >>> conclusion = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=combination,
        ...     guidance="Synthesize final answer"
        ... )
        >>> print(conclusion.type)  # ThoughtType.CONCLUSION
    """

    # Available cognitive tools
    ANALOGICAL = "analogical"
    DEDUCTIVE = "deductive"
    ABDUCTIVE = "abductive"
    INDUCTIVE = "inductive"

    ALL_TOOLS = frozenset({ANALOGICAL, DEDUCTIVE, ABDUCTIVE, INDUCTIVE})

    # Enable LLM sampling for cognitive operations
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Cognitive Tools method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "select_tool"  # select_tool, apply, evaluate, combine, conclude
        self._selected_tools: list[str] = []
        self._tool_results: dict[str, Any] = {}
        self.enable_elicitation = enable_elicitation
        self._ctx: Any = None
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.COGNITIVE_TOOLS

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return COGNITIVE_TOOLS_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return COGNITIVE_TOOLS_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Cognitive Tools method for execution.
        Resets counters, state, and tool tracking for a fresh reasoning session.

        Examples:
            >>> method = CognitiveTools()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "select_tool"
            >>> assert len(method._selected_tools) == 0
            >>> assert len(method._tool_results) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "select_tool"
        self._selected_tools = []
        self._tool_results = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Execute the Cognitive Tools method.

        This method initiates the tool selection phase, analyzing the problem
        to determine which cognitive tool(s) would be most appropriate.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include tool_preference)
            execution_context: Optional ExecutionContext for elicitation

        Returns:
            A ThoughtNode representing the tool selection phase

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = CognitiveTools()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="What explains the correlation between coffee and alertness?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.COGNITIVE_TOOLS
            >>> assert "selected_tools" in thought.metadata
            >>> assert thought.metadata["phase"] == "select_tool"
        """
        if not self._initialized:
            raise RuntimeError("Cognitive Tools method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "select_tool"
        self._selected_tools = []
        self._tool_results = {}

        # Tool selection based on problem analysis
        selected_tools, reasoning = await self._select_cognitive_tools(input_text, context)
        self._selected_tools = selected_tools

        # Optional elicitation: ask user which tool to apply
        if self.enable_elicitation and self._ctx and len(selected_tools) > 1:
            try:
                tool_options = [
                    {"id": tool, "label": f"{tool.capitalize()} reasoning"}
                    for tool in selected_tools
                ]
                elicit_config = ElicitationConfig(
                    timeout=60, required=False, default_on_timeout=None
                )
                selection = await elicit_selection(
                    self._ctx,
                    (
                        "Multiple cognitive tools were identified for this problem. "
                        "Which should we prioritize?"
                    ),
                    tool_options,
                    config=elicit_config,
                )
                # Move selected tool to front
                if selection.selected in selected_tools:
                    selected_tools.remove(selection.selected)
                    selected_tools.insert(0, selection.selected)
                    self._selected_tools = selected_tools
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Elicitation failed - continue without it
                pass
            except Exception as e:
                logger.error(
                    "elicitation_unexpected_error",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                # Re-raise to avoid masking unexpected errors
                raise

        # Generate tool selection content
        content = self._generate_tool_selection(input_text, selected_tools, reasoning, context)

        # Initial quality score (moderate - will improve through application)
        initial_quality = 0.5

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COGNITIVE_TOOLS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Moderate initial confidence
            quality_score=initial_quality,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "cognitive_tools",
                "phase": self._current_phase,
                "selected_tools": selected_tools,
                "selection_reasoning": reasoning,
                "tool_results": {},
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.COGNITIVE_TOOLS

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
        """Continue reasoning from a previous thought.

        This method implements the Cognitive Tools phase cycle:
        - If previous was select_tool: apply selected cognitive tools
        - If previous was apply: evaluate tool results
        - If previous was evaluate: combine insights (if multiple tools)
        - If previous was combine or evaluate (single tool): conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional ExecutionContext for elicitation

        Returns:
            A new ThoughtNode continuing the cognitive tools process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = CognitiveTools()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Solve problem X")
            >>> apply = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert apply.type == ThoughtType.REASONING
            >>> assert apply.metadata["phase"] == "apply"
            >>>
            >>> evaluate = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=apply
            ... )
            >>> assert evaluate.type == ThoughtType.VERIFICATION
            >>> assert evaluate.metadata["phase"] == "evaluate"
        """
        if not self._initialized:
            raise RuntimeError("Cognitive Tools method must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        # Increment step counter
        self._step_counter += 1

        # Get previous phase and tool data
        prev_phase = previous_thought.metadata.get("phase", "select_tool")
        selected_tools = previous_thought.metadata.get("selected_tools", [])
        tool_results = previous_thought.metadata.get("tool_results", {})

        # Optional elicitation: ask for feedback during evaluation phase
        elicited_feedback = ""
        if self.enable_elicitation and self._ctx and prev_phase == "apply" and not guidance:
            try:
                elicit_config = ElicitationConfig(
                    timeout=60, required=False, default_on_timeout=None
                )
                feedback = await elicit_feedback(
                    self._ctx,
                    (
                        f"The cognitive tool(s) have been applied: {', '.join(selected_tools)}. "
                        "Do you have any feedback or observations about the results?"
                    ),
                    config=elicit_config,
                )
                if feedback.feedback:
                    elicited_feedback = f"\n\n[User Feedback]: {feedback.feedback}"
                    guidance = feedback.feedback
                    session.metrics.elicitations_made += 1
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="continue_reasoning",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Elicitation failed - continue without it
                pass
            except Exception as e:
                logger.error(
                    "elicitation_unexpected_error",
                    method="continue_reasoning",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                # Re-raise to avoid masking unexpected errors
                raise

        # Determine next phase and generate content
        if prev_phase == "select_tool":
            # Next: apply selected tools
            thought_type, content, quality, confidence = await self._transition_to_apply(
                previous_thought, selected_tools, guidance, context
            )
            # Store tool results
            self._tool_results = {tool: f"Result from {tool}" for tool in selected_tools}

        elif prev_phase == "apply":
            # Next: evaluate tool results
            thought_type, content, quality, confidence = self._transition_to_evaluate(
                previous_thought, selected_tools, tool_results, guidance, context
            )
            # Add elicited feedback to content if present
            if elicited_feedback:
                content += elicited_feedback

        elif prev_phase == "evaluate":
            # Next: combine if multiple tools, otherwise conclude
            if len(selected_tools) > 1:
                thought_type, content, quality, confidence = self._transition_to_combine(
                    previous_thought, selected_tools, tool_results, guidance, context
                )
            else:
                thought_type, content, quality, confidence = self._transition_to_conclude(
                    previous_thought, selected_tools, tool_results, guidance, context
                )

        elif prev_phase == "combine":
            # Next: conclude
            thought_type, content, quality, confidence = self._transition_to_conclude(
                previous_thought, selected_tools, tool_results, guidance, context
            )

        else:
            # Fallback to conclusion
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = self._generate_conclusion(
                previous_thought, selected_tools, tool_results, guidance, context
            )
            quality = 0.9
            confidence = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.COGNITIVE_TOOLS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality,
            metadata={
                "phase": self._current_phase,
                "selected_tools": selected_tools,
                "tool_results": self._tool_results if self._tool_results else tool_results,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "cognitive_tools",
                "previous_phase": prev_phase,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Cognitive Tools, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = CognitiveTools()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _select_cognitive_tools(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> tuple[list[str], str]:
        """Select appropriate cognitive tools based on problem analysis.

        Analyzes the problem to determine which cognitive tool(s) would be
        most effective. Can select multiple tools for multi-faceted problems.

        Args:
            input_text: The problem or question to analyze
            context: Optional context that may include tool preferences

        Returns:
            Tuple of (selected tools list, selection reasoning)
        """
        # Check for explicit tool preference in context
        if context and "tool_preference" in context:
            pref = context["tool_preference"]
            if pref in self.ALL_TOOLS:
                return [pref], f"Tool explicitly requested: {pref}"

        # Try LLM sampling for tool selection
        if self._use_sampling and self._execution_context:
            prompt = (
                "Analyze the following problem and select the most appropriate "
                "cognitive tool(s) to apply.\n\n"
                f"Problem: {input_text}\n\n"
                "Available cognitive tools:\n"
                "1. ANALOGICAL - Find similar patterns/problems and transfer solutions\n"
                "2. DEDUCTIVE - Apply logical rules to derive conclusions from premises\n"
                "3. ABDUCTIVE - Infer the best explanation from observations\n"
                "4. INDUCTIVE - Generalize patterns from specific instances\n\n"
                "You may select one or multiple tools. For each tool you select, "
                "explain why it's appropriate.\n\n"
                "Provide your response in the following format:\n"
                "TOOLS: [comma-separated list of tool names in lowercase]\n"
                "REASONING: [explanation of why these tools are appropriate]"
            )

            system_prompt = (
                "You are an expert cognitive scientist analyzing problems to determine "
                "the best reasoning approach."
            )

            response_text = await self._sample_with_fallback(
                prompt,
                fallback_generator=lambda: "",
                system_prompt=system_prompt,
                temperature=0.7,
            )

            if response_text:
                selected_tools, reasoning = self._parse_tool_selection_response(response_text)

                if selected_tools:
                    return selected_tools, reasoning

        # Fallback: use heuristic keyword-based selection
        text_lower = input_text.lower()
        selected: list[str] = []
        reasoning_parts: list[str] = []

        # Analogical: look for similarity, patterns, comparison
        if any(
            kw in text_lower
            for kw in [
                "similar",
                "like",
                "pattern",
                "compare",
                "analogy",
                "reminds",
                "parallel",
            ]
        ):
            selected.append(self.ANALOGICAL)
            reasoning_parts.append("Problem involves finding similar patterns or analogies")

        # Deductive: look for logic, rules, if-then, must
        if any(
            kw in text_lower
            for kw in ["if", "then", "must", "follows", "therefore", "logic", "rule", "given"]
        ):
            selected.append(self.DEDUCTIVE)
            reasoning_parts.append("Problem involves logical deduction from premises")

        # Abductive: look for explain, why, cause, reason, best explanation
        if any(
            kw in text_lower
            for kw in [
                "why",
                "explain",
                "cause",
                "reason",
                "because",
                "explanation",
                "hypothesis",
            ]
        ):
            selected.append(self.ABDUCTIVE)
            reasoning_parts.append("Problem requires finding the best explanation")

        # Inductive: look for generalize, pattern, examples, instances
        if any(
            kw in text_lower
            for kw in [
                "generalize",
                "pattern",
                "always",
                "usually",
                "trend",
                "instances",
                "examples",
            ]
        ):
            selected.append(self.INDUCTIVE)
            reasoning_parts.append("Problem involves generalizing from specific instances")

        # Default: if no clear match, use abductive (most general)
        if not selected:
            selected.append(self.ABDUCTIVE)
            reasoning_parts.append("Using abductive reasoning as general-purpose tool")

        reasoning = "; ".join(reasoning_parts)
        return selected, reasoning

    async def _apply_tool_with_sampling(
        self,
        tool: str,
        input_text: str,
        guidance: str | None,
    ) -> str:
        """Apply a cognitive tool using LLM sampling.

        Args:
            tool: The cognitive tool to apply
            input_text: The problem or question
            guidance: Optional guidance

        Returns:
            The tool application result as formatted string
        """
        tool_descriptions = {
            self.ANALOGICAL: (
                "find similar patterns, problems, or situations and transfer solutions"
            ),
            self.DEDUCTIVE: "apply logical rules to derive conclusions from premises",
            self.ABDUCTIVE: "infer the best explanation from observations",
            self.INDUCTIVE: "generalize patterns from specific instances",
        }

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        prompt = f"""Apply {tool.upper()} reasoning to the following problem:

Problem: {input_text}{guidance_text}

Your task is to {tool_descriptions[tool]}.

Provide a detailed analysis using {tool} reasoning. Include:
1. Your reasoning process
2. Key insights or conclusions
3. Confidence in your analysis

Format your response clearly with headings and bullet points."""

        system_prompt = (
            f"You are an expert in {tool} reasoning. Provide thorough, insightful analysis."
        )

        def fallback_tool_result() -> str:
            return (
                f"**{tool.capitalize()} Reasoning:**\n"
                f"[LLM would apply {tool} reasoning to analyze the problem]\n"
                f"- {tool_descriptions[tool]}\n\n"
            )

        response_text = await self._sample_with_fallback(
            prompt,
            fallback_generator=fallback_tool_result,
            system_prompt=system_prompt,
            temperature=0.7,
        )

        return f"**{tool.capitalize()} Reasoning:**\n{response_text}\n\n"

    def _parse_tool_selection_response(self, response_text: str) -> tuple[list[str], str]:
        """Parse the LLM response for tool selection.

        Args:
            response_text: The LLM response text

        Returns:
            Tuple of (selected tools list, reasoning)
        """
        selected_tools: list[str] = []
        reasoning = ""

        try:
            lines = response_text.strip().split("\n")
            for line in lines:
                if line.startswith("TOOLS:"):
                    tools_str = line.replace("TOOLS:", "").strip()
                    tool_names = [t.strip().lower() for t in tools_str.split(",")]
                    # Validate and filter to known tools
                    selected_tools = [t for t in tool_names if t in self.ALL_TOOLS]
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
        except (ValueError, AttributeError) as e:
            logger.warning(
                "parsing_failed",
                method="_parse_tool_selection_response",
                error_type=type(e).__name__,
                error=str(e),
            )
            # If parsing fails, return empty results (will fall back to heuristic)
            return [], ""
        except Exception as e:
            logger.error(
                "parsing_unexpected_error",
                method="_parse_tool_selection_response",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            # Re-raise to avoid masking unexpected errors
            raise

        return selected_tools, reasoning

    def _generate_tool_selection(
        self,
        input_text: str,
        selected_tools: list[str],
        reasoning: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for the tool selection phase.

        Args:
            input_text: The problem or question
            selected_tools: List of selected cognitive tools
            reasoning: Reasoning for tool selection
            context: Optional context

        Returns:
            Content string for the tool selection thought
        """
        tools_desc = {
            self.ANALOGICAL: "finding similar patterns and transferring solutions",
            self.DEDUCTIVE: "logical deduction from premises to conclusions",
            self.ABDUCTIVE: "inferring the best explanation from observations",
            self.INDUCTIVE: "generalizing patterns from specific instances",
        }

        tools_str = ", ".join(selected_tools)
        tools_detail = "\n".join(
            f"  - {tool.capitalize()}: {tools_desc[tool]}" for tool in selected_tools
        )

        return (
            f"Step {self._step_counter}: Cognitive Tool Selection (Phase 1/5)\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing problem characteristics to select appropriate cognitive tools...\n\n"
            f"Selection Reasoning:\n{reasoning}\n\n"
            f"Selected Tool(s): {tools_str}\n\n"
            f"Tool Descriptions:\n{tools_detail}\n\n"
            f"Next: I will apply the selected cognitive tool(s) to analyze the problem."
        )

    async def _transition_to_apply(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to apply phase.

        Args:
            previous_thought: The tool selection thought
            selected_tools: List of selected tools
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "apply"
        thought_type = ThoughtType.REASONING
        content = await self._generate_application(
            previous_thought, selected_tools, guidance, context
        )
        quality_score = 0.7
        confidence = 0.75

        return thought_type, content, quality_score, confidence

    def _transition_to_evaluate(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to evaluate phase.

        Args:
            previous_thought: The application thought
            selected_tools: List of selected tools
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "evaluate"
        thought_type = ThoughtType.VERIFICATION
        content = self._generate_evaluation(
            previous_thought, selected_tools, tool_results, guidance, context
        )
        quality_score = 0.8
        confidence = 0.8

        return thought_type, content, quality_score, confidence

    def _transition_to_combine(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to combine phase (for multiple tools).

        Args:
            previous_thought: The evaluation thought
            selected_tools: List of selected tools
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "combine"
        thought_type = ThoughtType.SYNTHESIS
        content = self._generate_combination(
            previous_thought, selected_tools, tool_results, guidance, context
        )
        quality_score = 0.85
        confidence = 0.85

        return thought_type, content, quality_score, confidence

    def _transition_to_conclude(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float, float]:
        """Transition to conclude phase.

        Args:
            previous_thought: The evaluation or combination thought
            selected_tools: List of selected tools
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, quality_score, confidence)
        """
        self._current_phase = "conclude"
        thought_type = ThoughtType.CONCLUSION
        content = self._generate_conclusion(
            previous_thought, selected_tools, tool_results, guidance, context
        )
        quality_score = 0.9
        confidence = 0.9

        return thought_type, content, quality_score, confidence

    async def _generate_application(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for applying cognitive tools.

        Args:
            previous_thought: The tool selection thought
            selected_tools: List of tools to apply
            guidance: Optional guidance
            context: Optional context

        Returns:
            Content string for the application thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        input_text = previous_thought.metadata.get("input", "")

        content = (
            f"Step {self._step_counter}: Cognitive Tool Application (Phase 2/5)\n\n"
            f"Applying selected cognitive tool(s) from Step {previous_thought.step_number}...\n\n"
        )

        # Try LLM sampling for each tool application
        for tool in selected_tools:
            tool_result = ""

            if self._use_sampling and self._execution_context:
                can_sample = (
                    hasattr(self._execution_context, "can_sample")
                    and self._execution_context.can_sample
                )
                if can_sample:
                    try:
                        tool_result = await self._apply_tool_with_sampling(
                            tool, input_text, guidance
                        )
                    except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                        logger.warning(
                            "tool_sampling_failed",
                            method="_generate_application",
                            tool=tool,
                            error_type=type(e).__name__,
                            error=str(e),
                        )
                        # Fall through to heuristic method
                        pass
                    except Exception as e:
                        logger.error(
                            "tool_sampling_unexpected_error",
                            method="_generate_application",
                            tool=tool,
                            error_type=type(e).__name__,
                            error=str(e),
                            exc_info=True,
                        )
                        # Re-raise to avoid masking unexpected errors
                        raise

            # Fallback to heuristic placeholders if sampling failed
            if not tool_result:
                if tool == self.ANALOGICAL:
                    tool_result = (
                        "**Analogical Reasoning:**\n"
                        "[LLM would identify similar patterns, problems, or situations]\n"
                        "- Finding analogous cases...\n"
                        "- Mapping structural similarities...\n"
                        "- Transferring solutions from analogous domains...\n\n"
                    )
                elif tool == self.DEDUCTIVE:
                    tool_result = (
                        "**Deductive Reasoning:**\n"
                        "[LLM would apply logical rules to derive conclusions]\n"
                        "- Identifying premises and axioms...\n"
                        "- Applying logical inference rules...\n"
                        "- Deriving necessary conclusions...\n\n"
                    )
                elif tool == self.ABDUCTIVE:
                    tool_result = (
                        "**Abductive Reasoning:**\n"
                        "[LLM would infer the best explanation from observations]\n"
                        "- Observing key facts and patterns...\n"
                        "- Generating candidate explanations...\n"
                        "- Selecting the most plausible explanation...\n\n"
                    )
                elif tool == self.INDUCTIVE:
                    tool_result = (
                        "**Inductive Reasoning:**\n"
                        "[LLM would generalize patterns from specific instances]\n"
                        "- Examining specific instances and examples...\n"
                        "- Identifying common patterns and regularities...\n"
                        "- Formulating general principles...\n\n"
                    )

            content += tool_result

        content += f"Tool application complete. Next: Evaluate the results.{guidance_text}"

        return content

    def _generate_evaluation(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for evaluating tool results.

        Args:
            previous_thought: The application thought
            selected_tools: List of tools that were applied
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Content string for the evaluation thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Result Evaluation (Phase 3/5)\n\n"
            f"Evaluating the results from cognitive tool application in "
            f"Step {previous_thought.step_number}...\n\n"
        )

        for tool in selected_tools:
            content += (
                f"**{tool.capitalize()} Tool Evaluation:**\n"
                f"[LLM would assess quality, coherence, and usefulness of results]\n"
                f"- Quality: [Assessment of reasoning quality]\n"
                f"- Relevance: [How well it addresses the problem]\n"
                f"- Confidence: [Confidence in the conclusions]\n\n"
            )

        if len(selected_tools) > 1:
            content += "Multiple tools applied. Next: Combine insights for comprehensive answer."
        else:
            content += "Single tool applied. Next: Synthesize final conclusion."

        content += guidance_text

        return content

    def _generate_combination(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for combining insights from multiple tools.

        Args:
            previous_thought: The evaluation thought
            selected_tools: List of tools that were applied
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Content string for the combination thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        tools_str = ", ".join(selected_tools)

        content = (
            f"Step {self._step_counter}: Insight Combination (Phase 4/5)\n\n"
            f"Combining insights from {len(selected_tools)} cognitive tools: {tools_str}\n\n"
            f"[LLM would integrate insights from multiple cognitive perspectives]\n\n"
            f"Integration Strategy:\n"
            f"- Identifying complementary insights across tools\n"
            f"- Resolving any contradictions or tensions\n"
            f"- Synthesizing a coherent multi-perspective view\n\n"
            f"Combined Insights:\n"
            f"[LLM would present unified understanding from all tools]\n\n"
            f"Next: Formulate final conclusion based on combined insights.{guidance_text}"
        )

        return content

    def _generate_conclusion(
        self,
        previous_thought: ThoughtNode,
        selected_tools: list[str],
        tool_results: dict[str, Any],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate content for the final conclusion.

        Args:
            previous_thought: The evaluation or combination thought
            selected_tools: List of tools that were applied
            tool_results: Results from tool applications
            guidance: Optional guidance
            context: Optional context

        Returns:
            Content string for the conclusion thought
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        tools_str = ", ".join(selected_tools)
        phase_num = "5/5" if len(selected_tools) > 1 else "4/4"

        content = (
            f"Step {self._step_counter}: Final Conclusion (Phase {phase_num})\n\n"
            f"Synthesizing final answer based on cognitive tool analysis "
            f"from Step {previous_thought.step_number}...\n\n"
            f"Cognitive Tools Used: {tools_str}\n\n"
            f"Final Answer:\n"
            f"[LLM would provide comprehensive answer incorporating all cognitive insights]\n\n"
            f"Reasoning Summary:\n"
            f"[LLM would summarize the key reasoning steps and insights from each tool]\n\n"
            f"Confidence: High (based on multi-perspective cognitive analysis)\n\n"
            f"This conclusion integrates insights from {len(selected_tools)} cognitive tool(s) "
            f"for robust, multi-faceted reasoning.{guidance_text}"
        )

        return content


__all__ = ["CognitiveTools", "COGNITIVE_TOOLS_METADATA"]
