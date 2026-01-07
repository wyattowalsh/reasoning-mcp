"""ReAct (Reasoning and Acting) method implementation.

ReAct interleaves reasoning steps with actions and observations, following a
Reason → Act → Observe cycle until a conclusion is reached. This method is
particularly effective for problems requiring external tool interactions or
multi-step exploration.

References:
    Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023).
    ReAct: Synergizing Reasoning and Acting in Language Models.
    arXiv preprint arXiv:2210.03629.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for the ReAct method
REACT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.REACT,
    name="ReAct (Reasoning and Acting)",
    description=(
        "Interleaves reasoning, action, and observation in a cyclic pattern. "
        "Each cycle begins with reasoning about the current state, proposes an "
        "action to take, and then observes the results before continuing. "
        "Effective for multi-step problems requiring external interactions."
    ),
    category=MethodCategory.CORE,
    tags=frozenset({
        "iterative",
        "action-oriented",
        "tool-use",
        "observation",
        "multi-step",
        "interactive",
    }),
    complexity=5,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,  # At least one Reason → Act → Observe cycle
    max_thoughts=0,  # Unlimited - continues until conclusion
    avg_tokens_per_thought=400,
    best_for=(
        "Problems requiring external tool interactions",
        "Multi-step exploration and information gathering",
        "Situations needing iterative refinement based on feedback",
        "Tasks with observable state changes",
        "Information retrieval and synthesis problems",
        "Debugging and diagnostic reasoning",
    ),
    not_recommended_for=(
        "Simple one-shot reasoning tasks",
        "Pure mathematical or logical proofs",
        "Problems with no external actions possible",
        "Real-time critical decision making",
    ),
)


class ReActMethod:
    """ReAct (Reasoning and Acting) reasoning method.

    ReAct follows a structured cycle of:
    1. Reasoning - Analyze the current state and determine what to do next
    2. Action - Specify an action to take or information to gather
    3. Observation - Record the results or observations from the action

    This cycle continues until the reasoning determines a conclusion has been reached.
    The method is particularly powerful for problems that benefit from iterative
    exploration and learning from intermediate results.

    Examples:
        Basic usage:
        >>> method = ReActMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can I find the population of Tokyo?"
        ... )
        >>> assert result.type == ThoughtType.CONCLUSION

        With action context:
        >>> context = {
        ...     "available_tools": ["search", "calculate", "lookup"],
        ...     "max_cycles": 5
        ... }
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Calculate the area of a circle with radius 5",
        ...     context=context
        ... )
    """

    def __init__(self) -> None:
        """Initialize the ReAct method."""
        self._is_initialized = False
        self._max_cycles = 10  # Default maximum reasoning cycles

    @property
    def identifier(self) -> str:
        """Unique identifier for this method."""
        return str(MethodIdentifier.REACT)

    @property
    def name(self) -> str:
        """Human-readable name for the method."""
        return REACT_METADATA.name

    @property
    def description(self) -> str:
        """Brief description of what the method does."""
        return REACT_METADATA.description

    @property
    def category(self) -> str:
        """Category this method belongs to."""
        return str(REACT_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the method (load resources, etc.)."""
        # No special initialization needed for ReAct
        self._is_initialized = True

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the ReAct reasoning method.

        This method generates a series of Reason → Act → Observe cycles until
        a conclusion is reached. Each cycle:
        1. Reasons about the current state and what action to take
        2. Proposes an action (simulated for now)
        3. Records observations from the action result
        4. Evaluates if conclusion is reached

        Args:
            session: The current reasoning session
            input_text: The input problem or question to reason about
            context: Optional context including:
                - max_cycles: Maximum number of R→A→O cycles (default: 10)
                - available_tools: List of tools that could be used (simulated)
                - initial_observations: Initial facts or observations to start with

        Returns:
            A ThoughtNode representing the final conclusion

        Raises:
            ValueError: If session is not active
        """
        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Extract context parameters
        ctx = context or {}
        max_cycles = ctx.get("max_cycles", self._max_cycles)
        available_tools = ctx.get("available_tools", ["search", "lookup", "calculate"])
        initial_obs = ctx.get("initial_observations", [])

        # Create initial thought analyzing the problem
        initial_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.REACT,
            content=f"Problem Analysis: {input_text}\n\nI will use the ReAct method to solve this problem through iterative reasoning, action, and observation cycles.",
            confidence=0.7,
            depth=0,
            step_number=1,
            metadata={
                "cycle": 0,
                "phase": "initial",
                "available_tools": available_tools,
            },
        )
        session.add_thought(initial_thought)

        # Track the reasoning chain
        current_parent_id = initial_thought.id
        step_number = 2
        cycle = 1
        conclusion_reached = False

        # Add initial observations if provided
        if initial_obs:
            obs_content = "Initial Observations:\n" + "\n".join(f"- {obs}" for obs in initial_obs)
            obs_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.OBSERVATION,
                method_id=MethodIdentifier.REACT,
                content=obs_content,
                parent_id=current_parent_id,
                confidence=0.9,
                depth=1,
                step_number=step_number,
                metadata={"cycle": 0, "phase": "initial_observation"},
            )
            session.add_thought(obs_thought)
            current_parent_id = obs_thought.id
            step_number += 1

        # Main ReAct loop: Reason → Act → Observe
        while cycle <= max_cycles and not conclusion_reached:
            # Phase 1: Reasoning
            reasoning_thought = self._create_reasoning_thought(
                input_text=input_text,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
                session=session,
            )
            session.add_thought(reasoning_thought)
            current_parent_id = reasoning_thought.id
            step_number += 1

            # Check if reasoning determined we have enough information
            if self._should_conclude(reasoning_thought, cycle, max_cycles):
                conclusion_reached = True
                break

            # Phase 2: Action
            action_thought = self._create_action_thought(
                reasoning_thought=reasoning_thought,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
                available_tools=available_tools,
            )
            session.add_thought(action_thought)
            current_parent_id = action_thought.id
            step_number += 1

            # Phase 3: Observation
            observation_thought = self._create_observation_thought(
                action_thought=action_thought,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
            )
            session.add_thought(observation_thought)
            current_parent_id = observation_thought.id
            step_number += 1

            cycle += 1

        # Create final conclusion
        conclusion = self._create_conclusion(
            input_text=input_text,
            session=session,
            parent_id=current_parent_id,
            step_number=step_number,
            total_cycles=cycle - 1,
        )
        session.add_thought(conclusion)

        return conclusion

    def _create_reasoning_thought(
        self,
        input_text: str,
        cycle: int,
        step_number: int,
        parent_id: str,
        session: Session,
    ) -> ThoughtNode:
        """Create a reasoning thought analyzing the current state.

        Args:
            input_text: The original problem
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID
            session: Current session

        Returns:
            A ThoughtNode with REASONING type
        """
        # Generate reasoning based on cycle
        if cycle == 1:
            content = (
                f"Thought {cycle}: To solve '{input_text}', I need to break this down. "
                "Let me start by identifying what information I need and what actions "
                "would help me gather that information."
            )
            confidence = 0.7
        elif cycle == 2:
            content = (
                f"Thought {cycle}: Based on my previous observation, I should now "
                "refine my approach. I'll consider what additional information "
                "would help me reach a conclusion."
            )
            confidence = 0.75
        elif cycle >= 3:
            # Check if we have enough information
            recent_thoughts = session.get_recent_thoughts(n=5)
            observation_count = sum(1 for t in recent_thoughts if t.type == ThoughtType.OBSERVATION)

            if observation_count >= 2:
                content = (
                    f"Thought {cycle}: I've gathered sufficient information through "
                    f"{observation_count} observations. I can now synthesize these "
                    "findings to form a conclusion."
                )
                confidence = 0.9
            else:
                content = (
                    f"Thought {cycle}: I need to gather more specific information. "
                    "Let me identify the most critical missing piece."
                )
                confidence = 0.75
        else:
            content = f"Thought {cycle}: Continuing analysis..."
            confidence = 0.7

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=confidence,
            depth=session.current_depth + 1,
            step_number=step_number,
            metadata={"cycle": cycle, "phase": "reasoning"},
        )

    def _create_action_thought(
        self,
        reasoning_thought: ThoughtNode,
        cycle: int,
        step_number: int,
        parent_id: str,
        available_tools: list[str],
    ) -> ThoughtNode:
        """Create an action thought specifying what action to take.

        Args:
            reasoning_thought: The reasoning thought that led to this action
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID
            available_tools: List of available tools/actions

        Returns:
            A ThoughtNode with ACTION type
        """
        # Simulate action selection based on cycle
        if cycle == 1:
            tool = available_tools[0] if available_tools else "search"
            content = (
                f"Action {cycle}: I will use '{tool}' to gather initial information. "
                "This should help me understand the problem space better."
            )
        elif cycle == 2:
            tool = available_tools[1] if len(available_tools) > 1 else "lookup"
            content = (
                f"Action {cycle}: I will use '{tool}' to get more specific details. "
                "This will help me verify and refine my understanding."
            )
        else:
            tool = available_tools[-1] if available_tools else "calculate"
            content = (
                f"Action {cycle}: I will use '{tool}' to validate and finalize "
                "my findings from the previous observations."
            )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.ACTION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=0.8,
            depth=reasoning_thought.depth + 1,
            step_number=step_number,
            metadata={
                "cycle": cycle,
                "phase": "action",
                "tool_used": tool,
                "simulated": True,
            },
        )

    def _create_observation_thought(
        self,
        action_thought: ThoughtNode,
        cycle: int,
        step_number: int,
        parent_id: str,
    ) -> ThoughtNode:
        """Create an observation thought recording action results.

        Args:
            action_thought: The action thought that was executed
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID

        Returns:
            A ThoughtNode with OBSERVATION type
        """
        # Simulate observations based on the action
        tool_used = action_thought.metadata.get("tool_used", "unknown")

        if cycle == 1:
            content = (
                f"Observation {cycle}: The '{tool_used}' action provided useful "
                "initial context. I now have a better understanding of the problem "
                "domain and can identify what additional information is needed."
            )
            confidence = 0.75
        elif cycle == 2:
            content = (
                f"Observation {cycle}: The '{tool_used}' action revealed important "
                "details. This information builds on my previous findings and helps "
                "narrow down the solution space."
            )
            confidence = 0.85
        else:
            content = (
                f"Observation {cycle}: The '{tool_used}' action confirmed my "
                "hypothesis and provided the final pieces of information needed. "
                "I now have sufficient data to form a conclusion."
            )
            confidence = 0.9

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.OBSERVATION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=confidence,
            depth=action_thought.depth + 1,
            step_number=step_number,
            metadata={"cycle": cycle, "phase": "observation"},
        )

    def _should_conclude(
        self,
        reasoning_thought: ThoughtNode,
        cycle: int,
        max_cycles: int,
    ) -> bool:
        """Determine if we should conclude the reasoning process.

        Args:
            reasoning_thought: The latest reasoning thought
            cycle: Current cycle number
            max_cycles: Maximum allowed cycles

        Returns:
            True if we should conclude, False to continue
        """
        # Conclude if we've reached max cycles
        if cycle >= max_cycles:
            return True

        # Conclude if reasoning confidence is high and we're past cycle 2
        if cycle >= 3 and reasoning_thought.confidence >= 0.85:
            return True

        # Check if reasoning content suggests conclusion readiness
        content_lower = reasoning_thought.content.lower()
        conclusion_indicators = [
            "sufficient information",
            "can now synthesize",
            "form a conclusion",
            "ready to conclude",
        ]

        return any(indicator in content_lower for indicator in conclusion_indicators)

    def _create_conclusion(
        self,
        input_text: str,
        session: Session,
        parent_id: str,
        step_number: int,
        total_cycles: int,
    ) -> ThoughtNode:
        """Create the final conclusion thought.

        Args:
            input_text: The original problem
            session: Current session
            parent_id: Parent thought ID
            step_number: Current step number
            total_cycles: Total number of cycles completed

        Returns:
            A ThoughtNode with CONCLUSION type
        """
        # Gather insights from the reasoning chain
        recent_thoughts = session.get_recent_thoughts(n=10)
        observations = [t for t in recent_thoughts if t.type == ThoughtType.OBSERVATION]
        actions = [t for t in recent_thoughts if t.type == ThoughtType.ACTION]

        content = (
            f"Conclusion: After {total_cycles} cycles of reasoning, action, and observation, "
            f"I have reached a conclusion for: '{input_text}'\n\n"
            f"Through {len(actions)} actions and {len(observations)} observations, "
            "I have gathered and synthesized the necessary information. "
            "The iterative ReAct process allowed me to progressively refine my "
            "understanding and validate findings at each step.\n\n"
            "The solution emerges from the systematic exploration of the problem space, "
            "where each action informed the next reasoning step, creating a robust "
            "chain of evidence-based reasoning."
        )

        # Calculate conclusion confidence based on observations
        if observations:
            avg_obs_confidence = sum(t.confidence for t in observations) / len(observations)
            conclusion_confidence = min(0.95, avg_obs_confidence + 0.05)
        else:
            conclusion_confidence = 0.8

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=conclusion_confidence,
            quality_score=0.85,
            depth=session.current_depth + 1,
            step_number=step_number,
            metadata={
                "total_cycles": total_cycles,
                "total_actions": len(actions),
                "total_observations": len(observations),
                "phase": "conclusion",
            },
        )

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        This allows branching or extending the ReAct reasoning chain
        with additional cycles if needed.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the reasoning
        """
        # Determine what phase to continue from
        phase = previous_thought.metadata.get("phase", "unknown")
        cycle = previous_thought.metadata.get("cycle", 0)

        if phase == "conclusion":
            # If continuing from conclusion, start a new branch
            return ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.REACT,
                content=f"Branching: {guidance or 'Exploring alternative approach based on previous conclusion.'}",
                parent_id=previous_thought.id,
                branch_id=f"branch_{str(uuid4())[:8]}",
                confidence=0.7,
                depth=previous_thought.depth + 1,
                step_number=previous_thought.step_number + 1,
                metadata={"cycle": cycle + 1, "phase": "branch_reasoning"},
            )

        # Otherwise, continue the cycle
        next_step = previous_thought.step_number + 1

        if phase == "reasoning":
            # After reasoning, create action
            return self._create_action_thought(
                reasoning_thought=previous_thought,
                cycle=cycle,
                step_number=next_step,
                parent_id=previous_thought.id,
                available_tools=context.get("available_tools", []) if context else [],
            )
        elif phase == "action":
            # After action, create observation
            return self._create_observation_thought(
                action_thought=previous_thought,
                cycle=cycle,
                step_number=next_step,
                parent_id=previous_thought.id,
            )
        else:
            # Default: create new reasoning thought
            return self._create_reasoning_thought(
                input_text=guidance or "Continue reasoning",
                cycle=cycle + 1,
                step_number=next_step,
                parent_id=previous_thought.id,
                session=session,
            )
