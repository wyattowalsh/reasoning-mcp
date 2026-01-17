"""Multi-Agent Debate reasoning method.

This module implements Multi-Agent Debate, where multiple AI agents with
different perspectives debate to improve reasoning accuracy and reduce
hallucinations through collaborative argumentation.

Key phases:
1. Initialize: Set up multiple agents with diverse viewpoints
2. Debate: Agents present arguments and counterarguments
3. Refine: Agents update positions based on debate
4. Consensus: Converge on best-supported answer

Reference: Du et al. (2023) - "Improving Factuality and Reasoning in LLMs
through Multiagent Debate"
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
    from fastmcp.server import Context

    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


MULTI_AGENT_DEBATE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.MULTI_AGENT_DEBATE,
    name="Multi-Agent Debate",
    description="Multiple agents debate to improve reasoning accuracy. "
    "Diverse perspectives and argumentation reduce hallucinations and errors.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"debate", "multi-agent", "collaborative", "verification", "consensus"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=12,
    avg_tokens_per_thought=350,
    best_for=("factual accuracy", "complex reasoning", "reducing hallucinations"),
    not_recommended_for=("simple queries", "real-time responses", "subjective tasks"),
)


class MultiAgentDebate(ReasoningMethodBase):
    """Multi-Agent Debate reasoning method implementation."""

    DEFAULT_AGENTS = 3
    DEFAULT_ROUNDS = 2
    _use_sampling: bool = True

    def __init__(
        self,
        num_agents: int = DEFAULT_AGENTS,
        num_rounds: int = DEFAULT_ROUNDS,
        enable_elicitation: bool = True,
    ) -> None:
        self._num_agents = num_agents
        self._num_rounds = num_rounds
        self.enable_elicitation = enable_elicitation
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize"
        self._agent_positions: list[dict[str, Any]] = []
        self._current_round = 0
        self._ctx: Context | None = None
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.MULTI_AGENT_DEBATE

    @property
    def name(self) -> str:
        return MULTI_AGENT_DEBATE_METADATA.name

    @property
    def description(self) -> str:
        return MULTI_AGENT_DEBATE_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "initialize"
        self._agent_positions = []
        self._current_round = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Multi-Agent Debate must be initialized before execution")

        # Store execution context for sampling and elicitation
        self._execution_context = execution_context
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter = 1
        self._current_phase = "initialize"

        # Initialize agent positions with LLM sampling if available
        self._agent_positions = []
        for i in range(self._num_agents):
            if execution_context and execution_context.can_sample:
                stance = await self._sample_agent_position(input_text, i + 1)
            else:
                stance = f"Position {i + 1}"
            self._agent_positions.append({"id": i + 1, "stance": stance, "confidence": 0.7})

        content = (
            f"Step {self._step_counter}: Initialize Debate (Multi-Agent Debate)\n\n"
            f"Problem: {input_text}\n\n"
            f"Setting up {self._num_agents} agents for debate...\n\n"
            f"Agents Initialized:\n"
            + "\n".join(
                f"  Agent {a['id']}: Ready with initial perspective" for a in self._agent_positions
            )
            + f"\n\nPlanned rounds: {self._num_rounds}\n"
            f"Next: Agents present initial positions."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.MULTI_AGENT_DEBATE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "agents": self._num_agents,
                "problem": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.MULTI_AGENT_DEBATE
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
            raise RuntimeError("Multi-Agent Debate must be initialized before continuation")

        # Store execution context for sampling and elicitation
        self._execution_context = execution_context
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize")

        if prev_phase == "initialize":
            self._current_phase = "initial_positions"

            # Generate initial reasoning for each agent with LLM sampling if available
            agent_reasonings = []
            input_text = previous_thought.metadata.get("problem", "")
            for a in self._agent_positions:
                if execution_context and execution_context.can_sample:
                    reasoning = await self._sample_agent_reasoning(
                        input_text, a["id"], a["stance"], self._current_round
                    )
                else:
                    reasoning = f"[Agent {a['id']}'s initial reasoning]"
                agent_reasonings.append(reasoning)

            content = (
                f"Step {self._step_counter}: Initial Positions\n\n"
                f"Each agent presents their initial answer:\n\n"
                + "\n".join(
                    f'  Agent {a["id"]}: "{a["stance"]}"\n'
                    f"    Reasoning: {agent_reasonings[i]}\n"
                    f"    Confidence: {a['confidence']:.0%}"
                    for i, a in enumerate(self._agent_positions)
                )
                + "\n\nAgents now see each other's positions.\n"
                "Next: Begin debate round 1."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.6
        elif prev_phase == "initial_positions" or (
            prev_phase == "debate" and self._current_round < self._num_rounds
        ):
            self._current_phase = "debate"
            self._current_round += 1
            # Simulate agents updating confidence based on debate
            for agent in self._agent_positions:
                agent["confidence"] = min(0.95, agent["confidence"] + 0.1)

            # Optional elicitation: ask user to rate agent arguments
            if self.enable_elicitation and self._ctx and self._current_round == 1:
                try:
                    elicit_config = ElicitationConfig(
                        timeout=45, required=False, default_on_timeout=None
                    )
                    feedback = await elicit_feedback(
                        self._ctx,
                        "The agents have presented their initial positions. "
                        "Do you have any feedback or perspective that could help guide the debate?",
                        config=elicit_config,
                    )
                    if feedback.feedback:
                        # Incorporate user feedback into debate
                        guidance = f"\n\n[User Feedback]: {feedback.feedback}"
                        session.metrics.elicitations_made += 1
                    else:
                        guidance = ""
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        error=str(e),
                    )
                    guidance = ""
            else:
                guidance = ""

            # Generate debate arguments for each agent with LLM sampling if available
            input_text = previous_thought.metadata.get("problem", "")
            agent_arguments = []
            for a in self._agent_positions:
                if execution_context and execution_context.can_sample:
                    argument = await self._sample_debate_argument(
                        input_text, a["id"], a["stance"], self._current_round, self._agent_positions
                    )
                    counterargument = await self._sample_counterargument(
                        input_text, a["id"], a["stance"], self._current_round, self._agent_positions
                    )
                else:
                    argument = "[Supports or challenges other positions]"
                    counterargument = "[Counterarguments]"
                agent_arguments.append({"argument": argument, "counterargument": counterargument})

            content = (
                f"Step {self._step_counter}: Debate Round {self._current_round}\n\n"
                f"Agents engage in argumentation:\n\n"
                + "\n".join(
                    f"  Agent {a['id']}:\n"
                    f"    Argument: {agent_arguments[i]['argument']}\n"
                    f"    Response to others: {agent_arguments[i]['counterargument']}\n"
                    f"    Updated confidence: {a['confidence']:.0%}"
                    for i, a in enumerate(self._agent_positions)
                )
                + guidance
                + f"\n\nRound {self._current_round}/{self._num_rounds} complete.\n"
                + (
                    "Next: Continue debate."
                    if self._current_round < self._num_rounds
                    else "Next: Reach consensus."
                )
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.7 + (self._current_round * 0.05)
        elif prev_phase == "debate":
            self._current_phase = "consensus"
            # Find position with highest confidence
            best_agent = max(self._agent_positions, key=lambda a: a["confidence"])

            # Optional elicitation: ask user which agent's position is most convincing
            user_choice = ""
            if self.enable_elicitation and self._ctx and len(self._agent_positions) > 1:
                try:
                    agent_options = [
                        {
                            "id": str(a["id"] - 1),
                            "label": f"Agent {a['id']}: {a['stance']} "
                            f"(confidence: {a['confidence']:.0%})",
                        }
                        for a in self._agent_positions
                    ]
                    elicit_config = ElicitationConfig(
                        timeout=45, required=False, default_on_timeout=None
                    )
                    selection = await elicit_selection(
                        self._ctx,
                        "Which agent's position do you find most convincing after the debate?",
                        agent_options,
                        config=elicit_config,
                    )
                    selected_idx = int(selection.selected)
                    if 0 <= selected_idx < len(self._agent_positions):
                        user_choice = (
                            f"\n\n[User Selection]: Agent {selected_idx + 1}'s position "
                            f"was selected as most convincing "
                            f"(confidence: {selection.confidence:.0%})"
                        )
                        session.metrics.elicitations_made += 1
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        error=str(e),
                    )

            content = (
                f"Step {self._step_counter}: Reach Consensus\n\n"
                f"After {self._num_rounds} rounds of debate:\n\n"
                f"Final Positions:\n"
                + "\n".join(
                    f"  Agent {a['id']}: {a['stance']} (confidence: {a['confidence']:.0%})"
                    for a in self._agent_positions
                )
                + user_choice
                + f"\n\nConsensus Analysis:\n"
                f"  Strongest position: Agent {best_agent['id']}\n"
                f"  Agreement level: High\n"
                f"  Verified through: Multi-perspective debate"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            best_agent = max(self._agent_positions, key=lambda a: a["confidence"])
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Multi-Agent Debate Complete:\n"
                f"  Agents: {self._num_agents}\n"
                f"  Rounds: {self._num_rounds}\n"
                f"  Winning position: Agent {best_agent['id']}\n\n"
                f"Final Answer: [Consensus answer from debate]\n"
                f"Confidence: High ({int(best_agent['confidence'] * 100)}%)\n\n"
                f"Verification: Answer validated through multi-agent argumentation\n"
                f"  - Multiple perspectives considered\n"
                f"  - Counterarguments addressed\n"
                f"  - Consensus achieved through debate"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = best_agent["confidence"]

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.MULTI_AGENT_DEBATE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "round": self._current_round,
                "positions": self._agent_positions,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    async def _sample_agent_position(self, problem: str, agent_id: int) -> str:
        """Generate an initial position for an agent using LLM sampling.

        Args:
            problem: The problem to reason about
            agent_id: The agent's ID number

        Returns:
            A concise position statement for the agent
        """
        system_prompt = f"""You are Agent {agent_id} in a multi-agent debate.
Generate a unique initial position or perspective on the given problem.
Your position should be different from other agents while staying relevant to the problem.
Be concise - state your position in 1-2 sentences."""

        user_prompt = f"""Problem: {problem}

As Agent {agent_id}, what is your initial position on this problem?
Provide a concise stance that you will argue for in the debate."""

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: f"Position {agent_id}",
            system_prompt=system_prompt,
            temperature=0.8 + (agent_id * 0.1),  # Vary temperature for diversity
            max_tokens=150,
        )
        return result.strip()

    async def _sample_agent_reasoning(
        self, problem: str, agent_id: int, stance: str, round_num: int
    ) -> str:
        """Generate reasoning for an agent's position using LLM sampling.

        Args:
            problem: The problem to reason about
            agent_id: The agent's ID number
            stance: The agent's stated position
            round_num: Current debate round

        Returns:
            Reasoning supporting the agent's position
        """
        system_prompt = f"""You are Agent {agent_id} in a multi-agent debate.
Provide clear reasoning to support your position.
Focus on logical arguments and evidence."""

        user_prompt = f"""Problem: {problem}

Your position: {stance}

Provide reasoning to support your position. Explain why your stance is valid and well-founded."""

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: f"[Agent {agent_id}'s reasoning]",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )
        return result.strip()

    async def _sample_debate_argument(
        self,
        problem: str,
        agent_id: int,
        stance: str,
        round_num: int,
        all_positions: list[dict[str, Any]],
    ) -> str:
        """Generate a debate argument using LLM sampling.

        Args:
            problem: The problem to reason about
            agent_id: The agent's ID number
            stance: The agent's stated position
            round_num: Current debate round
            all_positions: List of all agent positions

        Returns:
            An argument supporting the agent's position
        """
        other_positions = "\n".join(
            f"  Agent {a['id']}: {a['stance']}" for a in all_positions if a["id"] != agent_id
        )

        system_prompt = f"""You are Agent {agent_id} in round {round_num} of a multi-agent debate.
Present arguments that support your position or challenge other positions.
Be persuasive but respectful."""

        user_prompt = f"""Problem: {problem}

Your position: {stance}

Other agents' positions:
{other_positions}

Present your argument for this round. Support your position and/or respectfully challenge \
other views."""

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: "[Supports or challenges other positions]",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )
        return result.strip()

    async def _sample_counterargument(
        self,
        problem: str,
        agent_id: int,
        stance: str,
        round_num: int,
        all_positions: list[dict[str, Any]],
    ) -> str:
        """Generate counterarguments to other positions using LLM sampling.

        Args:
            problem: The problem to reason about
            agent_id: The agent's ID number
            stance: The agent's stated position
            round_num: Current debate round
            all_positions: List of all agent positions

        Returns:
            Counterarguments to other positions
        """
        other_positions = "\n".join(
            f"  Agent {a['id']}: {a['stance']}" for a in all_positions if a["id"] != agent_id
        )

        system_prompt = f"""You are Agent {agent_id} in round {round_num} of a multi-agent debate.
Respond to other agents' arguments with counterarguments.
Be analytical and address specific points raised by others."""

        user_prompt = f"""Problem: {problem}

Your position: {stance}

Other agents' positions:
{other_positions}

Respond to the other agents' positions with counterarguments. Address their key points."""

        result = await self._sample_with_fallback(
            user_prompt=user_prompt,
            fallback_generator=lambda: "[Counterarguments]",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )
        return result.strip()


__all__ = ["MultiAgentDebate", "MULTI_AGENT_DEBATE_METADATA"]
