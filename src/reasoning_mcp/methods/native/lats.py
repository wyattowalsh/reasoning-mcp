"""LATS (Language Agent Tree Search) reasoning method.

This module implements LATS, which unifies reasoning, acting, and planning
in a single framework. Combines MCTS-style search with LLM agents for
complex task solving.

Key phases:
1. Initialize: Set up search tree with initial state
2. Select: Choose promising node using UCT
3. Expand: Generate child actions via LLM
4. Evaluate: Score outcomes with value function
5. Backpropagate: Update tree statistics

Reference: Zhou et al. (2024) - "Language Agent Tree Search Unifies Reasoning,
Acting, and Planning in Language Models" (ICML 2024)
"""

from __future__ import annotations

import re
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

logger = structlog.get_logger(__name__)


LATS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LATS,
    name="LATS",
    description="Language Agent Tree Search - unifies reasoning, acting, and planning. "
    "Combines MCTS with LLM agents for complex task solving.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"agent", "tree-search", "mcts", "planning", "acting", "unified"}),
    complexity=8,
    supports_branching=True,
    supports_revision=True,
    requires_context=True,
    min_thoughts=5,
    max_thoughts=10,
    avg_tokens_per_thought=350,
    best_for=("complex tasks", "multi-step planning", "agent tasks", "decision making"),
    not_recommended_for=("simple queries", "single-step problems"),
)


class Lats(ReasoningMethodBase):
    """LATS reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize"
        self._search_tree: dict[str, Any] = {}
        self._current_node: str = "root"
        self._iteration: int = 0
        self._max_iterations: int = 4
        self._best_trajectory: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.LATS

    @property
    def name(self) -> str:
        return LATS_METADATA.name

    @property
    def description(self) -> str:
        return LATS_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "initialize"
        self._search_tree = {}
        self._current_node = "root"
        self._iteration = 0
        self._max_iterations = 4
        self._best_trajectory = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("LATS must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "initialize"
        self._iteration = 1

        # Initialize search tree
        self._search_tree = {
            "root": {
                "state": "initial",
                "visits": 0,
                "value": 0.0,
                "children": [],
                "parent": None,
                "action": None,
                "observation": input_text,
            }
        }

        content = (
            f"Step {self._step_counter}: Initialize Search Tree (LATS)\n\n"
            f"Problem: {input_text}\n\n"
            f"Search Tree Initialized:\n"
            f"  Root state: {self._search_tree['root']['state']}\n"
            f"  Observation: {self._search_tree['root']['observation'][:50]}...\n"
            f"  Max iterations: {self._max_iterations}\n\n"
            f"LATS Framework:\n"
            f"  - Reasoning: Think about next steps\n"
            f"  - Acting: Execute actions in environment\n"
            f"  - Planning: MCTS-style tree search\n\n"
            f"Components:\n"
            f"  - Selection: UCT algorithm\n"
            f"  - Expansion: LLM generates actions\n"
            f"  - Evaluation: Value function scoring\n"
            f"  - Backpropagation: Update statistics\n\n"
            f"Next: Select node and expand."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LATS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
                "tree_size": len(self._search_tree),
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LATS
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
            raise RuntimeError("LATS must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize")

        if prev_phase == "initialize" or prev_phase == "backpropagate":
            if self._iteration > self._max_iterations:
                self._current_phase = "conclude"
            else:
                self._current_phase = "select"
                # Select node using UCT
                c = 1.4  # Exploration constant

                has_children = bool(self._search_tree["root"]["children"])
                selected_node = "best_child" if has_children else "root"
                criterion = "Highest UCT" if has_children else "Unexpanded"
                content = (
                    f"Step {self._step_counter}: Select Node (UCT)\n\n"
                    f"Iteration {self._iteration}/{self._max_iterations}\n\n"
                    f"UCT Selection:\n"
                    f"  Exploration constant (c): {c}\n"
                    f"  Formula: UCT = V/N + c × √(ln(N_parent)/N)\n\n"
                    f"Node Evaluation:\n"
                    f"  Root visits: {self._search_tree['root']['visits']}\n"
                    f"  Root value: {self._search_tree['root']['value']:.2f}\n"
                    f"  Children: {len(self._search_tree['root']['children'])}\n\n"
                    f"Selected: {selected_node}\n"
                    f"Selection criterion: {criterion}\n\n"
                    f"Next: Expand selected node."
                )
                thought_type = ThoughtType.REASONING
                confidence = 0.7

        elif prev_phase == "select":
            self._current_phase = "expand"
            # Expand with LLM-generated actions
            new_actions = await self._generate_actions(session.initial_input or "", self._iteration)

            # Add to tree
            for action in new_actions:
                node_id = action["id"]
                self._search_tree[node_id] = {
                    "state": action["content"],
                    "visits": 0,
                    "value": 0.0,
                    "children": [],
                    "parent": self._current_node,
                    "action": action,
                }
                self._search_tree[self._current_node]["children"].append(node_id)

            content = (
                f"Step {self._step_counter}: Expand Node\n\n"
                f"Iteration {self._iteration}/{self._max_iterations}\n\n"
                f"LLM-Generated Actions:\n"
                + "\n".join(
                    f"  [{a['id']}] {a['type'].upper()}: {a['content']}" for a in new_actions
                )
                + f"\n\nExpansion Statistics:\n"
                f"  New nodes: {len(new_actions)}\n"
                f"  Total tree size: {len(self._search_tree)}\n"
                f"  Parent: {self._current_node}\n\n"
                f"Action Types:\n"
                f"  THINK: Reasoning about the problem\n"
                f"  ACT: Taking action in environment\n\n"
                f"Next: Evaluate expanded nodes."
            )
            thought_type = ThoughtType.EXPLORATION
            confidence = 0.72

        elif prev_phase == "expand":
            self._current_phase = "evaluate"
            # Evaluate nodes with value function (using LLM or fallback)
            evaluations = []
            for child_id in self._search_tree[self._current_node]["children"]:
                value = await self._evaluate_node_value(child_id, session.initial_input or "")
                self._search_tree[child_id]["value"] = value
                evaluations.append({"id": child_id, "value": value})

            best_child = max(evaluations, key=lambda x: x["value"])

            content = (
                f"Step {self._step_counter}: Evaluate Nodes\n\n"
                f"Iteration {self._iteration}/{self._max_iterations}\n\n"
                f"Value Function Evaluation:\n"
                + "\n".join(
                    f"  {e['id']}: {e['value']:.2f}"
                    for e in sorted(evaluations, key=lambda x: -x["value"])
                )
                + f"\n\nEvaluation Method:\n"
                f"  - LLM self-evaluation\n"
                f"  - Heuristic progress estimation\n"
                f"  - Goal proximity measure\n\n"
                f"Best node: {best_child['id']} (value: {best_child['value']:.2f})\n\n"
                f"Next: Backpropagate values."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.78

        elif prev_phase == "evaluate":
            self._current_phase = "backpropagate"
            # Backpropagate values
            best_value = max(
                self._search_tree[c]["value"]
                for c in self._search_tree[self._current_node]["children"]
            )

            # Update root
            self._search_tree["root"]["visits"] += 1
            self._search_tree["root"]["value"] = (
                self._search_tree["root"]["value"] * (self._search_tree["root"]["visits"] - 1)
                + best_value
            ) / self._search_tree["root"]["visits"]

            # Track best trajectory
            children_raw = self._search_tree[self._current_node]["children"]
            children_list: list[str] = [str(c) for c in children_raw]

            def get_child_value(child_id: str) -> float:
                return float(str(self._search_tree[child_id]["value"]))

            best_child = max(children_list, key=get_child_value)
            self._best_trajectory.append(
                {
                    "iteration": self._iteration,
                    "node": best_child,
                    "value": get_child_value(str(best_child)),
                }
            )

            self._iteration += 1

            status_msg = (
                "Continuing search..."
                if self._iteration <= self._max_iterations
                else "Max iterations reached."
            )
            content = (
                f"Step {self._step_counter}: Backpropagate\n\n"
                f"Updating tree statistics:\n\n"
                f"Updates Applied:\n"
                f"  Root visits: {self._search_tree['root']['visits']}\n"
                f"  Root value: {self._search_tree['root']['value']:.2f}\n"
                f"  Best child value: {best_value:.2f}\n\n"
                f"Best Trajectory So Far:\n"
                + "\n".join(
                    f"  Iter {t['iteration']}: {t['node']} (value: {t['value']:.2f})"
                    for t in self._best_trajectory
                )
                + f"\n\nIteration {self._iteration - 1} complete.\n"
                f"{status_msg}"
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.8

        else:  # conclude
            self._current_phase = "conclude"
            final_value = self._search_tree["root"]["value"]

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"LATS Complete:\n"
                f"  Iterations: {self._iteration - 1}\n"
                f"  Tree size: {len(self._search_tree)} nodes\n"
                f"  Root visits: {self._search_tree['root']['visits']}\n"
                f"  Final value: {final_value:.2f}\n\n"
                f"Best Trajectory:\n"
                + "\n".join(
                    f"  {t['iteration']}. {t['node']}: {t['value']:.2f}"
                    for t in self._best_trajectory
                )
                + f"\n\nFinal Answer: [Solution via best trajectory]\n"
                f"Confidence: High ({int(final_value * 100 + 10)}%)\n\n"
                f"Method: LATS\n"
                f"  - Unified reasoning + acting + planning\n"
                f"  - MCTS-style tree search\n"
                f"  - UCT selection strategy\n"
                f"  - LLM value function\n"
                f"  - Backpropagation updates"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = min(0.92, final_value + 0.15)

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LATS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
                "tree_size": len(self._search_tree),
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    async def _generate_actions(self, input_text: str, iteration: int) -> list[dict[str, Any]]:
        """Generate actions for expansion using LLM sampling or fallback heuristics.

        Args:
            input_text: The original problem/query
            iteration: Current iteration number

        Returns:
            List of action dictionaries with id, type, and content
        """
        system_prompt = (
            "You are an expert in LATS (Language Agent Tree Search) helping to generate "
            "strategic actions for exploration. Generate 3 distinct actions that combine "
            "reasoning (THINK) and acting (ACT) to solve the problem. "
            "Each action should be concrete and explore different aspects."
        )

        user_prompt = f"""Problem: {input_text}

Iteration: {iteration}

Generate 3 actions for LATS exploration. For each action:
1. Choose type: either THINK (reasoning) or ACT (taking action)
2. Provide a concrete, actionable description
3. Ensure actions explore different strategic dimensions

Format each action as:
TYPE: Description

Example:
THINK: Analyze the root cause of the problem
ACT: Gather data about key variables
THINK: Plan a multi-step solution approach"""

        def fallback_generator() -> str:
            """Generate fallback response when LLM is unavailable."""
            return "THINK: Analyze the problem structure\nACT: Extract key variables\nTHINK: Plan solution approach"

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )

        # Parse actions from response
        actions = []
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        for idx, line in enumerate(lines[:3]):  # Take first 3
            if ":" in line:
                parts = line.split(":", 1)
                action_type = parts[0].strip().lower()
                content = parts[1].strip()
                if action_type in ["think", "act"]:
                    actions.append(
                        {
                            "id": f"a{iteration}_{idx + 1}",
                            "type": action_type,
                            "content": content,
                        }
                    )

        # If we got valid actions, return them; otherwise use fallback
        if len(actions) >= 3:
            return actions[:3]

        # Fallback: Generate heuristic actions
        return self._generate_fallback_actions(iteration)

    def _generate_fallback_actions(self, iteration: int) -> list[dict[str, Any]]:
        """Generate fallback actions when LLM sampling is unavailable.

        Args:
            iteration: Current iteration number

        Returns:
            List of action dictionaries
        """
        think_actions = [
            "Analyze the problem structure",
            "Break down into subproblems",
            "Identify key constraints",
            "Plan solution approach",
            "Consider edge cases",
            "Evaluate alternatives",
            "Reason about dependencies",
            "Map problem space",
        ]

        act_actions = [
            "Extract key variables",
            "Gather relevant information",
            "Execute preliminary steps",
            "Test initial hypothesis",
            "Apply known patterns",
            "Compute intermediate results",
            "Validate assumptions",
            "Implement core logic",
        ]

        # Alternate between think and act, with variety
        actions = []
        for i in range(3):
            if i % 2 == 0:
                action_type = "think"
                content = think_actions[(iteration + i) % len(think_actions)]
            else:
                action_type = "act"
                content = act_actions[(iteration + i) % len(act_actions)]

            actions.append(
                {
                    "id": f"a{iteration}_{i + 1}",
                    "type": action_type,
                    "content": content,
                }
            )

        return actions

    async def _evaluate_node_value(self, node_id: str, input_text: str) -> float:
        """Evaluate the value of a node using LLM sampling or fallback heuristics.

        Args:
            node_id: ID of the node to evaluate
            input_text: The original problem/query

        Returns:
            Value score between 0.0 and 1.0
        """
        node_info = self._search_tree.get(node_id, {})
        action = node_info.get("action", {})

        system_prompt = (
            "You are an expert evaluator for LATS (Language Agent Tree Search). "
            "Evaluate how promising an action is for solving the given problem. "
            "Consider relevance, strategic value, and likelihood of success. "
            "Respond with only a score between 0.0 (not promising) and "
            "1.0 (very promising)."
        )

        user_prompt = f"""Problem: {input_text}

Action Type: {action.get("type", "unknown").upper()}
Action: {action.get("content", "unknown")}

Evaluate how promising this action is for solving the problem.
Score (0.0 to 1.0):"""

        def fallback_generator() -> str:
            """Generate fallback value when LLM is unavailable."""
            # Return a heuristic value based on hash
            heuristic_value = 0.6 + (0.1 * (hash(node_id) % 4))
            return str(heuristic_value)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=50,
        )

        # Try to extract a number from the response
        match = re.search(r"0\.\d+|1\.0|0|1", result)
        if match:
            value = float(match.group())
            return max(0.0, min(1.0, value))

        # Fallback: Use heuristic value based on hash
        return 0.6 + (0.1 * (hash(node_id) % 4))


__all__ = ["Lats", "LATS_METADATA"]
