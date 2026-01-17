"""Branch-Solve-Merge (BSM) reasoning method.

This module implements BSM, which decomposes complex tasks into parallel
sub-tasks, solves each independently, and merges solutions. The LLM itself
decides the branching factor and sub-problems.

Key phases:
1. Branch: Decompose task into parallel sub-tasks
2. Solve: Independently solve each sub-task
3. Merge: Fuse solutions into coherent output

Reference: Saha et al. (2024) - "Branch-Solve-Merge Improves Large Language
Model Evaluation and Generation" (NAACL 2024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_confirmation,
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

    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


BRANCH_SOLVE_MERGE_METADATA = MethodMetadata(
    identifier=MethodIdentifier.BRANCH_SOLVE_MERGE,
    name="Branch-Solve-Merge",
    description="Decomposes complex tasks into parallel sub-tasks, solves each "
    "independently, then merges solutions. LLM decides branching and sub-problems.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"decomposition", "parallel", "merge", "planning", "multi-faceted"}),
    complexity=6,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=350,
    best_for=("multi-faceted tasks", "constrained generation", "evaluation", "complex criteria"),
    not_recommended_for=("simple tasks", "single-objective problems"),
)


class BranchSolveMerge(ReasoningMethodBase):
    """Branch-Solve-Merge reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Branch-Solve-Merge method.

        Args:
            enable_elicitation: Whether to enable user interaction (default: True)
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "branch"
        self._branches: list[dict[str, Any]] = []
        self._solutions: list[dict[str, Any]] = []
        self._merged_result: str | None = None
        self.enable_elicitation = enable_elicitation
        self._ctx: Context | None = None
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.BRANCH_SOLVE_MERGE

    @property
    def name(self) -> str:
        return BRANCH_SOLVE_MERGE_METADATA.name

    @property
    def description(self) -> str:
        return BRANCH_SOLVE_MERGE_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "branch"
        self._branches = []
        self._solutions = []
        self._merged_result = None

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("BSM must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter = 1
        self._current_phase = "branch"

        # Branch: LLM decides sub-tasks
        self._branches = await self._generate_branches(input_text)

        # Optional elicitation: ask user which branches to prioritize
        elicited_response = ""
        if self.enable_elicitation and self._ctx:
            try:
                branch_options = [
                    {
                        "id": str(b["id"]),
                        "label": f"{b['task']} (Criteria: {b['criteria']})",
                    }
                    for b in self._branches
                ]
                elicit_config = ElicitationConfig(
                    timeout=30, required=False, default_on_timeout=None
                )
                selection = await elicit_selection(
                    self._ctx,
                    "Branch-Solve-Merge has identified multiple branches to "
                    "explore. Which branch should be prioritized?",
                    branch_options,
                    config=elicit_config,
                )
                if selection and selection.selected:
                    # Reorder branches to prioritize selected one
                    selected_id = int(selection.selected)
                    for i, branch in enumerate(self._branches):
                        if branch["id"] == selected_id:
                            self._branches.insert(0, self._branches.pop(i))
                            elicited_response = (
                                f"\n\n[User Input]: Prioritized branch {selected_id}"
                            )
                            session.metrics.elicitations_made += 1
                            break
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Elicitation failed - continue without it
            except Exception as e:
                logger.error(
                    "unexpected_error_in_elicitation",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        content = (
            f"Step {self._step_counter}: Branch - Decompose into Sub-Tasks (BSM)\n\n"
            f"Problem: {input_text}\n\n"
            f"Decomposing into parallel sub-tasks...\n\n"
            f"Generated Branches:\n"
            + "\n".join(
                f"  [{b['id']}] {b['task']}\n      Criteria: {b['criteria']}"
                for b in self._branches
            )
            + f"\n\nBranching factor: {len(self._branches)}\n"
            f"Each branch will be solved independently.\n"
            f"Next: Solve each branch." + elicited_response
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.BRANCH_SOLVE_MERGE,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "branches": len(self._branches),
                "input_text": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.BRANCH_SOLVE_MERGE
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
            raise RuntimeError("BSM must be initialized before continuation")

        # Store context for elicitation
        if execution_context and hasattr(execution_context, "ctx"):
            self._ctx = execution_context.ctx

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "branch")

        if prev_phase == "branch":
            self._current_phase = "solve"
            # Solve each branch independently
            self._solutions = []
            # Get input_text from previous thought metadata
            input_text = previous_thought.metadata.get("input_text", "")
            for branch in self._branches:
                solution = await self._solve_branch(branch, input_text)
                self._solutions.append(solution)

            content = (
                f"Step {self._step_counter}: Solve - Process Each Branch\n\n"
                f"Solving {len(self._branches)} branches independently:\n\n"
                f"Branch Solutions:\n"
                + "\n".join(
                    f"  Branch {s['branch_id']}: {s['task']}\n"
                    f"    Result: {s['result']}\n"
                    f"    Quality: {s['score']:.0%}"
                    for s in self._solutions
                )
                + f"\n\nAll {len(self._solutions)} branches solved.\n"
                f"Next: Merge solutions into coherent output."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.75
        elif prev_phase == "solve":
            self._current_phase = "merge"
            # Merge solutions
            avg_score = sum(s["score"] for s in self._solutions) / len(self._solutions)
            # Get input_text from previous thought metadata
            input_text = previous_thought.metadata.get("input_text", "")
            self._merged_result = await self._merge_solutions(input_text)

            # Optional elicitation: confirm merge strategy
            elicited_response = ""
            if self.enable_elicitation and self._ctx:
                try:
                    elicit_config = ElicitationConfig(
                        timeout=30, required=False, default_on_timeout=None
                    )
                    confirmation = await elicit_confirmation(
                        self._ctx,
                        f"Branch-Solve-Merge is ready to merge "
                        f"{len(self._solutions)} solutions. Proceed with merge?",
                        config=elicit_config,
                    )
                    if confirmation:
                        merge_status = "approved" if confirmation.confirmed else "deferred"
                        elicited_response = f"\n\n[User Input]: Merge {merge_status}"
                        session.metrics.elicitations_made += 1
                        if not confirmation.confirmed:
                            elicited_response += " - Continuing with detailed analysis before merge"
                except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                    logger.warning(
                        "elicitation_failed",
                        method="continue_reasoning",
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    # Elicitation failed - continue without it
                except Exception as e:
                    logger.error(
                        "unexpected_error_in_elicitation",
                        method="continue_reasoning",
                        error_type=type(e).__name__,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

            content = (
                f"Step {self._step_counter}: Merge - Fuse Branch Solutions\n\n"
                f"Merging {len(self._solutions)} branch solutions:\n\n"
                f"Merge Strategy:\n"
                f"  - Combine insights from each branch\n"
                f"  - Resolve conflicts between branches\n"
                f"  - Ensure coherence across all aspects\n\n"
                f"Integration:\n"
                + "\n".join(
                    f"  + Branch {s['branch_id']} ({s['score']:.0%}): Integrated"
                    for s in self._solutions
                )
                + f"\n\nMerged Output:\n"
                f"  {self._merged_result}\n\n"
                f"Average branch quality: {avg_score:.0%}" + elicited_response
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            avg_score = (
                sum(s["score"] for s in self._solutions) / len(self._solutions)
                if self._solutions
                else 0.85
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Branch-Solve-Merge Complete:\n"
                f"  Branches created: {len(self._branches)}\n"
                f"  Solutions generated: {len(self._solutions)}\n"
                f"  Average quality: {avg_score:.0%}\n\n"
                f"Final Answer: {self._merged_result}\n"
                f"Confidence: High ({int(avg_score * 100)}%)\n\n"
                f"Method: Branch-Solve-Merge (BSM)\n"
                f"  - Decomposed into parallel sub-tasks\n"
                f"  - Solved each branch independently\n"
                f"  - Merged solutions coherently\n"
                f"  - LLM-driven branching factor"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = avg_score

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.BRANCH_SOLVE_MERGE,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "branches": len(self._branches),
                "solutions": len(self._solutions),
                "input_text": previous_thought.metadata.get("input_text", ""),
            },
        )
        session.add_thought(thought)
        return thought

    async def _generate_branches(self, input_text: str) -> list[dict[str, Any]]:
        """Generate branches using LLM sampling or fallback heuristics.

        Args:
            input_text: The problem to decompose into branches

        Returns:
            List of branch dictionaries with id, task, and criteria
        """

        def fallback_branches() -> str:
            """Generate fallback heuristic decomposition."""
            branches = [
                {
                    "id": 1,
                    "task": "Aspect 1: Core requirements analysis",
                    "criteria": "completeness",
                },
                {"id": 2, "task": "Aspect 2: Quality and coherence check", "criteria": "quality"},
                {"id": 3, "task": "Aspect 3: Constraint satisfaction", "criteria": "constraints"},
            ]
            return "\n".join(f"{b['id']}. {b['task']} | {b['criteria']}" for b in branches)

        prompt = f"""Analyze this problem and decompose it into 3-5 \
parallel sub-tasks for Branch-Solve-Merge reasoning.

Problem: {input_text}

For each sub-task, provide:
1. A clear task description
2. An evaluation criterion (e.g., completeness, quality, constraints, \
accuracy, feasibility)

Format your response as a numbered list where each item has the format:
[Task description] | [Criterion]

Example:
1. Analyze core requirements and constraints | completeness
2. Evaluate technical feasibility and approach | feasibility
3. Assess quality and coherence of solution | quality"""

        system_prompt = (
            "You are an expert at decomposing complex problems into "
            "parallel sub-tasks for divide-and-conquer reasoning."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_branches,
            system_prompt=system_prompt,
        )

        # Parse the LLM response into branches
        branches = []
        lines = result.strip().split("\n")
        branch_id = 1
        for line in lines:
            line = line.strip()
            if not line or not any(line.startswith(str(i)) for i in range(1, 10)):
                continue

            # Remove numbering (e.g., "1. " or "1) ")
            line = line.lstrip("0123456789.)- \t")

            if "|" in line:
                task, criteria = line.split("|", 1)
                branches.append(
                    {"id": branch_id, "task": task.strip(), "criteria": criteria.strip()}
                )
                branch_id += 1

        if len(branches) >= 3:
            return branches[:5]  # Limit to max 5 branches

        # If parsing failed, return fallback
        return [
            {"id": 1, "task": "Aspect 1: Core requirements analysis", "criteria": "completeness"},
            {"id": 2, "task": "Aspect 2: Quality and coherence check", "criteria": "quality"},
            {"id": 3, "task": "Aspect 3: Constraint satisfaction", "criteria": "constraints"},
        ]

    async def _solve_branch(self, branch: dict[str, Any], input_text: str) -> dict[str, Any]:
        """Solve a single branch using LLM sampling or fallback heuristics.

        Args:
            branch: Branch dictionary with id, task, and criteria
            input_text: Original problem text

        Returns:
            Solution dictionary with branch_id, task, result, and score
        """

        def fallback_solution() -> str:
            """Generate fallback solution."""
            return f"[Solution for {branch['task']}]"

        prompt = f"""Solve this specific aspect of the problem using \
the Branch-Solve-Merge approach.

Original Problem: {input_text}

Your Sub-Task: {branch["task"]}
Evaluation Criterion: {branch["criteria"]}

Provide a focused solution addressing only this sub-task. Consider the \
evaluation criterion when forming your answer."""

        system_prompt = (
            f"You are solving a specific branch of a larger problem. "
            f"Focus on the {branch['criteria']} aspect."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_solution,
            system_prompt=system_prompt,
        )

        # Estimate quality score based on response length and structure
        score = min(0.95, 0.70 + len(result) / 2000)

        return {
            "branch_id": branch["id"],
            "task": branch["task"],
            "result": result.strip(),
            "score": score,
        }

    async def _merge_solutions(self, input_text: str) -> str:
        """Merge all branch solutions using LLM sampling or fallback heuristics.

        Args:
            input_text: Original problem text

        Returns:
            Merged solution string
        """

        def fallback_merge() -> str:
            """Generate fallback merged solution."""
            return "[Merged solution combining all branch outputs]"

        solutions_text = "\n\n".join(
            f"Branch {s['branch_id']} ({s['task']}):\n{s['result']}" for s in self._solutions
        )

        prompt = f"""Merge these independent branch solutions into a \
single coherent answer.

Original Problem: {input_text}

Branch Solutions:
{solutions_text}

Provide a unified answer that:
1. Integrates insights from all branches
2. Resolves any conflicts between branches
3. Ensures coherence and completeness
4. Addresses all aspects of the original problem"""

        system_prompt = (
            "You are an expert at synthesizing multiple partial solutions into a coherent whole."
        )

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_merge,
            system_prompt=system_prompt,
        )

        return result.strip()

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["BranchSolveMerge", "BRANCH_SOLVE_MERGE_METADATA"]
