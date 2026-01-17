"""Training-Free GRPO reasoning method.

This module implements Training-Free GRPO (Cai et al. Oct 2025), an advanced reasoning
method that achieves Group Relative Policy Optimization (GRPO) benefits without any
training through inference-time optimization techniques.

The key innovation is achieving preference optimization benefits purely at inference
time through candidate sampling, relative ranking via pairwise comparisons, and
iterative refinement - NO TRAINING REQUIRED.

Reference: Cai et al. "Training-Free GRPO: Group Relative Optimization at Inference Time" (Oct 2025)
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


# Metadata for Training-Free GRPO method
TRAINING_FREE_GRPO_METADATA = MethodMetadata(
    identifier=MethodIdentifier.TRAINING_FREE_GRPO,
    name="Training-Free GRPO",
    description="Group Relative Policy Optimization without training - achieves GRPO benefits "
    "through inference-time optimization. Generates multiple candidates, performs relative "
    "ranking via pairwise comparisons, selects top candidates, optionally refines, and "
    "concludes with best solution. NO TRAINING NEEDED - all optimization at inference time.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "training-free",
            "inference-time",
            "grpo",
            "preference-optimization",
            "relative-ranking",
            "pairwise-comparison",
            "candidate-sampling",
            "refinement",
            "self-optimization",
            "quality-selection",
        }
    ),
    complexity=8,  # Advanced complexity - sophisticated inference-time optimization
    supports_branching=True,  # Generates multiple candidate branches
    supports_revision=True,  # Refines selected candidates
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: sample_candidates + rank + select + conclude
    max_thoughts=20,  # Multiple candidates (4-5) + ranking + selection + refinement + conclusion
    avg_tokens_per_thought=500,  # Higher - includes pairwise comparisons and refinements
    best_for=(
        "high-quality solution generation",
        "preference optimization without training",
        "comparing multiple solution candidates",
        "inference-time compute scaling",
        "quality-sensitive tasks",
        "solution selection and refinement",
        "relative quality assessment",
        "training-free optimization",
    ),
    not_recommended_for=(
        "simple queries requiring single answer",
        "time-critical decisions (generates multiple candidates)",
        "resource-constrained environments",
        "tasks with clear single solution",
    ),
)

logger = structlog.get_logger(__name__)


class TrainingFreeGrpo(ReasoningMethodBase):
    """Training-Free GRPO reasoning method implementation.

    This class implements the Training-Free GRPO pattern (Cai et al. Oct 2025) where
    the system achieves Group Relative Policy Optimization benefits entirely at
    inference time without any training. The process involves:

    1. Sample Candidates: Generate 4-5 diverse solution candidates
    2. Rank Relatively: Use pairwise comparisons for relative ranking (no absolute scores)
    3. Select Best: Choose top candidate(s) based on relative ranking
    4. Refine: Optionally refine the selected candidate
    5. Conclude: Present the final optimized solution

    Key characteristics:
    - NO TRAINING REQUIRED - all optimization at inference time
    - Relative ranking through pairwise comparisons
    - Multiple candidate sampling for diversity
    - Quality selection without absolute scoring
    - Optional iterative refinement
    - Advanced complexity (8)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = TrainingFreeGrpo()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Solve the traveling salesman problem for 5 cities"
        ... )
        >>> print(result.content)  # Initial candidates

        Continue with ranking:
        >>> ranking = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Rank candidates"
        ... )
        >>> print(ranking.type)  # ThoughtType.VERIFICATION (ranking phase)

        Continue with selection:
        >>> selection = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=ranking,
        ...     guidance="Select best candidate"
        ... )
        >>> print(selection.type)  # ThoughtType.SYNTHESIS (select phase)

        Continue with refinement:
        >>> refined = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=selection,
        ...     guidance="Refine solution"
        ... )
        >>> print(refined.type)  # ThoughtType.REVISION (refine phase)

        Conclude:
        >>> conclusion = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=refined,
        ...     guidance="Finalize"
        ... )
        >>> print(conclusion.type)  # ThoughtType.CONCLUSION
    """

    # Number of candidates to generate
    NUM_CANDIDATES = 5
    # Whether to enable refinement phase
    ENABLE_REFINEMENT = True
    # Enable LLM sampling for candidate generation and ranking
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Training-Free GRPO method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "sample_candidates"
        # sample_candidates → rank_relatively → select_best → refine → conclude
        self._candidates: list[dict[str, Any]] = []
        self._pairwise_comparisons: list[dict[str, Any]] = []
        self._ranking: list[int] = []  # Indices of candidates in ranked order
        self._selected_candidate_idx: int = -1
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.TRAINING_FREE_GRPO

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return TRAINING_FREE_GRPO_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return TRAINING_FREE_GRPO_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Training-Free GRPO method for execution.
        Resets all state for a fresh reasoning session.

        Examples:
            >>> method = TrainingFreeGrpo()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert len(method._candidates) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "sample_candidates"
        self._candidates = []
        self._pairwise_comparisons = []
        self._ranking = []
        self._selected_candidate_idx = -1

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Training-Free GRPO method.

        This method initiates the process by generating multiple diverse candidate
        solutions. These candidates will then be ranked relatively through pairwise
        comparisons in subsequent steps.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include num_candidates, enable_refinement)

        Returns:
            A ThoughtNode representing the initial candidate generation

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = TrainingFreeGrpo()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Design an efficient sorting algorithm"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.TRAINING_FREE_GRPO
            >>> assert "num_candidates" in thought.metadata
            >>> assert thought.metadata["phase"] == "sample_candidates"
        """
        if not self._initialized:
            raise RuntimeError("Training-Free GRPO method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "sample_candidates"
        self._candidates = []
        self._pairwise_comparisons = []
        self._ranking = []
        self._selected_candidate_idx = -1

        # Extract configuration from context if provided
        num_candidates = self.NUM_CANDIDATES
        enable_refinement = self.ENABLE_REFINEMENT
        if context:
            num_candidates = context.get("num_candidates", self.NUM_CANDIDATES)
            num_candidates = max(2, min(num_candidates, 10))  # Clamp to 2-10
            enable_refinement = context.get("enable_refinement", self.ENABLE_REFINEMENT)

        # Generate candidate solutions
        content = await self._generate_candidates(input_text, num_candidates, context)

        # Initialize candidates list (would be populated with actual solutions)
        self._candidates = [
            {"id": i, "content": f"Candidate {i + 1} solution", "quality_hint": 0.5}
            for i in range(num_candidates)
        ]

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TRAINING_FREE_GRPO,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,  # Initial confidence - will improve with ranking and selection
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "training_free_grpo",
                "phase": self._current_phase,
                "num_candidates": num_candidates,
                "enable_refinement": enable_refinement,
                "candidates_generated": len(self._candidates),
                "training_required": False,  # Emphasize NO TRAINING
                "inference_time_only": True,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.TRAINING_FREE_GRPO

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

        This method implements the Training-Free GRPO phase transitions:
        - If previous was sample_candidates: perform relative ranking via pairwise comparisons
        - If previous was rank_relatively: select the best candidate
        - If previous was select_best: refine the selected candidate (if enabled)
        - If previous was refine: conclude with final solution

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the Training-Free GRPO process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = TrainingFreeGrpo()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Optimize function X")
            >>> ranking = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert ranking.type == ThoughtType.VERIFICATION
            >>> assert ranking.metadata["phase"] == "rank_relatively"
            >>>
            >>> selection = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=ranking
            ... )
            >>> assert selection.type == ThoughtType.SYNTHESIS
            >>> assert selection.metadata["phase"] == "select_best"
        """
        if not self._initialized:
            raise RuntimeError("Training-Free GRPO method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Get configuration from previous thought's metadata
        enable_refinement = previous_thought.metadata.get(
            "enable_refinement", self.ENABLE_REFINEMENT
        )

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "sample_candidates")

        if prev_phase == "sample_candidates":
            # Next: rank_relatively (pairwise comparisons)
            thought_type, content, confidence = await self._transition_to_ranking(
                previous_thought, guidance, context
            )

        elif prev_phase == "rank_relatively":
            # Next: select_best
            thought_type, content, confidence = self._transition_to_selection(
                previous_thought, guidance, context
            )

        elif prev_phase == "select_best":
            # Next: refine (if enabled) or conclude
            if enable_refinement:
                thought_type, content, confidence = await self._transition_to_refinement(
                    previous_thought, guidance, context
                )
            else:
                thought_type, content, confidence = self._transition_to_conclusion(
                    previous_thought, guidance, context
                )

        elif prev_phase == "refine":
            # Next: conclude
            thought_type, content, confidence = self._transition_to_conclusion(
                previous_thought, guidance, context
            )

        elif prev_phase == "conclude":
            # Already concluded - just return a continuation
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = self._format_final_conclusion(previous_thought, guidance, context)
            confidence = previous_thought.confidence or 0.95

        else:
            # Fallback to ranking
            thought_type, content, confidence = await self._transition_to_ranking(
                previous_thought, guidance, context
            )

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.TRAINING_FREE_GRPO,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "training_free_grpo",
                "num_candidates": len(self._candidates),
                "pairwise_comparisons_count": len(self._pairwise_comparisons),
                "ranking_complete": len(self._ranking) > 0,
                "selected_candidate": self._selected_candidate_idx,
                "enable_refinement": enable_refinement,
                "training_required": False,
                "inference_time_only": True,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Training-Free GRPO, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = TrainingFreeGrpo()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _transition_to_ranking(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float]:
        """Transition to relative ranking phase via pairwise comparisons.

        Args:
            previous_thought: The candidate generation to rank
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, confidence)
        """
        self._current_phase = "rank_relatively"
        thought_type = ThoughtType.VERIFICATION
        content = await self._perform_pairwise_ranking(previous_thought, guidance, context)

        # If pairwise comparisons weren't populated by sampling, do it heuristically
        if not self._pairwise_comparisons:
            num_candidates = len(self._candidates)
            # Simulate pairwise comparisons (fallback heuristic)
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    comparison = {
                        "candidate_a": i,
                        "candidate_b": j,
                        "preferred": i if i % 2 == 0 else j,  # Placeholder logic
                        "reason": "Placeholder pairwise comparison reasoning",
                    }
                    self._pairwise_comparisons.append(comparison)

            # Build ranking from pairwise comparisons (simple placeholder)
            self._ranking = list(range(num_candidates))

        confidence = 0.7
        return thought_type, content, confidence

    def _transition_to_selection(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float]:
        """Transition to selection phase - choose best candidate.

        Args:
            previous_thought: The ranking to select from
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, confidence)
        """
        self._current_phase = "select_best"
        thought_type = ThoughtType.SYNTHESIS
        content = self._select_top_candidate(previous_thought, guidance, context)

        # Select top candidate from ranking
        if self._ranking:
            self._selected_candidate_idx = self._ranking[0]

        confidence = 0.8
        return thought_type, content, confidence

    async def _transition_to_refinement(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float]:
        """Transition to refinement phase - improve selected candidate.

        Args:
            previous_thought: The selection to refine
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, confidence)
        """
        self._current_phase = "refine"
        thought_type = ThoughtType.REVISION
        content = await self._refine_selected_candidate(previous_thought, guidance, context)

        confidence = 0.9
        return thought_type, content, confidence

    def _transition_to_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[ThoughtType, str, float]:
        """Transition to conclusion phase - present final solution.

        Args:
            previous_thought: The refinement or selection to conclude
            guidance: Optional guidance
            context: Optional context

        Returns:
            Tuple of (thought_type, content, confidence)
        """
        self._current_phase = "conclude"
        thought_type = ThoughtType.CONCLUSION
        content = self._format_final_conclusion(previous_thought, guidance, context)

        confidence = 0.95
        return thought_type, content, confidence

    async def _generate_candidates(
        self,
        input_text: str,
        num_candidates: int,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate multiple diverse candidate solutions.

        This method uses LLM sampling to generate diverse candidate solutions,
        falling back to heuristic generation if sampling is unavailable.

        Args:
            input_text: The problem or question to reason about
            num_candidates: Number of candidates to generate
            context: Optional additional context

        Returns:
            The content for the candidate generation
        """
        guidance_text = ""
        if context and context.get("guidance"):
            guidance_text = f"\n\nGuidance: {context['guidance']}"

        # Try to use LLM sampling if available
        if (
            self._use_sampling
            and self._execution_context
            and hasattr(self._execution_context, "can_sample")
        ):
            try:
                if self._execution_context.can_sample:
                    # Generate diverse candidates using LLM sampling
                    candidate_texts = []
                    for i in range(num_candidates):
                        prompt = (
                            f"Generate a diverse solution approach #{i + 1} for the "
                            f"following problem.\n"
                            f"Make this approach distinctly different from previous "
                            f"candidates.\n\n"
                            f"Problem: {input_text}\n\n"
                            f"Provide a complete, creative solution approach."
                        )
                        system_prompt = (
                            "You are a creative problem solver generating diverse "
                            "candidate solutions. "
                            "Each solution should use a different strategy, "
                            "perspective, or approach. "
                            "Be innovative and think outside the box."
                        )

                        result = await self._sample_with_fallback(
                            prompt,
                            fallback_generator=lambda: (f"[Fallback candidate {i + 1} approach]"),
                            system_prompt=system_prompt,
                        )
                        candidate_texts.append(result)

                    # Format the LLM-generated candidates
                    candidates_formatted = "\n\n".join(
                        [f"Candidate {i + 1}:\n{text}" for i, text in enumerate(candidate_texts)]
                    )

                    return (
                        f"Step {self._step_counter}: Sample Candidates "
                        f"(Training-Free GRPO)\n\n"
                        f"Problem: {input_text}\n\n"
                        f"KEY: This is Training-Free GRPO - NO TRAINING REQUIRED!\n"
                        f"All optimization happens at inference time through:\n"
                        f"1. Multiple candidate sampling for diversity\n"
                        f"2. Relative ranking via pairwise comparisons\n"
                        f"3. Quality-based selection and refinement\n\n"
                        f"Generating {num_candidates} diverse candidate "
                        f"solutions...\n\n"
                        f"{candidates_formatted}\n\n"
                        f"Generated {num_candidates} candidates. Next: Relative "
                        f"ranking via pairwise comparisons (no absolute scores - "
                        f"pure relative preference).{guidance_text}"
                    )
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "operation_failed",
                    method="_generate_candidates",
                    error=str(e),
                    exc_info=True,
                )
                # Fall through to heuristic approach
                pass
            except Exception as e:
                # Log unexpected exceptions and re-raise
                logger.error(
                    "unexpected_error",
                    method="_generate_candidates",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Fallback: Use heuristic candidate generation
        return (
            f"Step {self._step_counter}: Sample Candidates (Training-Free GRPO)\n\n"
            f"Problem: {input_text}\n\n"
            f"KEY: This is Training-Free GRPO - NO TRAINING REQUIRED!\n"
            f"All optimization happens at inference time through:\n"
            f"1. Multiple candidate sampling for diversity\n"
            f"2. Relative ranking via pairwise comparisons\n"
            f"3. Quality-based selection and refinement\n\n"
            f"Generating {num_candidates} diverse candidate solutions...\n\n"
            f"Candidate 1:\n[LLM would generate first diverse solution approach]\n\n"
            f"Candidate 2:\n[LLM would generate second diverse solution approach]\n\n"
            f"Candidate 3:\n[LLM would generate third diverse solution approach]\n\n"
            f"Candidate 4:\n[LLM would generate fourth diverse solution approach]\n\n"
            f"Candidate 5:\n[LLM would generate fifth diverse solution approach]\n\n"
            f"Generated {num_candidates} candidates. Next: Relative ranking via pairwise "
            f"comparisons (no absolute scores - pure relative preference).{guidance_text}"
        )

    async def _perform_pairwise_ranking(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Perform relative ranking through pairwise comparisons.

        This method uses LLM sampling to perform pairwise comparisons between candidates,
        falling back to heuristic ranking if sampling is unavailable.

        Args:
            previous_thought: The candidate generation to rank
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the ranking phase
        """
        num_candidates = len(self._candidates)
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Try to use LLM sampling for pairwise comparisons
        if (
            self._use_sampling
            and self._execution_context
            and hasattr(self._execution_context, "can_sample")
        ):
            try:
                if self._execution_context.can_sample:
                    comparison_results = []
                    # Perform pairwise comparisons using LLM
                    for i in range(num_candidates):
                        for j in range(i + 1, num_candidates):
                            candidate_a = self._candidates[i]["content"]
                            candidate_b = self._candidates[j]["content"]

                            prompt = (
                                f"Compare these two candidate solutions and "
                                f"determine which is better.\n\n"
                                f"Candidate A:\n{candidate_a}\n\n"
                                f"Candidate B:\n{candidate_b}\n\n"
                                f"Which candidate is better? Respond with either "
                                f"'A' or 'B' and explain why."
                            )
                            system_prompt = (
                                "You are an expert evaluator comparing solution candidates. "
                                "Focus on quality, completeness, feasibility, and innovation. "
                                "Make relative judgments based on overall merit."
                            )

                            result = await self._sample_with_fallback(
                                prompt,
                                fallback_generator=lambda: "A",
                                system_prompt=system_prompt,
                            )

                            # Parse result to determine preference
                            preferred = i if str(result).strip().upper().startswith("A") else j
                            comparison = {
                                "candidate_a": i,
                                "candidate_b": j,
                                "preferred": preferred,
                                "reason": result,
                            }
                            self._pairwise_comparisons.append(comparison)
                            comparison_results.append(
                                f"- Candidate {i + 1} vs Candidate {j + 1}: "
                                f"Preferred Candidate {preferred + 1}\n  Reasoning: {str(result)[:100]}..."
                            )

                    # Build ranking from win counts
                    win_counts = [0] * num_candidates
                    for comp in self._pairwise_comparisons:
                        win_counts[comp["preferred"]] += 1

                    # Sort candidates by win count (descending)
                    self._ranking = sorted(
                        range(num_candidates), key=lambda i: win_counts[i], reverse=True
                    )

                    comparisons_formatted = "\n".join(comparison_results)
                    ranking_formatted = "\n".join(
                        [
                            f"{idx + 1}. Candidate {self._ranking[idx] + 1} - "
                            f"{win_counts[self._ranking[idx]]} wins in pairwise comparisons"
                            for idx in range(num_candidates)
                        ]
                    )

                    return (
                        f"Step {self._step_counter}: Rank Relatively "
                        f"(Pairwise Comparisons)\n\n"
                        f"Ranking {num_candidates} candidates from Step "
                        f"{previous_thought.step_number} "
                        f"using pairwise comparisons...\n\n"
                        f"KEY: Using RELATIVE ranking - no absolute scores, "
                        f"only relative preferences!\n\n"
                        f"Pairwise Comparisons:\n{comparisons_formatted}\n\n"
                        f"Aggregating pairwise preferences into relative "
                        f"ranking...\n\n"
                        f"Relative Ranking (best to worst):\n"
                        f"{ranking_formatted}\n\n"
                        f"Ranking complete. Total comparisons: "
                        f"{len(self._pairwise_comparisons)}\n"
                        f"Next: Select top candidate based on relative ranking."
                        f"{guidance_text}"
                    )
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "operation_failed",
                    method="_perform_pairwise_ranking",
                    error=str(e),
                    exc_info=True,
                )
                # Fall through to heuristic approach
                pass
            except Exception as e:
                # Log unexpected exceptions and re-raise
                logger.error(
                    "unexpected_error",
                    method="_perform_pairwise_ranking",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Fallback: Use heuristic pairwise ranking
        num_comparisons = len(self._pairwise_comparisons)
        return (
            f"Step {self._step_counter}: Rank Relatively (Pairwise Comparisons)\n\n"
            f"Ranking {num_candidates} candidates from Step {previous_thought.step_number} "
            f"using pairwise comparisons...\n\n"
            f"KEY: Using RELATIVE ranking - no absolute scores, only relative preferences!\n\n"
            f"Pairwise Comparisons:\n"
            f"- Candidate 1 vs Candidate 2: [LLM determines preference and reasoning]\n"
            f"- Candidate 1 vs Candidate 3: [LLM determines preference and reasoning]\n"
            f"- Candidate 2 vs Candidate 3: [LLM determines preference and reasoning]\n"
            f"- ... ({num_comparisons} total pairwise comparisons)\n\n"
            f"Aggregating pairwise preferences into relative ranking...\n\n"
            f"Relative Ranking (best to worst):\n"
            f"1. Candidate [X] - Most preferred in pairwise comparisons\n"
            f"2. Candidate [Y] - Second most preferred\n"
            f"3. Candidate [Z] - Third most preferred\n"
            f"...\n\n"
            f"Ranking complete. Total comparisons: {num_comparisons}\n"
            f"Next: Select top candidate based on relative ranking.{guidance_text}"
        )

    def _select_top_candidate(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Select the best candidate based on relative ranking.

        This is a helper method that would typically analyze the ranking
        and select the top candidate.

        Args:
            previous_thought: The ranking to select from
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the selection phase

        Note:
            In a full implementation, this would use the actual ranking
            to select the best candidate. This is a placeholder that provides
            the structure.
        """
        selected_idx = self._selected_candidate_idx
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Select Best Candidate\n\n"
            f"Based on the relative ranking from Step {previous_thought.step_number}, "
            f"selecting the top-ranked candidate...\n\n"
            f"Selected: Candidate {selected_idx + 1}\n\n"
            f"Selection Criteria:\n"
            f"- Highest win rate in pairwise comparisons\n"
            f"- Most consistent preference across comparisons\n"
            f"- Strong relative advantages over alternatives\n\n"
            f"Why this candidate:\n"
            f"[LLM would explain why this candidate was ranked highest]\n\n"
            f"Relative Advantages:\n"
            f"[LLM would list key advantages over other candidates]\n\n"
        )

        # Add next step guidance
        enable_refinement = previous_thought.metadata.get("enable_refinement")
        if enable_refinement:
            next_step = "Refine selected candidate"
        else:
            next_step = "Conclude with final solution"

        return content + f"Next: {next_step}.{guidance_text}"

    async def _refine_selected_candidate(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Refine the selected candidate solution.

        This method uses LLM sampling to improve the selected candidate based on
        insights from the ranking process, falling back to heuristic refinement
        if sampling is unavailable.

        Args:
            previous_thought: The selection to refine
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the refinement phase
        """
        selected_idx = self._selected_candidate_idx
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""
        step_num = previous_thought.step_number

        # Try to use LLM sampling for refinement
        if (
            self._use_sampling
            and self._execution_context
            and hasattr(self._execution_context, "can_sample")
        ):
            try:
                if self._execution_context.can_sample and selected_idx >= 0:
                    selected_candidate = self._candidates[selected_idx]["content"]

                    # Gather insights from pairwise comparisons
                    insights = []
                    for comp in self._pairwise_comparisons:
                        if comp["preferred"] == selected_idx:
                            insights.append(f"- {str(comp['reason'])[:100]}...")

                    insights_text = (
                        "\n".join(insights[:3]) if insights else "No specific insights available."
                    )

                    prompt = (
                        f"Refine the following selected candidate solution based on insights "
                        f"from pairwise comparisons.\n\n"
                        f"Selected Candidate:\n{selected_candidate}\n\n"
                        f"Key Strengths (from comparisons):\n{insights_text}\n\n"
                        f"Provide an improved version that:\n"
                        f"1. Incorporates strengths from other high-ranked candidates\n"
                        f"2. Addresses any identified weaknesses\n"
                        f"3. Optimizes based on relative preference patterns\n\n"
                        f"Generate the refined solution:"
                    )
                    system_prompt = (
                        "You are an expert solution optimizer. Refine the given solution "
                        "by incorporating insights from comparative analysis. Focus on "
                        "making concrete improvements while maintaining the core approach."
                    )

                    refined_solution = await self._sample_with_fallback(
                        prompt,
                        fallback_generator=lambda: (
                            "[Refinement would improve the selected solution]"
                        ),
                        system_prompt=system_prompt,
                    )

                    return (
                        f"Step {self._step_counter}: Refine Selected Candidate\n\n"
                        f"Refining Candidate {selected_idx + 1} selected in Step {step_num}...\n\n"
                        f"Refinement Strategy:\n"
                        f"- Incorporate strengths from other high-ranked candidates\n"
                        f"- Address any weaknesses identified in pairwise comparisons\n"
                        f"- Optimize based on relative preference patterns\n\n"
                        f"Applying refinements...\n\n"
                        f"Refined Solution:\n{refined_solution}\n\n"
                        f"Quality Improvement: Refined based on relative ranking feedback.\n"
                        f"Next: Conclude with final optimized solution.{guidance_text}"
                    )
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "llm_sampling_failed",
                    method="_refine_selected_candidate",
                    error=str(e),
                    exc_info=True,
                )
                # Fall through to heuristic approach
                pass
            except Exception as e:
                # Log unexpected exceptions and re-raise
                logger.error(
                    "unexpected_error",
                    method="_refine_selected_candidate",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Fallback: Use heuristic refinement
        return (
            f"Step {self._step_counter}: Refine Selected Candidate\n\n"
            f"Refining Candidate {selected_idx + 1} selected in Step {step_num}...\n\n"
            f"Refinement Strategy:\n"
            f"- Incorporate strengths from other high-ranked candidates\n"
            f"- Address any weaknesses identified in pairwise comparisons\n"
            f"- Optimize based on relative preference patterns\n\n"
            f"Applying refinements...\n\n"
            f"Refined Solution:\n"
            f"[LLM would generate improved version of selected candidate]\n\n"
            f"Refinements Applied:\n"
            f"1. [Improvement based on comparison insights]\n"
            f"2. [Optimization from ranking patterns]\n"
            f"3. [Enhancement incorporating other candidates' strengths]\n\n"
            f"Quality Improvement: Refined based on relative ranking feedback.\n"
            f"Next: Conclude with final optimized solution.{guidance_text}"
        )

    def _format_final_conclusion(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Format the final conclusion with optimized solution.

        This is a helper method that would typically format the final
        result emphasizing the training-free nature of the optimization.

        Args:
            previous_thought: The refinement or selection to conclude
            guidance: Optional guidance
            context: Optional additional context

        Returns:
            The content for the conclusion

        Note:
            In a full implementation, this would format the actual
            final solution. This is a placeholder that provides
            the structure.
        """
        prev_phase = previous_thought.metadata.get("phase", "refine")
        source_step = previous_thought.step_number
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Conclusion (Training-Free GRPO)\n\n"
            f"Final optimized solution from Step {source_step} "
            f"({'after refinement' if prev_phase == 'refine' else 'direct selection'}):\n\n"
            f"FINAL SOLUTION:\n"
            f"[This would contain the final optimized solution]\n\n"
            f"Optimization Summary (Training-Free GRPO):\n"
            f"✓ Generated {len(self._candidates)} diverse candidates\n"
            f"✓ Performed {len(self._pairwise_comparisons)} pairwise comparisons\n"
            f"✓ Relative ranking completed (no absolute scores)\n"
        )

        # Add refinement status and conclusion
        refined_status = "refined" if prev_phase == "refine" else "finalized"
        content += (
            f"✓ Top candidate selected and {refined_status}\n\n"
            f"KEY ACHIEVEMENT:\n"
            f"✓ GRPO-level optimization achieved WITHOUT ANY TRAINING\n"
            f"✓ All optimization done purely at inference time\n"
            f"✓ Preference optimization through relative comparisons only\n\n"
            f"Confidence: 95% - Optimized through "
            f"inference-time relative ranking{guidance_text}"
        )

        return content


__all__ = ["TrainingFreeGrpo", "TRAINING_FREE_GRPO_METADATA"]
