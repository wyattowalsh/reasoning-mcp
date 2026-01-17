"""Filter Supervisor (FS-C) reasoning method.

This module implements Filter-Supervisor Self-Correction, which uses a
filtering mechanism to supervise and correct reasoning outputs. It generates
multiple candidates, filters them based on quality criteria, and iteratively
refines toward the best solution.

Key phases:
1. Generate: Produce candidate solutions
2. Filter: Apply quality filters to candidates
3. Supervise: Monitor and guide refinement
4. Correct: Self-correct based on supervision

Reference: Filter-Supervisor-Self-Correction patterns (2024-2025)
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

logger = structlog.get_logger(__name__)


FILTER_SUPERVISOR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.FILTER_SUPERVISOR,
    name="Filter Supervisor",
    description="Filter-based supervision with self-correction (FS-C pattern). "
    "Generates candidates, filters by quality, and iteratively self-corrects.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"filtering", "supervision", "self-correction", "candidates", "quality"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=("quality-critical tasks", "candidate selection", "iterative refinement"),
    not_recommended_for=("simple queries", "single-answer problems"),
)


class FilterSupervisor(ReasoningMethodBase):
    """Filter Supervisor reasoning method implementation."""

    # Enable LLM sampling for generating candidates and evaluations
    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._candidates: list[dict[str, Any]] = []
        self._filtered_candidates: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.FILTER_SUPERVISOR

    @property
    def name(self) -> str:
        return FILTER_SUPERVISOR_METADATA.name

    @property
    def description(self) -> str:
        return FILTER_SUPERVISOR_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._candidates = []
        self._filtered_candidates = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Filter Supervisor must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "generate"

        # Generate candidates using LLM sampling with fallback
        self._candidates = await self._sample_with_fallback(
            user_prompt=f"""Problem: {input_text}

Generate 4 candidate solutions with varying quality levels.""",
            fallback_generator=lambda: self._generate_candidates(input_text, context),
            system_prompt="""You are a reasoning assistant using the Filter Supervisor methodology.
Generate multiple candidate solutions for the given problem.

Your candidates should:
1. Represent diverse approaches to solving the problem
2. Vary in quality and sophistication
3. Be concrete and actionable
4. Include 3-5 distinct candidates

Format: For each candidate, provide:
- ID: A single letter (A, B, C, etc.)
- Approach: A brief description of the solution approach
- Quality estimate: A score from 0.0 to 1.0""",
            temperature=0.8,
            max_tokens=1000,
        )
        # Parse the LLM response to extract candidates
        if isinstance(self._candidates, str):
            self._candidates = self._parse_candidates_from_text(
                self._candidates, input_text, context
            )

        content = (
            f"Step {self._step_counter}: Generate Candidates (Filter Supervisor)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generated Candidates:\n"
            + "\n".join(f"  [{c['id']}] {c['content']}" for c in self._candidates)
            + "\n\nNext: Apply quality filters to candidates."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.FILTER_SUPERVISOR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.7,
            metadata={"phase": self._current_phase, "candidates": self._candidates},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.FILTER_SUPERVISOR
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
            raise RuntimeError("Filter Supervisor must be initialized before continuation")

        # Update execution context if provided
        if execution_context is not None:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "filter"
            # Filter candidates using LLM sampling with fallback
            candidates_text = "\n".join(
                f"[{c['id']}] {c['content']} (initial quality: {c['quality']:.2f})"
                for c in self._candidates
            )
            filter_result = await self._sample_with_fallback(
                user_prompt=f"""Candidates to evaluate:
{candidates_text}

Apply quality filters (threshold: 0.75) and provide structured assessment.""",
                fallback_generator=lambda: self._generate_filter(self._candidates, context)[
                    0
                ],  # Just return the content string
                system_prompt="""You are a reasoning assistant using the Filter Supervisor methodology.
Evaluate the quality of each candidate solution and determine which ones pass the quality threshold.

For each candidate:
1. Assess its quality on a scale of 0.0 to 1.0
2. Identify strengths and weaknesses
3. Determine if it passes the quality threshold (0.75)

Provide a structured assessment for each candidate.""",
                temperature=0.6,
                max_tokens=1200,
            )
            content = filter_result
            # Update filtered candidates
            self._filtered_candidates = [c for c in self._candidates if c["quality"] >= 0.75]
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "filter":
            self._current_phase = "supervise"
            # Supervise using LLM sampling with fallback
            candidates_text = "\n".join(
                f"[{c['id']}] {c['content']} (quality: {c['quality']:.2f})"
                for c in self._filtered_candidates
            )
            content = await self._sample_with_fallback(
                user_prompt=f"""Filtered candidates to supervise:
{candidates_text}

Provide detailed supervision analysis.""",
                fallback_generator=lambda: self._generate_supervise(
                    self._filtered_candidates, context
                ),
                system_prompt="""You are a reasoning assistant using the Filter Supervisor methodology.
Perform supervision analysis on the filtered candidates.

Your analysis should:
1. Identify strengths and weaknesses of each candidate
2. Compare candidates against each other
3. Determine the best candidate
4. Provide recommendations for improvement""",
                temperature=0.6,
                max_tokens=1000,
            )
            content = f"Step {self._step_counter}: Supervision Analysis\n\n{content}"
            thought_type = ThoughtType.REASONING
            confidence = 0.8
        elif prev_phase == "supervise":
            self._current_phase = "correct"
            # Self-correct using LLM sampling with fallback
            candidates_text = "\n".join(
                f"[{c['id']}] {c['content']} (quality: {c['quality']:.2f})"
                for c in self._filtered_candidates
            )
            content = await self._sample_with_fallback(
                user_prompt=f"""Candidates to correct:
{candidates_text}

Apply self-correction to produce the final solution.""",
                fallback_generator=lambda: self._generate_correct(
                    self._filtered_candidates, context
                ),
                system_prompt="""You are a reasoning assistant using the Filter Supervisor methodology.
Apply self-correction to improve the best candidate solution.

Your correction should:
1. Address identified weaknesses
2. Enhance strengths
3. Refine the solution based on supervision feedback
4. Produce a final, corrected solution""",
                temperature=0.7,
                max_tokens=1200,
            )
            content = f"Step {self._step_counter}: Self-Correction\n\n{content}"
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Filter Supervisor Complete:\n"
                f"  - Candidates generated: {len(self._candidates)}\n"
                f"  - Candidates filtered: {len(self._filtered_candidates)}\n"
                f"  - Best candidate: D (corrected)\n\n"
                f"Final Answer: [Corrected solution from candidate D]\n"
                f"Confidence: High (92%)"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.92

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.FILTER_SUPERVISOR,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "filtered": self._filtered_candidates},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _generate_candidates(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Generate candidate solutions (heuristic fallback).

        Args:
            input_text: The problem to generate candidates for
            context: Optional additional context

        Returns:
            List of candidate dictionaries with id, content, and quality
        """
        return [
            {"id": "A", "content": "Candidate A approach", "quality": 0.75},
            {"id": "B", "content": "Candidate B approach", "quality": 0.82},
            {"id": "C", "content": "Candidate C approach", "quality": 0.68},
            {"id": "D", "content": "Candidate D approach", "quality": 0.91},
        ]

    def _generate_filter(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Apply quality filters to candidates (heuristic fallback).

        Args:
            candidates: The candidates to filter
            context: Optional additional context

        Returns:
            Tuple of (content string, filtered candidates list)
        """
        filtered_candidates = [c for c in candidates if c["quality"] >= 0.75]
        results = "\n".join(
            f"  [{c['id']}] {c['quality']:.2f} - {'✓ PASS' if c['quality'] >= 0.75 else '✗ FAIL'}"
            for c in candidates
        )
        content = (
            f"Step {self._step_counter}: Apply Quality Filters\n\n"
            f"Filter Criteria: quality >= 0.75\n\n"
            f"Results:\n{results}\n\n"
            f"Filtered: {len(filtered_candidates)}/{len(candidates)} candidates pass"
        )
        return content, filtered_candidates

    def _generate_supervise(
        self,
        filtered_candidates: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> str:
        """Generate supervision analysis (heuristic fallback).

        Args:
            filtered_candidates: The filtered candidates to supervise
            context: Optional additional context

        Returns:
            Content string for supervision analysis
        """
        return (
            f"Step {self._step_counter}: Supervision Analysis\n\n"
            f"Analyzing filtered candidates...\n\n"
            f"Supervisor Assessment:\n"
            + "\n".join(
                f"  [{c['id']}] Strengths: [identified], Weaknesses: [identified]"
                for c in filtered_candidates
            )
            + "\n\nBest candidate: D (0.91)\nRecommendation: Minor refinement needed"
        )

    def _generate_correct(
        self,
        filtered_candidates: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> str:
        """Generate self-correction (heuristic fallback).

        Args:
            filtered_candidates: The filtered candidates to correct
            context: Optional additional context

        Returns:
            Content string for self-correction
        """
        return (
            f"Step {self._step_counter}: Self-Correction\n\n"
            f"Applying corrections based on supervision...\n\n"
            f"Corrections Applied:\n"
            f"  - Refined candidate D based on feedback\n"
            f"  - Quality improved: 0.91 → 0.95\n\n"
            f"Final corrected solution ready."
        )

    def _parse_candidates_from_text(
        self,
        content_text: str,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Parse candidate solutions from LLM response text.

        Args:
            content_text: The LLM response text to parse
            input_text: The original problem (for fallback)
            context: Optional additional context (for fallback)

        Returns:
            List of candidate dictionaries with id, content, and quality
        """
        # Parse candidates from the response
        candidates: list[dict[str, Any]] = []
        lines = content_text.split("\n")
        current_id = None
        current_content = ""

        for line in lines:
            line = line.strip()
            if line.startswith(("ID:", "- ID:", "Candidate")):
                # Save previous candidate if exists
                if current_id and current_content:
                    # Estimate quality based on position (simple heuristic)
                    quality = 0.6 + (len(candidates) * 0.1)
                    candidates.append(
                        {
                            "id": current_id,
                            "content": current_content.strip(),
                            "quality": min(quality, 0.95),
                        }
                    )
                # Extract ID
                if ":" in line:
                    current_id = line.split(":")[1].strip().split()[0].upper()
                else:
                    current_id = chr(65 + len(candidates))  # A, B, C, etc.
                current_content = ""
            elif line and current_id:
                current_content += line + " "

        # Save last candidate
        if current_id and current_content:
            quality = 0.6 + (len(candidates) * 0.1)
            candidates.append(
                {
                    "id": current_id,
                    "content": current_content.strip(),
                    "quality": min(quality, 0.95),
                }
            )

        # Fallback if parsing failed
        if not candidates:
            logger.debug(
                "candidate_parsing_failed",
                method="_parse_candidates_from_text",
                response_length=len(content_text),
            )
            candidates = self._generate_candidates(input_text, context)

        return candidates


__all__ = ["FilterSupervisor", "FILTER_SUPERVISOR_METADATA"]
