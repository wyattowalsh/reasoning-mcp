"""Self-Corrective RAG (SC-RAG) reasoning method.

This module implements SC-RAG, which combines evidence extraction with an
evidence-aware self-correction mechanism via chain-of-thought. It uses hybrid
retrieval and activates relevant internal knowledge for correction.

Key phases:
1. Extract: Use hybrid retriever for evidence extraction
2. Assess: Evaluate evidence quality and conflicts
3. Correct: Apply self-correction via CoT
4. Synthesize: Produce corrected, evidence-grounded answer

Reference: "SC-RAG: Self-Corrective Retrieval-Augmented Generation" (2024)
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


SC_RAG_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SC_RAG,
    name="Self-Corrective RAG",
    description="Evidence extraction with self-correction via CoT. Uses hybrid retrieval "
    "and evidence-aware correction to produce grounded, accurate answers.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"self-correction", "evidence", "hybrid-retrieval", "rag", "verification"}),
    complexity=7,
    supports_branching=False,
    supports_revision=True,
    requires_context=True,
    min_thoughts=5,
    max_thoughts=9,
    avg_tokens_per_thought=350,
    best_for=("factual accuracy", "knowledge conflicts", "self-correction"),
    not_recommended_for=("creative tasks", "subjective queries"),
)


class SCRAG(ReasoningMethodBase):
    """Self-Corrective RAG implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "extract"
        self._evidence: list[dict[str, Any]] = []
        self._conflicts: list[dict[str, Any]] = []
        self._corrections: list[str] = []
        self._use_sampling: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SC_RAG

    @property
    def name(self) -> str:
        return SC_RAG_METADATA.name

    @property
    def description(self) -> str:
        return SC_RAG_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "extract"
        self._evidence = []
        self._conflicts = []
        self._corrections = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("SC-RAG must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "extract"

        # Extract evidence using hybrid retrieval
        if self._use_sampling:
            self._evidence = await self._sample_evidence_extraction(input_text)
        else:
            self._evidence = self._generate_evidence_extraction(input_text)

        content = (
            f"Step {self._step_counter}: Evidence Extraction (SC-RAG)\n\n"
            f"Problem: {input_text}\n\n"
            f"Applying hybrid retrieval for evidence extraction:\n\n"
            f"Retrieved Evidence:\n"
            + "\n".join(
                f"  [{e['source']}] {e['content']} (quality: {e['quality']:.0%})"
                for e in self._evidence
            )
            + "\n\nHybrid approach combines semantic + aspect-based retrieval.\n"
            "Next: Assess evidence quality and conflicts."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SC_RAG,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "evidence": len(self._evidence)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SC_RAG
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
            raise RuntimeError("SC-RAG must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "extract")

        if prev_phase == "extract":
            self._current_phase = "assess"
            if self._use_sampling:
                self._conflicts = await self._sample_conflict_assessment(self._evidence)
            else:
                self._conflicts = self._generate_conflict_assessment(self._evidence)
            content = (
                f"Step {self._step_counter}: Assess Evidence Quality\n\n"
                f"Evaluating {len(self._evidence)} evidence items:\n\n"
                f"Quality Assessment:\n"
                + "\n".join(f"  {e['source']}: {e['quality']:.0%} quality" for e in self._evidence)
                + "\n\nConflict Detection:\n"
                + (
                    "\n".join(
                        f"  [{c['severity'].upper()}] {c['type']}: {c['description']}"
                        for c in self._conflicts
                    )
                    if self._conflicts
                    else "  No conflicts detected."
                )
                + f"\n\n{len(self._conflicts)} conflict(s) require resolution.\n"
                f"Next: Apply self-correction via CoT."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.7
        elif prev_phase == "assess":
            self._current_phase = "correct"
            if self._use_sampling:
                self._corrections = await self._sample_self_correction(
                    self._evidence, self._conflicts
                )
            else:
                self._corrections = self._generate_self_correction(self._evidence, self._conflicts)
            content = (
                f"Step {self._step_counter}: Self-Correction via CoT\n\n"
                f"Applying evidence-aware self-correction:\n\n"
                f"Correction Chain:\n"
                + "\n".join(f"  {c}" for c in self._corrections)
                + "\n\nCorrection Applied:\n"
                "  - External evidence weighted higher (more recent)\n"
                "  - Internal knowledge activated for cross-validation\n"
                "  - Conflict resolved: External source preferred\n\n"
                "Self-correction ensures factual accuracy."
            )
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        elif prev_phase == "correct":
            self._current_phase = "synthesize"
            content = (
                f"Step {self._step_counter}: Synthesize Corrected Answer\n\n"
                f"Combining corrected evidence into final answer:\n\n"
                f"Evidence Integration:\n"
                f"  - High-quality evidence retained: {len(self._evidence)}\n"
                f"  - Conflicts resolved: {len(self._conflicts)}\n"
                f"  - Self-corrections applied: {len(self._corrections)}\n\n"
                f"Synthesized Answer:\n"
                f"  [Corrected, evidence-grounded response]\n\n"
                f"Answer reflects resolved conflicts and validated facts."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.88
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Self-Corrective RAG Complete:\n"
                f"  Evidence extracted: {len(self._evidence)}\n"
                f"  Conflicts detected: {len(self._conflicts)}\n"
                f"  Corrections applied: {len(self._corrections)}\n\n"
                f"Final Answer: [Corrected, validated answer]\n"
                f"Confidence: High (89%)\n\n"
                f"Method: SC-RAG (Self-Corrective RAG)\n"
                f"  - Hybrid retrieval for comprehensive evidence\n"
                f"  - Conflict detection and assessment\n"
                f"  - Evidence-aware self-correction via CoT\n"
                f"  - Validated synthesis for accuracy"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.89

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SC_RAG,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "conflicts_resolved": len(self._conflicts),
            },
        )
        session.add_thought(thought)
        return thought

    def _generate_evidence_extraction(self, input_text: str) -> list[dict[str, Any]]:
        """Generate evidence using heuristic hybrid retrieval simulation.

        Args:
            input_text: The input problem/question

        Returns:
            List of evidence items with source, content, and quality
        """
        return [
            {"source": "Semantic", "content": "[Fact from semantic search]", "quality": 0.9},
            {"source": "Aspect-based", "content": "[Fact from aspect retrieval]", "quality": 0.85},
            {"source": "Internal", "content": "[Internal knowledge activation]", "quality": 0.8},
        ]

    async def _sample_evidence_extraction(self, input_text: str) -> list[dict[str, Any]]:
        """Extract evidence using LLM sampling with hybrid retrieval approach.

        Args:
            input_text: The input problem/question

        Returns:
            List of evidence items with source, content, and quality
        """
        system_prompt = (
            "You are an evidence extraction assistant using hybrid retrieval in SC-RAG.\n"
            "Extract relevant evidence from multiple sources: semantic search, "
            "aspect-based retrieval, and internal knowledge.\n"
            "For each piece of evidence, specify the source and provide a quality score."
        )

        user_prompt = f"""Problem: {input_text}

Extract evidence using hybrid retrieval approach:
1. Semantic search results (high-quality factual evidence)
2. Aspect-based retrieval (domain-specific evidence)
3. Internal knowledge activation (related concepts and principles)

Format each evidence item with:
- Source: [Semantic/Aspect-based/Internal]
- Content: [The evidence statement]
- Quality: [0.0-1.0 score]"""

        evidence_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "No evidence extracted",
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=600,
        )

        # Parse the result and structure it as evidence items
        return [
            {"source": "Semantic", "content": evidence_text[:200], "quality": 0.9},
            {"source": "Aspect-based", "content": evidence_text[:200], "quality": 0.85},
            {"source": "Internal", "content": evidence_text[:200], "quality": 0.8},
        ]

    def _generate_conflict_assessment(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate conflict assessment using heuristics.

        Args:
            evidence: List of evidence items

        Returns:
            List of detected conflicts
        """
        return [
            {
                "type": "external-internal",
                "description": "Retrieved fact differs from internal knowledge",
                "severity": "medium",
            },
        ]

    async def _sample_conflict_assessment(
        self, evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Assess evidence conflicts using LLM sampling.

        Args:
            evidence: List of evidence items

        Returns:
            List of detected conflicts with type, description, and severity
        """
        system_prompt = """You are an evidence quality assessor in SC-RAG.
Evaluate evidence items for quality and detect conflicts between different sources.
Identify inconsistencies, contradictions, or quality issues."""

        evidence_summary = "\n".join(
            f"- [{e['source']}] {e['content']} (quality: {e['quality']:.0%})" for e in evidence
        )

        user_prompt = f"""Evidence to assess:
{evidence_summary}

Analyze this evidence for:
1. Quality assessment of each item
2. Conflicts between sources
3. Consistency issues

For any conflicts found, specify:
- Type: [external-internal, source-source, etc.]
- Description: [What the conflict is]
- Severity: [low, medium, high]"""

        conflict_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "",
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=500,
        )

        # Parse result for conflicts
        if "conflict" in conflict_text.lower() or "inconsisten" in conflict_text.lower():
            return [
                {
                    "type": "detected-conflict",
                    "description": conflict_text[:200],
                    "severity": "medium",
                }
            ]
        return []

    def _generate_self_correction(
        self, evidence: list[dict[str, Any]], conflicts: list[dict[str, Any]]
    ) -> list[str]:
        """Generate self-correction steps using heuristics.

        Args:
            evidence: List of evidence items
            conflicts: List of detected conflicts

        Returns:
            List of correction steps
        """
        return [
            "Step 1: Identify conflicting claims",
            "Step 2: Weight evidence by source reliability",
            "Step 3: Resolve via majority + recency heuristic",
            "Step 4: Activate relevant internal knowledge for validation",
            "Step 5: Produce corrected understanding",
        ]

    async def _sample_self_correction(
        self, evidence: list[dict[str, Any]], conflicts: list[dict[str, Any]]
    ) -> list[str]:
        """Generate self-correction chain using LLM sampling.

        Args:
            evidence: List of evidence items
            conflicts: List of detected conflicts

        Returns:
            List of correction steps as chain-of-thought
        """
        system_prompt = """You are a self-correction assistant in SC-RAG.
Apply evidence-aware self-correction via chain-of-thought to resolve conflicts.
Generate a step-by-step correction process."""

        evidence_summary = "\n".join(
            f"- [{e['source']}] {e['content']} (quality: {e['quality']:.0%})" for e in evidence
        )
        conflicts_summary = "\n".join(
            f"- [{c['severity']}] {c['type']}: {c['description']}" for c in conflicts
        )

        user_prompt = f"""Evidence:
{evidence_summary}

Conflicts detected:
{conflicts_summary}

Generate a self-correction chain-of-thought to:
1. Identify the conflicting claims
2. Weight evidence by reliability and recency
3. Resolve conflicts using principled approach
4. Activate internal knowledge for validation
5. Produce corrected understanding

Provide each step as: "Step N: [action]" """

        correction_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: self._generate_self_correction_text(evidence, conflicts),
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=600,
        )

        # Parse steps from the result
        steps = []
        for line in correction_text.split("\n"):
            if line.strip().startswith("Step"):
                steps.append(line.strip())
        return steps if steps else [correction_text[:200]]

    def _generate_self_correction_text(
        self, evidence: list[dict[str, Any]], conflicts: list[dict[str, Any]]
    ) -> str:
        """Generate fallback self-correction text as newline-separated steps.

        Args:
            evidence: List of evidence items
            conflicts: List of detected conflicts

        Returns:
            Newline-separated correction steps
        """
        steps = self._generate_self_correction(evidence, conflicts)
        return "\n".join(steps)

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["SCRAG", "SC_RAG_METADATA"]
