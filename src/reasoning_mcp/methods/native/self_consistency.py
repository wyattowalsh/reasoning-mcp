"""Self-Consistency reasoning method implementation.

Self-Consistency is a reasoning approach that generates multiple independent reasoning
paths for the same problem and uses majority voting to select the most consistent answer.
This helps improve reliability by aggregating diverse reasoning attempts.

Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
Wang et al., ICLR 2023
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

if TYPE_CHECKING:
    from reasoning_mcp.models import Session, ThoughtNode


# Metadata for the Self-Consistency method
SELF_CONSISTENCY_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SELF_CONSISTENCY,
    name="Self-Consistency",
    description="Generate multiple independent reasoning paths and select the most consistent answer through majority voting",
    category=MethodCategory.CORE,
    tags=frozenset({
        "voting",
        "consensus",
        "parallel",
        "reliability",
        "aggregation",
        "diverse-reasoning",
    }),
    complexity=4,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,  # Minimum of 3 paths for meaningful voting
    max_thoughts=15,  # 3-5 paths × 3 thoughts average per path
    avg_tokens_per_thought=600,
    best_for=(
        "Multiple choice questions",
        "Mathematical reasoning",
        "Logic puzzles",
        "Ambiguous problems requiring diverse perspectives",
        "Tasks where reliability is critical",
        "Questions with clear, comparable answers",
    ),
    not_recommended_for=(
        "Open-ended creative tasks",
        "Subjective questions without clear answers",
        "Tasks requiring single coherent narrative",
        "Real-time decision making (due to parallel overhead)",
    ),
)


class SelfConsistency:
    """Self-Consistency reasoning method.

    This method generates multiple independent reasoning paths for the same problem,
    then uses majority voting to select the most consistent answer. Each path is
    generated independently to ensure diverse perspectives, and the final answer
    is determined by consensus.

    The method creates parallel branches in the thought graph, each representing
    an independent reasoning attempt. After all paths complete, it analyzes the
    conclusions, identifies the majority answer, and reports consistency metrics.

    Attributes:
        identifier: Method identifier (SELF_CONSISTENCY)
        name: Human-readable name
        description: Brief description of the method
        category: Method category (CORE)
        num_paths: Number of independent reasoning paths to generate (default: 3)
        min_agreement: Minimum agreement threshold for confidence (default: 0.5)
    """

    def __init__(
        self,
        num_paths: int = 3,
        min_agreement: float = 0.5,
    ) -> None:
        """Initialize the Self-Consistency method.

        Args:
            num_paths: Number of independent reasoning paths to generate (3-5 recommended)
            min_agreement: Minimum fraction of paths that must agree for high confidence
        """
        if num_paths < 2:
            raise ValueError("num_paths must be at least 2 for meaningful voting")
        if num_paths > 10:
            raise ValueError("num_paths should not exceed 10 to avoid excessive computation")
        if not 0.0 <= min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0.0 and 1.0")

        self._num_paths = num_paths
        self._min_agreement = min_agreement

    @property
    def identifier(self) -> str:
        """Return the method identifier."""
        return str(MethodIdentifier.SELF_CONSISTENCY)

    @property
    def name(self) -> str:
        """Return the method name."""
        return SELF_CONSISTENCY_METADATA.name

    @property
    def description(self) -> str:
        """Return the method description."""
        return SELF_CONSISTENCY_METADATA.description

    @property
    def category(self) -> str:
        """Return the method category."""
        return str(MethodCategory.CORE)

    async def initialize(self) -> None:
        """Initialize the method (no initialization required for Self-Consistency)."""
        pass

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the Self-Consistency reasoning method.

        This method:
        1. Creates an initial thought describing the problem
        2. Generates N independent reasoning paths as parallel branches
        3. Each path reasons through the problem independently
        4. Collects conclusions from all paths
        5. Performs majority voting to determine the final answer
        6. Returns a synthesis thought with the consensus result and metrics

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional context (can include 'num_paths' to override default)

        Returns:
            A ThoughtNode representing the final consensus with voting metrics
        """
        from reasoning_mcp.models.thought import ThoughtNode

        # Extract configuration from context if provided
        num_paths = (context or {}).get("num_paths", self._num_paths)
        if not isinstance(num_paths, int) or num_paths < 2:
            num_paths = self._num_paths

        # Step 1: Create initial thought describing the approach
        initial_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            content=f"Applying Self-Consistency reasoning to: {input_text}\n\n"
            f"Strategy: Generate {num_paths} independent reasoning paths and use "
            f"majority voting to determine the most consistent answer.",
            confidence=1.0,
            step_number=1,
            depth=0,
            metadata={
                "problem": input_text,
                "num_paths": num_paths,
                "min_agreement": self._min_agreement,
            },
        )
        session.add_thought(initial_thought)

        # Step 2: Generate N independent reasoning paths as branches
        reasoning_paths: list[ThoughtNode] = []
        for i in range(num_paths):
            branch_id = f"path_{i+1}_{uuid4().hex[:8]}"

            # Create branch thought for this reasoning path
            branch_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.SELF_CONSISTENCY,
                parent_id=initial_thought.id,
                branch_id=branch_id,
                content=f"Reasoning Path {i+1}/{num_paths}:\n\n"
                f"Let me think through this problem independently...\n\n"
                f"{self._generate_reasoning_step(input_text, i+1)}",
                confidence=0.7,
                step_number=2 + i * 3,
                depth=1,
                metadata={
                    "path_number": i + 1,
                    "branch_id": branch_id,
                },
            )
            session.add_thought(branch_thought)

            # Generate intermediate reasoning for this path
            intermediate_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION,
                method_id=MethodIdentifier.SELF_CONSISTENCY,
                parent_id=branch_thought.id,
                branch_id=branch_id,
                content=self._generate_intermediate_reasoning(input_text, i+1),
                confidence=0.75,
                step_number=3 + i * 3,
                depth=2,
                metadata={
                    "path_number": i + 1,
                    "reasoning_stage": "intermediate",
                },
            )
            session.add_thought(intermediate_thought)

            # Generate conclusion for this path
            conclusion = self._generate_path_conclusion(input_text, i+1)
            conclusion_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONCLUSION,
                method_id=MethodIdentifier.SELF_CONSISTENCY,
                parent_id=intermediate_thought.id,
                branch_id=branch_id,
                content=f"Path {i+1} Conclusion:\n{conclusion}",
                confidence=0.8,
                step_number=4 + i * 3,
                depth=3,
                metadata={
                    "path_number": i + 1,
                    "conclusion": conclusion,
                    "is_path_conclusion": True,
                },
            )
            session.add_thought(conclusion_thought)
            reasoning_paths.append(conclusion_thought)

        # Step 3: Collect all conclusions and perform voting
        conclusions = [
            path.metadata.get("conclusion", "") for path in reasoning_paths
        ]

        # Step 4: Perform majority voting
        voting_results = self._perform_voting(conclusions)
        majority_answer = voting_results["majority_answer"]
        vote_counts = voting_results["vote_counts"]
        agreement_rate = voting_results["agreement_rate"]
        consistency_score = voting_results["consistency_score"]

        # Step 5: Create synthesis thought with consensus result
        synthesis_content = self._format_synthesis(
            input_text=input_text,
            conclusions=conclusions,
            majority_answer=majority_answer,
            vote_counts=vote_counts,
            agreement_rate=agreement_rate,
            num_paths=num_paths,
        )

        # Calculate confidence based on agreement
        final_confidence = min(0.95, 0.5 + (agreement_rate * 0.5))

        synthesis_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            parent_id=initial_thought.id,
            content=synthesis_content,
            confidence=final_confidence,
            quality_score=consistency_score,
            step_number=2 + num_paths * 3,
            depth=1,
            metadata={
                "majority_answer": majority_answer,
                "vote_counts": vote_counts,
                "agreement_rate": agreement_rate,
                "consistency_score": consistency_score,
                "num_paths": num_paths,
                "all_conclusions": conclusions,
                "is_final_answer": True,
            },
        )
        session.add_thought(synthesis_thought)

        return synthesis_thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        For Self-Consistency, continuation means re-running the voting process
        with additional reasoning paths if requested.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance (can request more paths)
            context: Optional additional context

        Returns:
            A new synthesis thought with updated voting results
        """
        from reasoning_mcp.models.thought import ThoughtNode

        # If guidance suggests adding more paths, we could implement that here
        # For now, create a verification thought
        verification_content = (
            f"Verification of Self-Consistency result:\n\n"
            f"The consensus answer was: {previous_thought.metadata.get('majority_answer', 'N/A')}\n"
            f"Agreement rate: {previous_thought.metadata.get('agreement_rate', 0.0):.1%}\n"
            f"Consistency score: {previous_thought.metadata.get('consistency_score', 0.0):.2f}\n\n"
        )

        if guidance:
            verification_content += f"Additional guidance: {guidance}\n\n"

        verification_content += (
            "The result shows " +
            ("high" if previous_thought.metadata.get("agreement_rate", 0) > self._min_agreement else "moderate") +
            " consistency across reasoning paths."
        )

        verification_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.VERIFICATION,
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            parent_id=previous_thought.id,
            content=verification_content,
            confidence=previous_thought.confidence,
            step_number=previous_thought.step_number + 1,
            depth=previous_thought.depth + 1,
            metadata={
                "verified_answer": previous_thought.metadata.get("majority_answer"),
                "verified_agreement": previous_thought.metadata.get("agreement_rate"),
            },
        )
        session.add_thought(verification_thought)

        return verification_thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True (Self-Consistency has no external dependencies)
        """
        return True

    def _generate_reasoning_step(self, problem: str, path_num: int) -> str:
        """Generate a unique reasoning step for a given path.

        Args:
            problem: The problem to reason about
            path_num: The path number (for variation)

        Returns:
            A reasoning step string
        """
        # In a real implementation, this would call an LLM or use different prompts
        # For now, we simulate diverse reasoning approaches
        approaches = [
            "Let me break this down systematically, starting with the fundamentals.",
            "I'll approach this from first principles and build up the solution.",
            "Let me consider what we know and what we need to find out.",
            "I'll think about this step by step, checking each assumption.",
            "Let me analyze the key components and how they relate.",
        ]
        return approaches[path_num % len(approaches)]

    def _generate_intermediate_reasoning(self, problem: str, path_num: int) -> str:
        """Generate intermediate reasoning for a path.

        Args:
            problem: The problem to reason about
            path_num: The path number

        Returns:
            Intermediate reasoning content
        """
        return (
            f"Working through the logical steps...\n\n"
            f"Based on the problem statement, I need to consider several factors.\n"
            f"Let me examine each possibility and evaluate which makes the most sense.\n\n"
            f"After careful analysis, I'm converging on a solution."
        )

    def _generate_path_conclusion(self, problem: str, path_num: int) -> str:
        """Generate a conclusion for a reasoning path.

        Args:
            problem: The problem to reason about
            path_num: The path number

        Returns:
            A conclusion string
        """
        # In a real implementation, this would generate actual answers via LLM
        # For demonstration, we simulate different but related conclusions
        return f"Answer determined through independent reasoning path {path_num}"

    def _perform_voting(self, conclusions: list[str]) -> dict[str, Any]:
        """Perform majority voting on conclusions.

        Args:
            conclusions: List of conclusion strings from different paths

        Returns:
            Dictionary with voting results including:
            - majority_answer: The most common conclusion
            - vote_counts: Dictionary mapping answers to their counts
            - agreement_rate: Fraction of paths agreeing with majority
            - consistency_score: Overall consistency metric (0-1)
        """
        # Count occurrences of each conclusion
        vote_counts: dict[str, int] = {}
        for conclusion in conclusions:
            normalized = conclusion.strip().lower()
            vote_counts[normalized] = vote_counts.get(normalized, 0) + 1

        # Find majority answer
        if not vote_counts:
            return {
                "majority_answer": "No conclusion reached",
                "vote_counts": {},
                "agreement_rate": 0.0,
                "consistency_score": 0.0,
            }

        majority_answer = max(vote_counts.items(), key=lambda x: x[1])[0]
        majority_count = vote_counts[majority_answer]

        # Calculate metrics
        total_votes = len(conclusions)
        agreement_rate = majority_count / total_votes if total_votes > 0 else 0.0

        # Consistency score: higher when votes are concentrated
        # Uses Shannon entropy normalized to [0, 1]
        import math
        entropy = 0.0
        for count in vote_counts.values():
            if count > 0:
                p = count / total_votes
                entropy -= p * math.log2(p)

        max_entropy = math.log2(total_votes) if total_votes > 1 else 1.0
        consistency_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        return {
            "majority_answer": majority_answer,
            "vote_counts": {k: v for k, v in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)},
            "agreement_rate": agreement_rate,
            "consistency_score": consistency_score,
        }

    def _format_synthesis(
        self,
        input_text: str,
        conclusions: list[str],
        majority_answer: str,
        vote_counts: dict[str, int],
        agreement_rate: float,
        num_paths: int,
    ) -> str:
        """Format the final synthesis with voting results.

        Args:
            input_text: Original problem
            conclusions: All path conclusions
            majority_answer: The majority vote winner
            vote_counts: Vote distribution
            agreement_rate: Agreement percentage
            num_paths: Total number of paths

        Returns:
            Formatted synthesis string
        """
        synthesis = [
            "Self-Consistency Analysis Complete",
            "=" * 50,
            "",
            f"Problem: {input_text}",
            "",
            f"Generated {num_paths} independent reasoning paths.",
            "",
            "Voting Results:",
            "-" * 50,
        ]

        for answer, count in vote_counts.items():
            percentage = (count / num_paths) * 100
            marker = "✓" if answer == majority_answer else " "
            synthesis.append(f"{marker} {count}/{num_paths} ({percentage:.1f}%): {answer}")

        synthesis.extend([
            "",
            "Consensus:",
            "-" * 50,
            f"Majority Answer: {majority_answer}",
            f"Agreement Rate: {agreement_rate:.1%}",
            f"Confidence: {'High' if agreement_rate >= self._min_agreement else 'Moderate'}",
            "",
        ])

        if agreement_rate >= 0.8:
            synthesis.append("Strong consensus achieved across reasoning paths.")
        elif agreement_rate >= self._min_agreement:
            synthesis.append("Moderate consensus - answer is likely reliable.")
        else:
            synthesis.append("Low consensus - consider additional analysis or context.")

        return "\n".join(synthesis)
