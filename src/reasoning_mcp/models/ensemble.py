"""Ensemble models for reasoning-mcp.

This module defines models for ensemble reasoning, which combines multiple
reasoning methods to produce more robust and accurate results. It includes
voting strategies and member configuration models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VotingStrategy(StrEnum):
    """Voting strategies for ensemble decision-making.

    This enum defines different strategies for combining outputs from multiple
    reasoning methods in an ensemble. Each strategy has different properties
    and is suited for different types of problems.

    Strategies:
        MAJORITY: Simple majority vote - most common answer wins
        WEIGHTED: Weighted voting based on member weights
        CONSENSUS: Require agreement above a threshold
        BEST_SCORE: Select answer with highest confidence/quality score
        SYNTHESIS: Synthesize answers into a combined result
        RANKED_CHOICE: Ranked choice voting with preference ordering
        BORDA_COUNT: Borda count voting system with ranked preferences

    Examples:
        >>> strategy = VotingStrategy.MAJORITY
        >>> assert strategy == "majority"
        >>> assert strategy.value == "majority"

        >>> strategy = VotingStrategy.WEIGHTED
        >>> assert strategy in [VotingStrategy.WEIGHTED, VotingStrategy.MAJORITY]
    """

    MAJORITY = "majority"
    """Simple majority vote - the most common answer wins."""

    WEIGHTED = "weighted"
    """Weighted voting where each member's vote is multiplied by their weight."""

    CONSENSUS = "consensus"
    """Require agreement above a threshold for a decision to be made."""

    BEST_SCORE = "best_score"
    """Select the answer with the highest confidence or quality score."""

    SYNTHESIS = "synthesis"
    """Synthesize multiple answers into a combined, integrated result."""

    RANKED_CHOICE = "ranked_choice"
    """Ranked choice voting where members rank their preferences."""

    BORDA_COUNT = "borda_count"
    """Borda count voting system where points are assigned based on rank."""


class EnsembleMember(BaseModel):
    """Configuration for a single member of an ensemble.

    An EnsembleMember represents one reasoning method participating in an
    ensemble. Each member has a method name, optional weight for weighted
    voting, and optional method-specific configuration.

    Examples:
        Create a basic member:
        >>> member = EnsembleMember(method_name="chain_of_thought")
        >>> assert member.method_name == "chain_of_thought"
        >>> assert member.weight == 1.0
        >>> assert member.config is None

        Create a weighted member:
        >>> member = EnsembleMember(
        ...     method_name="tree_of_thoughts",
        ...     weight=2.0
        ... )
        >>> assert member.weight == 2.0

        Create a member with custom config:
        >>> member = EnsembleMember(
        ...     method_name="mcts",
        ...     weight=1.5,
        ...     config={
        ...         "max_iterations": 100,
        ...         "exploration_weight": 1.4,
        ...         "temperature": 0.7
        ...     }
        ... )
        >>> assert member.config["max_iterations"] == 100
        >>> assert member.config["exploration_weight"] == 1.4

        Create ensemble with multiple members:
        >>> members = [
        ...     EnsembleMember(method_name="chain_of_thought", weight=1.0),
        ...     EnsembleMember(method_name="tree_of_thoughts", weight=1.5),
        ...     EnsembleMember(method_name="self_consistency", weight=0.8),
        ... ]
        >>> total_weight = sum(m.weight for m in members)
        >>> assert total_weight == 3.3
    """

    method_name: str = Field(
        description="Name of the reasoning method for this ensemble member",
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight of this member in weighted voting (must be positive)",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional method-specific configuration parameters",
    )


class MemberResult(BaseModel):
    """Result from a single ensemble member execution.

    A MemberResult captures the output, confidence, and execution time
    from a single reasoning method within an ensemble. This information
    is used for voting, aggregation, and analysis.

    Examples:
        Create a basic result:
        >>> member = EnsembleMember(method_name="chain_of_thought")
        >>> result = MemberResult(
        ...     member=member,
        ...     result="The answer is 42",
        ...     confidence=0.95,
        ...     execution_time_ms=150
        ... )
        >>> assert result.result == "The answer is 42"
        >>> assert result.confidence == 0.95
        >>> assert result.execution_time_ms == 150

        Create multiple results:
        >>> results = [
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="cot"),
        ...         result="Answer A",
        ...         confidence=0.9,
        ...         execution_time_ms=100
        ...     ),
        ...     MemberResult(
        ...         member=EnsembleMember(method_name="tot"),
        ...         result="Answer A",
        ...         confidence=0.95,
        ...         execution_time_ms=250
        ...     ),
        ... ]
        >>> assert len(results) == 2
        >>> avg_confidence = sum(r.confidence for r in results) / len(results)
        >>> assert avg_confidence == 0.925
    """

    member: EnsembleMember = Field(
        description="The ensemble member that produced this result",
    )
    result: str = Field(
        description="The reasoning result text produced by this member",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this result (0.0 to 1.0)",
    )
    execution_time_ms: int = Field(
        ge=0,
        description="Execution time in milliseconds (must be non-negative)",
    )


class EnsembleConfig(BaseModel):
    """Configuration for ensemble reasoning.

    EnsembleConfig specifies which reasoning methods to use in an ensemble,
    how to combine their results, and what thresholds to apply for agreement
    and timeout.

    Examples:
        Create a basic ensemble config:
        >>> config = EnsembleConfig(
        ...     members=[
        ...         EnsembleMember(method_name="chain_of_thought"),
        ...         EnsembleMember(method_name="tree_of_thoughts"),
        ...     ]
        ... )
        >>> assert len(config.members) == 2
        >>> assert config.strategy == VotingStrategy.MAJORITY
        >>> assert config.min_agreement == 0.5

        Create weighted voting config:
        >>> config = EnsembleConfig(
        ...     members=[
        ...         EnsembleMember(method_name="cot", weight=1.0),
        ...         EnsembleMember(method_name="tot", weight=2.0),
        ...         EnsembleMember(method_name="mcts", weight=1.5),
        ...     ],
        ...     strategy=VotingStrategy.WEIGHTED,
        ...     min_agreement=0.6,
        ...     timeout_ms=60000
        ... )
        >>> assert config.strategy == VotingStrategy.WEIGHTED
        >>> assert config.min_agreement == 0.6
        >>> assert config.timeout_ms == 60000
    """

    members: list[EnsembleMember] = Field(
        description="List of ensemble members to participate in reasoning",
    )
    strategy: VotingStrategy = Field(
        default=VotingStrategy.MAJORITY,
        description="Voting strategy for combining member results",
    )
    min_agreement: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum agreement threshold for consensus (0.0 to 1.0)",
    )
    timeout_ms: int = Field(
        default=30000,
        gt=0,
        description="Timeout in milliseconds for ensemble execution (must be positive)",
    )


class EnsembleResult(BaseModel):
    """Result from ensemble reasoning execution.

    EnsembleResult contains the final aggregated answer from the ensemble,
    along with confidence scores, agreement metrics, and detailed results
    from each member for transparency and analysis.

    Examples:
        Create a basic result:
        >>> member1 = EnsembleMember(method_name="cot")
        >>> member2 = EnsembleMember(method_name="tot")
        >>> result = EnsembleResult(
        ...     final_answer="The answer is 42",
        ...     confidence=0.95,
        ...     agreement_score=0.85,
        ...     member_results=[
        ...         MemberResult(
        ...             member=member1,
        ...             result="42",
        ...             confidence=0.9,
        ...             execution_time_ms=100
        ...         ),
        ...         MemberResult(
        ...             member=member2,
        ...             result="42",
        ...             confidence=1.0,
        ...             execution_time_ms=200
        ...         ),
        ...     ],
        ...     voting_details={"strategy": "majority", "votes": {"42": 2}}
        ... )
        >>> assert result.final_answer == "The answer is 42"
        >>> assert result.confidence == 0.95
        >>> assert len(result.member_results) == 2
    """

    final_answer: str = Field(
        description="The final aggregated answer from the ensemble",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the final answer (0.0 to 1.0)",
    )
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score indicating level of agreement among members (0.0 to 1.0)",
    )
    member_results: list[MemberResult] = Field(
        description="Individual results from each ensemble member",
    )
    voting_details: dict[str, Any] = Field(
        description="Detailed voting information including strategy and vote counts",
    )


class VoteRecord(BaseModel):
    """Record of a single vote in ensemble decision-making.

    VoteRecord captures the details of one member's vote, including
    the vote itself, the weight applied, and optional reasoning for
    transparency and auditability.

    Examples:
        Create a basic vote record:
        >>> vote = VoteRecord(
        ...     member_name="chain_of_thought",
        ...     vote="Answer A",
        ...     weight=1.0
        ... )
        >>> assert vote.member_name == "chain_of_thought"
        >>> assert vote.vote == "Answer A"
        >>> assert vote.weight == 1.0
        >>> assert vote.reasoning is None

        Create a weighted vote with reasoning:
        >>> vote = VoteRecord(
        ...     member_name="tree_of_thoughts",
        ...     vote="Answer B",
        ...     weight=2.0,
        ...     reasoning="High confidence path through decision tree"
        ... )
        >>> assert vote.weight == 2.0
        >>> assert vote.reasoning is not None

        Aggregate votes:
        >>> votes = [
        ...     VoteRecord(member_name="cot", vote="A", weight=1.0),
        ...     VoteRecord(member_name="tot", vote="A", weight=2.0),
        ...     VoteRecord(member_name="mcts", vote="B", weight=1.0),
        ... ]
        >>> weighted_votes = {"A": 3.0, "B": 1.0}
        >>> assert max(weighted_votes, key=weighted_votes.get) == "A"
    """

    member_name: str = Field(
        description="Name of the ensemble member casting this vote",
    )
    vote: str = Field(
        description="The vote cast by this member",
    )
    weight: float = Field(
        gt=0.0,
        description="Weight applied to this vote (must be positive)",
    )
    reasoning: str | None = Field(
        default=None,
        description="Optional explanation for this vote",
    )
