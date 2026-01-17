"""MCP tools for ensemble reasoning operations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    EnsembleResult,
    VotingStrategy,
)


# Task 9.1: Input model
class EnsembleToolInput(BaseModel):
    """Input model for the ensemble_reason tool.

    Attributes:
        query: The query to reason about
        methods: List of method names to include (uses defaults if None)
        strategy: Voting strategy to use (default: MAJORITY)
        weights: Optional weights per method name
    """

    query: str = Field(..., description="The query to reason about")
    methods: list[str] | None = Field(
        default=None,
        description="Reasoning methods to use (uses defaults if not specified)",
    )
    strategy: VotingStrategy = Field(
        default=VotingStrategy.MAJORITY,
        description="Voting strategy for aggregation",
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for each method (method_name -> weight)",
    )
    timeout_ms: int = Field(
        default=30000,
        description="Timeout in milliseconds",
        gt=0,
    )


# Task 9.2: Main ensemble tool
async def ensemble_reason(input_data: EnsembleToolInput) -> EnsembleResult:
    """Perform ensemble reasoning with multiple methods.

    Executes multiple reasoning methods in parallel and aggregates
    their results using the specified voting strategy.

    Args:
        input_data: Configuration for ensemble execution

    Returns:
        EnsembleResult with aggregated answer and details
    """
    from reasoning_mcp.ensemble.orchestrator import EnsembleOrchestrator
    from reasoning_mcp.methods.native.ensemble_reasoning import EnsembleReasoning

    # Build members list
    if input_data.methods:
        members = [
            EnsembleMember(
                method_name=method,
                weight=input_data.weights.get(method, 1.0) if input_data.weights else 1.0,
            )
            for method in input_data.methods
        ]
    else:
        # Use default config
        default_config = EnsembleReasoning.get_default_config()
        members = default_config.members

    config = EnsembleConfig(
        members=members,
        strategy=input_data.strategy,
        timeout_ms=input_data.timeout_ms,
    )

    orchestrator = EnsembleOrchestrator(config=config)
    return await orchestrator.execute(input_data.query)


# Task 9.3: List strategies tool
def list_voting_strategies() -> list[dict[str, str]]:
    """List all available voting strategies with descriptions.

    Returns:
        List of strategy info dicts with 'name', 'value', 'description'
    """
    descriptions = {
        VotingStrategy.MAJORITY: "Simple majority voting - most common answer wins",
        VotingStrategy.WEIGHTED: "Weighted voting - considers member weights and confidence",
        VotingStrategy.CONSENSUS: "Requires minimum agreement threshold to accept answer",
        VotingStrategy.BEST_SCORE: "Selects result with highest confidence score",
        VotingStrategy.SYNTHESIS: "Uses LLM to synthesize all results into unified answer",
        VotingStrategy.RANKED_CHOICE: "Instant runoff voting with elimination rounds",
        VotingStrategy.BORDA_COUNT: "Borda count scoring based on rankings",
    }

    return [
        {
            "name": strategy.name,
            "value": strategy.value,
            "description": descriptions.get(strategy, ""),
        }
        for strategy in VotingStrategy
    ]


__all__ = ["EnsembleToolInput", "ensemble_reason", "list_voting_strategies"]
