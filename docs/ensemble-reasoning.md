# Ensemble Reasoning

Ensemble reasoning combines outputs from multiple reasoning methods to produce more robust and accurate results. This feature leverages the complementary strengths of different approaches while mitigating their individual weaknesses.

## Overview

The ensemble reasoning system provides:

- **Multiple voting strategies** for combining results
- **Parallel execution** of reasoning methods
- **Confidence calibration** based on historical accuracy
- **Meta-aggregation** across multiple ensemble runs
- **Pre-configured presets** for common use cases

## Quick Start

### Using the MCP Tool

```python
from reasoning_mcp.tools.ensemble import ensemble_reason, EnsembleToolInput

# Basic usage with defaults
result = await ensemble_reason(EnsembleToolInput(
    query="What is the capital of France?"
))

# Custom configuration
result = await ensemble_reason(EnsembleToolInput(
    query="Solve this complex problem",
    methods=["chain_of_thought", "tree_of_thoughts", "react"],
    strategy=VotingStrategy.WEIGHTED,
    weights={"chain_of_thought": 2.0, "tree_of_thoughts": 1.5, "react": 1.0}
))
```

### Using the Orchestrator Directly

```python
from reasoning_mcp.ensemble import EnsembleOrchestrator
from reasoning_mcp.models.ensemble import EnsembleConfig, EnsembleMember, VotingStrategy

config = EnsembleConfig(
    members=[
        EnsembleMember(method_name="chain_of_thought", weight=1.5),
        EnsembleMember(method_name="tree_of_thoughts", weight=1.0),
        EnsembleMember(method_name="react", weight=1.0),
    ],
    strategy=VotingStrategy.MAJORITY,
    timeout_ms=30000,
)

orchestrator = EnsembleOrchestrator(config)
result = await orchestrator.execute("Your query here")

print(f"Answer: {result.final_answer}")
print(f"Confidence: {result.confidence}")
print(f"Agreement: {result.agreement_score}")
```

## Voting Strategies

### MAJORITY
Simple majority voting. The answer that appears most frequently wins.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.MAJORITY
)
```

**Best for:** Quick decisions, factual questions, when all methods are equally reliable.

### WEIGHTED
Weighted voting based on member weights and confidence scores.

```python
config = EnsembleConfig(
    members=[
        EnsembleMember(method_name="cot", weight=2.0),  # Higher weight
        EnsembleMember(method_name="tot", weight=1.0),
    ],
    strategy=VotingStrategy.WEIGHTED
)
```

**Best for:** When certain methods are more reliable for specific problem types.

### CONSENSUS
Requires a minimum agreement threshold before accepting an answer.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.CONSENSUS,
    min_agreement=0.7  # 70% agreement required
)
```

**Best for:** High-stakes decisions where certainty is important.

### BEST_SCORE
Selects the result with the highest confidence score.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.BEST_SCORE
)
```

**Best for:** When confidence scores are well-calibrated.

### SYNTHESIS
Uses an LLM to synthesize insights from all member results into a unified answer.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.SYNTHESIS
)
```

**Best for:** Complex questions requiring integration of multiple perspectives.

### RANKED_CHOICE
Instant runoff voting - eliminates lowest-ranked options iteratively.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.RANKED_CHOICE
)
```

**Best for:** When there are many distinct answers to choose from.

### BORDA_COUNT
Borda count scoring - assigns points based on ranking position.

```python
config = EnsembleConfig(
    members=[...],
    strategy=VotingStrategy.BORDA_COUNT
)
```

**Best for:** Preference aggregation, nuanced decision-making.

## Presets

Pre-configured ensemble settings for common scenarios:

```python
from reasoning_mcp.ensemble import get_preset

# Available presets: balanced, accuracy, speed, consensus
config = get_preset("balanced")
orchestrator = EnsembleOrchestrator(config)
```

### balanced
General-purpose configuration with multiple methods and majority voting.
- Methods: chain_of_thought, tree_of_thoughts, react
- Strategy: MAJORITY
- Timeout: 30s

### accuracy
Optimized for accuracy with more methods and weighted voting.
- Methods: chain_of_thought, tree_of_thoughts, self_consistency, react
- Strategy: WEIGHTED (higher weights for analytical methods)
- Timeout: 60s

### speed
Fast execution with fewer, simpler methods.
- Methods: chain_of_thought, react
- Strategy: BEST_SCORE
- Timeout: 15s

### consensus
High-confidence decisions requiring agreement.
- Methods: chain_of_thought, tree_of_thoughts, self_consistency
- Strategy: CONSENSUS
- Min Agreement: 0.7
- Timeout: 45s

## Confidence Calibration

The `ConfidenceCalibrator` learns calibration adjustments based on historical accuracy:

```python
from reasoning_mcp.ensemble import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Record historical accuracy
calibrator.calibrate("chain_of_thought", predicted=0.9, actual=1.0)
calibrator.calibrate("chain_of_thought", predicted=0.9, actual=1.0)
calibrator.calibrate("chain_of_thought", predicted=0.8, actual=0.0)

# Get calibrated confidence for new predictions
calibrated = calibrator.get_calibrated_confidence("chain_of_thought", 0.85)
```

## Meta-Aggregation

For running multiple ensemble passes and aggregating results:

```python
from reasoning_mcp.ensemble import EnsembleAggregator, EnsembleOrchestrator, get_preset

aggregator = EnsembleAggregator()
orchestrator = EnsembleOrchestrator(get_preset("balanced"))

# Run multiple passes
for _ in range(3):
    result = await orchestrator.execute("Your query")
    aggregator.add_result(result)

# Get final aggregated result
final = aggregator.aggregate_results()
```

## Native Method

Ensemble reasoning is also available as a native reasoning method:

```python
from reasoning_mcp.methods.native.ensemble_reasoning import EnsembleReasoning

method = EnsembleReasoning()
await method.initialize()

# Execute through the standard method interface
thought = await method.execute(session, "Your query", context={
    "strategy": "weighted",
    "methods": ["cot", "tot", "react"]
})
```

## Configuration

### Environment/Config Settings

```toml
# In config.toml or environment variables
[ensemble]
enabled = true
default_strategy = "majority"
default_timeout_ms = 30000
max_parallel_methods = 5
```

### EnsembleConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `members` | List[EnsembleMember] | Required | List of ensemble members |
| `strategy` | VotingStrategy | MAJORITY | Voting strategy to use |
| `timeout_ms` | int | 30000 | Execution timeout in milliseconds |
| `min_agreement` | float | 0.5 | Minimum agreement for CONSENSUS |
| `fail_fast` | bool | False | Stop on first member failure |
| `parallel` | bool | True | Execute members in parallel |

### EnsembleMember Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method_name` | str | Required | Name of the reasoning method |
| `weight` | float | 1.0 | Weight for weighted voting |
| `config` | dict | None | Method-specific configuration |
| `timeout_ms` | int | None | Per-member timeout override |

## API Reference

### Tools

- `ensemble_reason(input: EnsembleToolInput) -> EnsembleResult`: Execute ensemble reasoning
- `list_voting_strategies() -> list[dict]`: List available voting strategies

### Classes

- `EnsembleOrchestrator`: Main orchestration class
- `EnsembleAggregator`: Meta-aggregation of multiple runs
- `ConfidenceCalibrator`: Confidence score calibration
- `EnsembleReasoning`: Native reasoning method

### Models

- `EnsembleConfig`: Configuration for ensemble execution
- `EnsembleMember`: Individual member configuration
- `EnsembleResult`: Result from ensemble execution
- `MemberResult`: Result from individual member
- `VotingStrategy`: Enum of available strategies
- `VoteRecord`: Record of individual vote

## Best Practices

1. **Start with presets**: Use presets for common scenarios before customizing.

2. **Match strategies to problems**: 
   - Factual questions → MAJORITY
   - Complex reasoning → SYNTHESIS
   - High-stakes decisions → CONSENSUS

3. **Calibrate confidence**: Use the calibrator to improve accuracy over time.

4. **Monitor agreement scores**: Low agreement may indicate problem ambiguity.

5. **Balance speed vs accuracy**: Use `speed` preset for interactive scenarios, `accuracy` for batch processing.

6. **Leverage method strengths**:
   - CoT for step-by-step reasoning
   - ToT for exploration
   - ReAct for tool-use scenarios
   - Self-consistency for robustness
