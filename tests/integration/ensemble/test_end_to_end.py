"""End-to-end tests for ensemble reasoning feature.

These tests verify the complete ensemble reasoning workflow from configuration
through execution to result aggregation. They test real integration between
components without mocking.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.ensemble import (
    ConfidenceCalibrator,
    EnsembleAggregator,
    EnsembleOrchestrator,
)
from reasoning_mcp.ensemble.presets import ENSEMBLE_PRESETS, get_preset
from reasoning_mcp.ensemble.strategies import get_strategy
from reasoning_mcp.methods.native.ensemble_reasoning import EnsembleReasoning
from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    EnsembleResult,
    MemberResult,
    VotingStrategy,
)
from reasoning_mcp.tools.ensemble import EnsembleToolInput, ensemble_reason


class TestEnsembleE2E:
    """End-to-end tests for ensemble reasoning."""

    async def test_basic_ensemble_execution(self) -> None:
        """Test basic ensemble with multiple methods."""
        config = EnsembleConfig(
            members=[
                EnsembleMember(method_name="chain_of_thought"),
                EnsembleMember(method_name="tree_of_thoughts"),
            ],
            strategy=VotingStrategy.MAJORITY,
        )
        orchestrator = EnsembleOrchestrator(config)
        result = await orchestrator.execute("What is 2+2?")

        assert result.final_answer is not None
        assert 0 <= result.confidence <= 1
        assert 0 <= result.agreement_score <= 1
        assert len(result.member_results) >= 1

    async def test_preset_configurations(self) -> None:
        """Test all preset configurations work."""
        for preset_name in ENSEMBLE_PRESETS:
            config = get_preset(preset_name)
            orchestrator = EnsembleOrchestrator(config)
            result = await orchestrator.execute("Test query")
            assert result.final_answer is not None
            assert isinstance(result, EnsembleResult)

    async def test_voting_strategies_with_mock_results(self) -> None:
        """Test each voting strategy works with mock data."""
        member1 = EnsembleMember(method_name="cot", weight=1.0)
        member2 = EnsembleMember(method_name="tot", weight=1.5)
        member3 = EnsembleMember(method_name="react", weight=1.0)

        mock_results = [
            MemberResult(member=member1, result="Answer A", confidence=0.9, execution_time_ms=100),
            MemberResult(member=member2, result="Answer A", confidence=0.85, execution_time_ms=150),
            MemberResult(member=member3, result="Answer B", confidence=0.75, execution_time_ms=120),
        ]

        for strategy in VotingStrategy:
            if strategy == VotingStrategy.SYNTHESIS:
                continue  # Skip synthesis as it requires LLM context

            strategy_impl = get_strategy(strategy)
            answer, confidence, details = strategy_impl.aggregate(mock_results)

            assert answer is not None
            assert isinstance(answer, str)
            assert 0 <= confidence <= 1
            assert isinstance(details, dict)

    async def test_ensemble_tool(self) -> None:
        """Test the MCP ensemble_reason tool."""
        input_data = EnsembleToolInput(
            query="What is the capital of France?",
            strategy=VotingStrategy.MAJORITY,
        )
        result = await ensemble_reason(input_data)
        assert result.final_answer is not None
        assert isinstance(result, EnsembleResult)

    async def test_ensemble_tool_with_custom_methods(self) -> None:
        """Test ensemble_reason tool with custom methods."""
        input_data = EnsembleToolInput(
            query="Solve this problem step by step",
            methods=["chain_of_thought", "react"],
            strategy=VotingStrategy.WEIGHTED,
            weights={"chain_of_thought": 2.0, "react": 1.0},
        )
        result = await ensemble_reason(input_data)
        assert result.final_answer is not None

    def test_calibration_integration(self) -> None:
        """Test confidence calibration with ensemble."""
        calibrator = ConfidenceCalibrator()

        calibrator.calibrate("cot", predicted=0.9, actual=1.0)
        calibrator.calibrate("cot", predicted=0.9, actual=1.0)
        calibrator.calibrate("cot", predicted=0.8, actual=0.0)

        calibrated = calibrator.get_calibrated_confidence("cot", 0.85)
        assert 0 <= calibrated <= 1

    async def test_aggregator_with_multiple_runs(self) -> None:
        """Test meta-aggregation of multiple ensemble runs."""
        aggregator = EnsembleAggregator()

        config = get_preset("balanced")
        orchestrator = EnsembleOrchestrator(config)

        for _ in range(3):
            result = await orchestrator.execute("What is machine learning?")
            aggregator.add_result(result)

        final = aggregator.aggregate_results()
        assert final.final_answer is not None
        assert 0 <= final.confidence <= 1
        assert len(final.member_results) > 0

    async def test_ensemble_reasoning_method(self) -> None:
        """Test EnsembleReasoning as a native method."""
        method = EnsembleReasoning()

        default_config = method.get_default_config()
        assert len(default_config.members) >= 2
        assert default_config.strategy == VotingStrategy.MAJORITY

        assert method.identifier == "ensemble_reasoning"
        assert method.name == "Ensemble Reasoning"

        # health_check returns False before initialization
        assert await method.health_check() is False
        await method.initialize()
        assert await method.health_check() is True

    async def test_weighted_voting_strategy(self) -> None:
        """Test weighted voting with different weights."""
        config = EnsembleConfig(
            members=[
                EnsembleMember(method_name="cot", weight=3.0),
                EnsembleMember(method_name="tot", weight=1.0),
            ],
            strategy=VotingStrategy.WEIGHTED,
        )
        orchestrator = EnsembleOrchestrator(config)
        result = await orchestrator.execute("Test weighted voting")

        assert result.final_answer is not None

    async def test_consensus_voting_strategy(self) -> None:
        """Test consensus voting with threshold."""
        config = EnsembleConfig(
            members=[
                EnsembleMember(method_name="cot"),
                EnsembleMember(method_name="tot"),
                EnsembleMember(method_name="react"),
            ],
            strategy=VotingStrategy.CONSENSUS,
            min_agreement=0.5,
        )
        orchestrator = EnsembleOrchestrator(config)
        result = await orchestrator.execute("Test consensus")

        assert result.final_answer is not None

    def test_preset_get_invalid(self) -> None:
        """Test error handling for invalid preset."""
        with pytest.raises(KeyError) as exc_info:
            get_preset("nonexistent_preset")
        assert "nonexistent_preset" in str(exc_info.value)

    def test_all_presets_valid(self) -> None:
        """Test all presets have valid configurations."""
        for name, config in ENSEMBLE_PRESETS.items():
            assert len(config.members) >= 1, f"Preset {name} has no members"
            assert config.strategy in VotingStrategy, f"Preset {name} has invalid strategy"
            assert config.timeout_ms > 0, f"Preset {name} has invalid timeout"

    async def test_agreement_score_calculation(self) -> None:
        """Test agreement score is calculated correctly."""
        config = EnsembleConfig(
            members=[
                EnsembleMember(method_name="cot"),
                EnsembleMember(method_name="tot"),
                EnsembleMember(method_name="react"),
            ],
            strategy=VotingStrategy.MAJORITY,
        )
        orchestrator = EnsembleOrchestrator(config)
        result = await orchestrator.execute("Agreement test")

        # Agreement score should be valid (0-1 range)
        # In placeholder mode each method returns unique result, so agreement is low
        assert 0 <= result.agreement_score <= 1
        # With 3 methods returning unique results, max agreement is 1/3
        assert result.agreement_score <= 0.5
