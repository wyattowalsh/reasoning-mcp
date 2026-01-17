"""Unit tests for sampling integration with reasoning methods.

Tests for:
- ExecutionContext sampling capability
- ReAct method sampling integration
- Fallback behavior when sampling unavailable
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reasoning_mcp.engine.executor import ExecutionContext
from reasoning_mcp.methods.native.react import ReActMethod
from reasoning_mcp.models import Session


class TestExecutionContextSampling:
    """Tests for ExecutionContext sampling capability."""

    def test_can_sample_without_ctx(self) -> None:
        """Test can_sample returns False when ctx is None."""
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
        )
        assert context.can_sample is False

    def test_can_sample_with_ctx(self) -> None:
        """Test can_sample returns True when ctx is present."""
        mock_ctx = MagicMock()
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
            ctx=mock_ctx,
        )
        assert context.can_sample is True

    @pytest.mark.asyncio
    async def test_sample_raises_without_ctx(self) -> None:
        """Test sample raises RuntimeError when ctx is None."""
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
        )
        with pytest.raises(RuntimeError, match="Sampling requires FastMCP Context"):
            await context.sample("Test prompt")

    @pytest.mark.asyncio
    async def test_sample_calls_sampling_module(self) -> None:
        """Test sample calls the sampling module correctly."""
        mock_ctx = MagicMock()
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
            ctx=mock_ctx,
        )

        # Patch where sample_reasoning_step is imported (inside the sample method)
        with patch(
            "reasoning_mcp.sampling.sample_reasoning_step",
            new_callable=AsyncMock,
        ) as mock_sample:
            mock_sample.return_value = "Sampled response"

            result = await context.sample(
                "Test prompt",
                system_prompt="Test system",
                temperature=0.5,
                max_tokens=100,
            )

            assert result == "Sampled response"
            mock_sample.assert_called_once()
            call_args = mock_sample.call_args
            assert call_args[0][0] == mock_ctx
            assert call_args[0][1] == "Test prompt"

    def test_with_update_preserves_ctx(self) -> None:
        """Test with_update preserves ctx when not overridden."""
        mock_ctx = MagicMock()
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
            ctx=mock_ctx,
        )

        new_context = context.with_update(metadata={"test": "value"})
        assert new_context.ctx is mock_ctx
        assert new_context.metadata == {"test": "value"}

    def test_with_update_can_override_ctx(self) -> None:
        """Test with_update can override ctx."""
        mock_ctx1 = MagicMock()
        mock_ctx2 = MagicMock()
        context = ExecutionContext(
            session=MagicMock(),
            registry=MagicMock(),
            input_data={},
            ctx=mock_ctx1,
        )

        new_context = context.with_update(ctx=mock_ctx2)
        assert new_context.ctx is mock_ctx2


class TestReActSamplingIntegration:
    """Tests for ReAct method sampling integration."""

    @pytest.fixture
    def method(self) -> ReActMethod:
        """Create and initialize a ReAct method."""
        m = ReActMethod()
        return m

    @pytest.fixture
    def active_session(self) -> Session:
        """Create an active session."""
        session = Session()
        session.start()
        return session

    @pytest.mark.asyncio
    async def test_execute_without_execution_context(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test execute works without execution_context (placeholder mode)."""
        await method.initialize()
        result = await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
        )

        # Should complete successfully in placeholder mode
        assert result is not None
        assert "Conclusion" in result.content

        # Verify sampled metadata is False for all thoughts
        for thought in active_session.graph.nodes.values():
            if "sampled" in thought.metadata:
                assert thought.metadata["sampled"] is False

    @pytest.mark.asyncio
    async def test_execute_with_mock_sampling(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test execute uses sampling when execution_context provides it."""
        await method.initialize()

        # Create mock execution context with sampling capability
        mock_ctx = MagicMock()
        mock_execution_context = MagicMock()
        mock_execution_context.can_sample = True
        mock_execution_context.ctx = mock_ctx

        # Mock the sample method
        sample_responses = iter(
            [
                "Analyzing the problem systematically",  # reasoning
                "TOOL: search\nACTION: Search for relevant information",  # action
                "Found relevant information about the topic",  # observation
                "Refined analysis based on findings",  # reasoning 2
                "TOOL: lookup\nACTION: Look up specific details",  # action 2
                "Confirmed the key findings",  # observation 2
                "Conclusion: The problem is solved",  # conclusion
            ]
        )

        async def mock_sample(*args: Any, **kwargs: Any) -> str:
            try:
                return next(sample_responses)
            except StopIteration:
                return "Default response"

        mock_execution_context.sample = mock_sample

        result = await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
            execution_context=mock_execution_context,
        )

        # Should complete successfully
        assert result is not None
        assert "Conclusion" in result.content

        # Verify sampled metadata is True for thoughts that used sampling
        sampled_thoughts = [
            t for t in active_session.graph.nodes.values() if t.metadata.get("sampled", False)
        ]
        assert len(sampled_thoughts) > 0

    @pytest.mark.asyncio
    async def test_sampling_fallback_on_error(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test fallback to placeholder when sampling raises exception."""
        await method.initialize()

        # Create mock execution context that raises on sample
        mock_ctx = MagicMock()
        mock_execution_context = MagicMock()
        mock_execution_context.can_sample = True
        mock_execution_context.ctx = mock_ctx

        async def mock_sample_error(*args: Any, **kwargs: Any) -> str:
            # Use TimeoutError as it's one of the expected exceptions that triggers fallback
            # (RuntimeError is not caught and would be re-raised as a potential bug)
            raise TimeoutError("Sampling timed out")

        mock_execution_context.sample = mock_sample_error

        result = await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
            execution_context=mock_execution_context,
        )

        # Should still complete successfully with fallback
        assert result is not None
        assert "Conclusion" in result.content


class TestReActSamplingHelpers:
    """Tests for ReAct sampling helper methods."""

    @pytest.fixture
    def method(self) -> ReActMethod:
        """Create and initialize a ReAct method."""
        m = ReActMethod()
        return m

    @pytest.fixture
    def active_session(self) -> Session:
        """Create an active session."""
        session = Session()
        session.start()
        return session

    @pytest.mark.asyncio
    async def test_generate_placeholder_reasoning(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test placeholder reasoning generation for different cycles."""
        await method.initialize()

        # Test cycle 1
        content, confidence = method._generate_placeholder_reasoning(
            input_text="Test problem",
            cycle=1,
            session=active_session,
        )
        assert "break this down" in content.lower()
        assert confidence == 0.7

        # Test cycle 2
        content, confidence = method._generate_placeholder_reasoning(
            input_text="Test problem",
            cycle=2,
            session=active_session,
        )
        assert "refine" in content.lower()
        assert confidence == 0.75

    @pytest.mark.asyncio
    async def test_generate_placeholder_action(self, method: ReActMethod) -> None:
        """Test placeholder action generation uses available tools."""
        await method.initialize()

        tools = ["custom_search", "custom_lookup"]

        # Test cycle 1 uses first tool
        content, tool = method._generate_placeholder_action(
            cycle=1,
            available_tools=tools,
        )
        assert tool == "custom_search"
        assert "custom_search" in content

        # Test cycle 2 uses second tool
        content, tool = method._generate_placeholder_action(
            cycle=2,
            available_tools=tools,
        )
        assert tool == "custom_lookup"
        assert "custom_lookup" in content

    @pytest.mark.asyncio
    async def test_generate_placeholder_observation(self, method: ReActMethod) -> None:
        """Test placeholder observation generation."""
        await method.initialize()

        # Test different cycles produce different confidence
        content1, conf1 = method._generate_placeholder_observation(
            cycle=1,
            tool_used="search",
        )
        content2, conf2 = method._generate_placeholder_observation(
            cycle=2,
            tool_used="lookup",
        )
        content3, conf3 = method._generate_placeholder_observation(
            cycle=3,
            tool_used="calculate",
        )

        assert conf1 < conf2 < conf3  # Confidence increases with cycles
        assert "search" in content1
        assert "lookup" in content2
        assert "calculate" in content3

    @pytest.mark.asyncio
    async def test_generate_placeholder_conclusion(self, method: ReActMethod) -> None:
        """Test placeholder conclusion generation."""
        await method.initialize()

        content = method._generate_placeholder_conclusion(
            input_text="Test problem",
            total_cycles=3,
            num_actions=3,
            num_observations=3,
        )

        assert "Conclusion" in content
        assert "3 cycles" in content
        assert "3 actions" in content
        assert "3 observations" in content


class TestSampledMetadata:
    """Tests for sampled metadata tracking."""

    @pytest.fixture
    def method(self) -> ReActMethod:
        """Create and initialize a ReAct method."""
        m = ReActMethod()
        return m

    @pytest.fixture
    def active_session(self) -> Session:
        """Create an active session."""
        session = Session()
        session.start()
        return session

    @pytest.mark.asyncio
    async def test_reasoning_thoughts_track_sampled(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test reasoning thoughts track whether they were sampled."""
        await method.initialize()

        await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
        )

        # Get reasoning thoughts
        from reasoning_mcp.models.core import ThoughtType

        reasoning_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.REASONING
        ]

        # All should have sampled=False without execution_context
        for thought in reasoning_thoughts:
            assert "sampled" in thought.metadata
            assert thought.metadata["sampled"] is False

    @pytest.mark.asyncio
    async def test_action_thoughts_track_sampled(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test action thoughts track whether they were sampled."""
        await method.initialize()

        await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
        )

        # Get action thoughts
        from reasoning_mcp.models.core import ThoughtType

        action_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.ACTION
        ]

        # All should have sampled=False without execution_context
        for thought in action_thoughts:
            assert "sampled" in thought.metadata
            assert thought.metadata["sampled"] is False

    @pytest.mark.asyncio
    async def test_observation_thoughts_track_sampled(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test observation thoughts track whether they were sampled."""
        await method.initialize()

        await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
        )

        # Get observation thoughts
        from reasoning_mcp.models.core import ThoughtType

        observation_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.OBSERVATION
        ]

        # All should have sampled=False without execution_context
        for thought in observation_thoughts:
            assert "sampled" in thought.metadata
            assert thought.metadata["sampled"] is False

    @pytest.mark.asyncio
    async def test_conclusion_tracks_sampled(
        self, method: ReActMethod, active_session: Session
    ) -> None:
        """Test conclusion thought tracks whether it was sampled."""
        await method.initialize()

        await method.execute(
            session=active_session,
            input_text="Test problem",
            context={"max_cycles": 2},
        )

        # Get conclusion thought
        from reasoning_mcp.models.core import ThoughtType

        conclusion_thoughts = [
            t for t in active_session.graph.nodes.values() if t.type == ThoughtType.CONCLUSION
        ]

        assert len(conclusion_thoughts) == 1
        assert "sampled" in conclusion_thoughts[0].metadata
        assert conclusion_thoughts[0].metadata["sampled"] is False
