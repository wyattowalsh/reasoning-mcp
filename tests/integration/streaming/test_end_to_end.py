"""End-to-end tests for streaming functionality."""

import asyncio

import pytest

from reasoning_mcp.streaming import (
    AsyncStreamEmitter,
    CompleteEvent,
    ProgressEvent,
    StreamingContext,
    ThoughtEvent,
)


class TestStreamingE2E:
    """End-to-end streaming tests."""

    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self):
        """Test complete streaming workflow from emission to consumption."""
        emitter = AsyncStreamEmitter(queue_size=100)
        ctx = StreamingContext(emitter=emitter, session_id="e2e-test")

        received_events = []

        async def consumer():
            async for event in emitter.subscribe():
                received_events.append(event)
                if isinstance(event, CompleteEvent):
                    break

        # Start consumer
        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        # Simulate reasoning workflow
        await ctx.emit_stage_start("initialization")
        await ctx.emit_thought("Analyzing the problem...", "test_method", confidence=0.7)
        await ctx.emit_progress(1, 3, "Step 1 complete")
        await ctx.emit_thought("Developing solution...", "test_method", confidence=0.8)
        await ctx.emit_progress(2, 3, "Step 2 complete")
        await ctx.emit_thought("Finalizing answer...", "test_method", confidence=0.95)
        await ctx.emit_progress(3, 3, "Step 3 complete")
        await ctx.emit_stage_end("initialization")
        await ctx.emit_complete({"answer": "42"}, total_duration_ms=1000)

        # Wait for consumer
        await asyncio.wait_for(consumer_task, timeout=2.0)

        # Verify events
        assert len(received_events) >= 9  # stage_start, 3 thoughts, 3 progress, stage_end, complete

        thought_events = [e for e in received_events if isinstance(e, ThoughtEvent)]
        assert len(thought_events) == 3

        progress_events = [e for e in received_events if isinstance(e, ProgressEvent)]
        assert len(progress_events) == 3
        assert progress_events[-1].percentage == 100.0

        complete_events = [e for e in received_events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].final_result == {"answer": "42"}

        await ctx.close()

    @pytest.mark.asyncio
    async def test_streaming_disabled(self):
        """Test that disabled streaming doesn't emit events."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter, enabled=False)

        # These should be no-ops
        await ctx.emit_thought("Test", "test")
        await ctx.emit_progress(1, 10)
        await ctx.emit_complete({}, 100)

        # Verify no events were emitted (metrics should show 0)
        metrics = emitter.metrics.get_metrics()
        assert metrics.events_emitted == 0

        await ctx.close()

    @pytest.mark.asyncio
    async def test_token_streaming(self):
        """Test streaming individual tokens."""
        emitter = AsyncStreamEmitter()
        ctx = StreamingContext(emitter=emitter)

        received = []

        async def consumer():
            async for event in emitter.subscribe():
                received.append(event)
                if len(received) >= 5:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        # Stream tokens
        tokens = ["The ", "answer ", "is ", "42", "."]
        for token in tokens:
            await ctx.emit_token(token)

        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 5
        assert received[-1].cumulative_text == "The answer is 42."

        await ctx.close()
