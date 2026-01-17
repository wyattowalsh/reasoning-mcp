"""Unit tests for background task support module.

Tests for:
- TaskStatus enum values
- TaskResult model
- TaskProgress model
- TaskInfo dataclass
- TaskManager operations
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from reasoning_mcp.tasks import (
    TaskInfo,
    TaskManager,
    TaskProgress,
    TaskResult,
    TaskStatus,
    get_task_manager,
    run_reasoning_task,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_pending_status(self) -> None:
        """Test PENDING status value."""
        assert TaskStatus.PENDING == "pending"

    def test_running_status(self) -> None:
        """Test RUNNING status value."""
        assert TaskStatus.RUNNING == "running"

    def test_completed_status(self) -> None:
        """Test COMPLETED status value."""
        assert TaskStatus.COMPLETED == "completed"

    def test_failed_status(self) -> None:
        """Test FAILED status value."""
        assert TaskStatus.FAILED == "failed"

    def test_cancelled_status(self) -> None:
        """Test CANCELLED status value."""
        assert TaskStatus.CANCELLED == "cancelled"

    def test_all_status_values(self) -> None:
        """Test all status values exist."""
        statuses = [s.value for s in TaskStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "cancelled" in statuses


class TestTaskResult:
    """Tests for TaskResult Pydantic model."""

    def test_create_basic_result(self) -> None:
        """Test creating a basic TaskResult."""
        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(),
        )
        assert result.task_id == "test-123"
        assert result.status == TaskStatus.COMPLETED

    def test_result_with_all_fields(self) -> None:
        """Test TaskResult with all fields."""
        started = datetime.now()
        completed = datetime.now()
        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            result={"answer": 42},
            error=None,
            started_at=started,
            completed_at=completed,
            duration_ms=100.5,
        )
        assert result.result == {"answer": 42}
        assert result.error is None
        assert result.duration_ms == 100.5

    def test_result_with_error(self) -> None:
        """Test TaskResult with error."""
        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.FAILED,
            error="Something went wrong",
            started_at=datetime.now(),
        )
        assert result.status == TaskStatus.FAILED
        assert result.error == "Something went wrong"


class TestTaskProgress:
    """Tests for TaskProgress Pydantic model."""

    def test_create_basic_progress(self) -> None:
        """Test creating a basic TaskProgress."""
        progress = TaskProgress(
            task_id="test-123",
            status=TaskStatus.RUNNING,
            progress=0.5,
        )
        assert progress.task_id == "test-123"
        assert progress.status == TaskStatus.RUNNING
        assert progress.progress == 0.5

    def test_progress_with_steps(self) -> None:
        """Test TaskProgress with step information."""
        progress = TaskProgress(
            task_id="test-123",
            status=TaskStatus.RUNNING,
            progress=0.6,
            message="Processing step 3",
            steps_completed=3,
            total_steps=5,
        )
        assert progress.message == "Processing step 3"
        assert progress.steps_completed == 3
        assert progress.total_steps == 5

    def test_progress_bounds(self) -> None:
        """Test progress value bounds."""
        progress = TaskProgress(
            task_id="test-123",
            status=TaskStatus.RUNNING,
            progress=1.0,
        )
        assert progress.progress == 1.0

        progress_zero = TaskProgress(
            task_id="test-123",
            status=TaskStatus.PENDING,
            progress=0.0,
        )
        assert progress_zero.progress == 0.0


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_create_basic_info(self) -> None:
        """Test creating basic TaskInfo."""
        info = TaskInfo(
            task_id="test-123",
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
        )
        assert info.task_id == "test-123"
        assert info.status == TaskStatus.PENDING
        assert info.progress == 0.0
        assert info.message == ""

    def test_info_default_values(self) -> None:
        """Test TaskInfo default values."""
        info = TaskInfo(
            task_id="test-123",
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
        )
        assert info.started_at is None
        assert info.completed_at is None
        assert info.result is None
        assert info.error is None
        assert info.steps_completed == 0
        assert info.total_steps == 0


class TestTaskManager:
    """Tests for TaskManager class."""

    def test_create_task(self) -> None:
        """Test creating a task."""
        manager = TaskManager()
        task = manager.create_task()
        assert task.status == TaskStatus.PENDING
        assert task.task_id in manager.tasks

    def test_create_task_with_custom_id(self) -> None:
        """Test creating a task with custom ID."""
        manager = TaskManager()
        task = manager.create_task(task_id="custom-123")
        assert task.task_id == "custom-123"

    def test_create_task_with_total_steps(self) -> None:
        """Test creating a task with total steps."""
        manager = TaskManager()
        task = manager.create_task(total_steps=10)
        assert task.total_steps == 10

    def test_start_task(self) -> None:
        """Test starting a task."""
        manager = TaskManager()
        task = manager.create_task()
        started = manager.start_task(task.task_id)
        assert started is not None
        assert started.status == TaskStatus.RUNNING
        assert started.started_at is not None

    def test_start_nonexistent_task(self) -> None:
        """Test starting a nonexistent task."""
        manager = TaskManager()
        result = manager.start_task("nonexistent")
        assert result is None

    def test_update_progress(self) -> None:
        """Test updating task progress."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        updated = manager.update_progress(
            task.task_id,
            progress=0.5,
            message="Halfway done",
            steps_completed=5,
        )
        assert updated is not None
        assert updated.progress == 0.5
        assert updated.message == "Halfway done"
        assert updated.steps_completed == 5

    def test_update_progress_clamps_values(self) -> None:
        """Test progress values are clamped to [0, 1]."""
        manager = TaskManager()
        task = manager.create_task()
        manager.update_progress(task.task_id, progress=1.5)
        assert manager.tasks[task.task_id].progress == 1.0
        manager.update_progress(task.task_id, progress=-0.5)
        assert manager.tasks[task.task_id].progress == 0.0

    def test_complete_task(self) -> None:
        """Test completing a task."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        completed = manager.complete_task(task.task_id, result={"answer": 42})
        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"answer": 42}
        assert completed.progress == 1.0
        assert completed.completed_at is not None

    def test_fail_task(self) -> None:
        """Test failing a task."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        failed = manager.fail_task(task.task_id, error="Something went wrong")
        assert failed is not None
        assert failed.status == TaskStatus.FAILED
        assert failed.error == "Something went wrong"
        assert failed.completed_at is not None

    def test_cancel_task(self) -> None:
        """Test cancelling a task."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        cancelled = manager.cancel_task(task.task_id)
        assert cancelled is not None
        assert cancelled.status == TaskStatus.CANCELLED
        assert cancelled.completed_at is not None

    def test_cancel_completed_task(self) -> None:
        """Test cancelling a completed task has no effect."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        manager.complete_task(task.task_id)
        cancelled = manager.cancel_task(task.task_id)
        assert cancelled is not None
        assert cancelled.status == TaskStatus.COMPLETED

    def test_get_task(self) -> None:
        """Test getting a task by ID."""
        manager = TaskManager()
        manager.create_task(task_id="test-123")
        retrieved = manager.get_task("test-123")
        assert retrieved is not None
        assert retrieved.task_id == "test-123"

    def test_get_nonexistent_task(self) -> None:
        """Test getting a nonexistent task."""
        manager = TaskManager()
        result = manager.get_task("nonexistent")
        assert result is None

    def test_get_progress(self) -> None:
        """Test getting task progress as Pydantic model."""
        manager = TaskManager()
        task = manager.create_task(total_steps=10)
        manager.start_task(task.task_id)
        manager.update_progress(task.task_id, progress=0.3, steps_completed=3)
        progress = manager.get_progress(task.task_id)
        assert progress is not None
        assert isinstance(progress, TaskProgress)
        assert progress.progress == 0.3
        assert progress.steps_completed == 3
        assert progress.total_steps == 10

    def test_get_result(self) -> None:
        """Test getting task result as Pydantic model."""
        manager = TaskManager()
        task = manager.create_task()
        manager.start_task(task.task_id)
        manager.complete_task(task.task_id, result="Done!")
        result = manager.get_result(task.task_id)
        assert result is not None
        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "Done!"
        assert result.duration_ms is not None


class TestTaskManagerCleanup:
    """Tests for TaskManager cleanup behavior."""

    def test_cleanup_at_capacity(self) -> None:
        """Test cleanup occurs when at capacity."""
        manager = TaskManager(max_tasks=5)
        # Fill to capacity with completed tasks
        for i in range(5):
            task = manager.create_task(task_id=f"task-{i}")
            manager.start_task(task.task_id)
            manager.complete_task(task.task_id)
        # Add one more
        manager.create_task(task_id="task-5")
        # Some tasks should have been cleaned up
        assert len(manager.tasks) <= 5


class TestGetTaskManager:
    """Tests for get_task_manager function."""

    def test_returns_task_manager(self) -> None:
        """Test get_task_manager returns a TaskManager."""
        manager = get_task_manager()
        assert isinstance(manager, TaskManager)

    def test_returns_same_instance(self) -> None:
        """Test get_task_manager returns the same instance."""
        manager1 = get_task_manager()
        manager2 = get_task_manager()
        assert manager1 is manager2


class TestRunReasoningTask:
    """Tests for run_reasoning_task function."""

    @pytest.mark.asyncio
    async def test_run_successful_task(self) -> None:
        """Test running a successful async task."""

        async def success_func() -> str:
            return "Success!"

        result = await run_reasoning_task(success_func)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "Success!"

    @pytest.mark.asyncio
    async def test_run_failing_task(self) -> None:
        """Test running a failing async task."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        result = await run_reasoning_task(fail_func)
        assert result.status == TaskStatus.FAILED
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_run_task_with_args(self) -> None:
        """Test running a task with arguments."""

        async def add_func(a: int, b: int) -> int:
            return a + b

        result = await run_reasoning_task(add_func, 2, 3)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 5

    @pytest.mark.asyncio
    async def test_run_task_with_kwargs(self) -> None:
        """Test running a task with keyword arguments."""

        async def greet_func(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await run_reasoning_task(greet_func, "World", greeting="Hi")
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_run_task_with_custom_id(self) -> None:
        """Test running a task with custom ID."""

        async def simple_func() -> str:
            return "Done"

        result = await run_reasoning_task(simple_func, task_id="custom-task-123")
        assert result.task_id == "custom-task-123"

    @pytest.mark.asyncio
    async def test_run_task_with_total_steps(self) -> None:
        """Test running a task with total steps."""

        async def simple_func() -> str:
            return "Done"

        result = await run_reasoning_task(simple_func, total_steps=5)
        # Task should complete regardless of steps
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_task_has_duration(self) -> None:
        """Test task result includes duration."""

        async def slow_func() -> str:
            await asyncio.sleep(0.01)
            return "Done"

        result = await run_reasoning_task(slow_func)
        assert result.duration_ms is not None
        assert result.duration_ms >= 10  # At least 10ms
