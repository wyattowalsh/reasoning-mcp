"""Background task support for reasoning-mcp.

This module provides utilities for long-running reasoning operations using
FastMCP v2.14+'s background task system with Docket backend support.

FastMCP v2.14+ Task Features:
- @mcp.tool(task=True) for background execution
- Task state tracking (pending, running, completed, failed)
- Redis backend support for persistence across restarts
- Memory backend for simple in-process tasks
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

# Default TTL for completed tasks (1 hour)
DEFAULT_COMPLETED_TASK_TTL_SECONDS = 3600

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult(BaseModel):
    """Result of a completed background task."""

    task_id: str = Field(description="Unique task identifier")
    status: TaskStatus = Field(description="Final task status")
    result: Any | None = Field(default=None, description="Task result if completed")
    error: str | None = Field(default=None, description="Error message if failed")
    started_at: datetime = Field(description="When the task started")
    completed_at: datetime | None = Field(default=None, description="When the task completed")
    duration_ms: float | None = Field(
        default=None, description="Execution duration in milliseconds"
    )


class TaskProgress(BaseModel):
    """Progress update for a running task."""

    task_id: str = Field(description="Unique task identifier")
    status: TaskStatus = Field(description="Current task status")
    progress: float = Field(ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")
    message: str = Field(default="", description="Current status message")
    steps_completed: int = Field(default=0, description="Number of steps completed")
    total_steps: int = Field(default=0, description="Total number of steps")


@dataclass
class TaskInfo:
    """Internal task tracking information."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    message: str = ""
    result: Any | None = None
    error: str | None = None
    steps_completed: int = 0
    total_steps: int = 0


@dataclass
class TaskManager:
    """Manager for background reasoning tasks.

    This class provides a simple in-memory task tracking system that works
    alongside FastMCP's built-in task support. It can be used to track
    progress and results of long-running reasoning operations.

    For production use with persistence, configure FastMCP with Redis backend.
    """

    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    max_tasks: int = 1000
    completed_task_ttl_seconds: int = DEFAULT_COMPLETED_TASK_TTL_SECONDS

    def create_task(
        self,
        task_id: str | None = None,
        total_steps: int = 0,
    ) -> TaskInfo:
        """Create a new task entry.

        Args:
            task_id: Optional task ID (generated if not provided)
            total_steps: Expected total number of steps

        Returns:
            TaskInfo for the new task
        """
        if task_id is None:
            task_id = str(uuid4())

        # Clean up expired and old tasks
        self._cleanup_expired_tasks()
        if len(self.tasks) >= self.max_tasks:
            self._cleanup_completed_tasks()

        task = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            total_steps=total_steps,
        )
        self.tasks[task_id] = task
        return task

    def start_task(self, task_id: str) -> TaskInfo | None:
        """Mark a task as running.

        Args:
            task_id: The task to start

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        return task

    def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str = "",
        steps_completed: int | None = None,
    ) -> TaskInfo | None:
        """Update task progress.

        Args:
            task_id: The task to update
            progress: Progress value (0.0 to 1.0)
            message: Status message
            steps_completed: Number of steps completed

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        task.progress = min(max(progress, 0.0), 1.0)
        task.message = message
        if steps_completed is not None:
            task.steps_completed = steps_completed

        return task

    def complete_task(
        self,
        task_id: str,
        result: Any = None,
    ) -> TaskInfo | None:
        """Mark a task as completed.

        Args:
            task_id: The task to complete
            result: Task result

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.progress = 1.0
        task.result = result
        return task

    def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> TaskInfo | None:
        """Mark a task as failed.

        Args:
            task_id: The task to fail
            error: Error message

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error = error
        return task

    def cancel_task(self, task_id: str) -> TaskInfo | None:
        """Cancel a running task.

        Args:
            task_id: The task to cancel

        Returns:
            Updated TaskInfo or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return task  # Cannot cancel finished tasks

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        return task

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information.

        Args:
            task_id: The task to retrieve

        Returns:
            TaskInfo or None if not found
        """
        return self.tasks.get(task_id)

    def get_progress(self, task_id: str) -> TaskProgress | None:
        """Get task progress as a Pydantic model.

        Args:
            task_id: The task to retrieve progress for

        Returns:
            TaskProgress model or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        return TaskProgress(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            message=task.message,
            steps_completed=task.steps_completed,
            total_steps=task.total_steps,
        )

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get task result as a Pydantic model.

        Args:
            task_id: The task to retrieve result for

        Returns:
            TaskResult model or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        duration_ms = None
        if task.started_at and task.completed_at:
            duration_ms = (task.completed_at - task.started_at).total_seconds() * 1000

        return TaskResult(
            task_id=task.task_id,
            status=task.status,
            result=task.result,
            error=task.error,
            started_at=task.started_at or task.created_at,
            completed_at=task.completed_at,
            duration_ms=duration_ms,
        )

    def _cleanup_completed_tasks(self) -> int:
        """Remove oldest completed tasks to make room.

        Returns:
            Number of tasks removed
        """
        completed = [
            (tid, t)
            for tid, t in self.tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]

        # Sort by completion time, oldest first
        completed.sort(key=lambda x: x[1].completed_at or x[1].created_at)

        # Remove oldest half of completed tasks
        to_remove = len(completed) // 2
        for task_id, _ in completed[:to_remove]:
            del self.tasks[task_id]

        return to_remove

    def _cleanup_expired_tasks(self) -> int:
        """Remove completed tasks that have exceeded their TTL.

        Returns:
            Number of tasks removed
        """
        if self.completed_task_ttl_seconds <= 0:
            return 0  # TTL disabled

        now = datetime.now()
        ttl = timedelta(seconds=self.completed_task_ttl_seconds)
        expired_ids: list[str] = []

        for task_id, task in self.tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                completion_time = task.completed_at or task.created_at
                if now - completion_time > ttl:
                    expired_ids.append(task_id)

        for task_id in expired_ids:
            del self.tasks[task_id]

        return len(expired_ids)


# Global task manager instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


async def run_reasoning_task(
    task_func: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    task_id: str | None = None,
    total_steps: int = 0,
    **kwargs: Any,
) -> TaskResult:
    """Run a reasoning function as a tracked background task.

    This is a helper for running reasoning operations with progress tracking.
    For FastMCP's built-in task support, use the @mcp.tool(task=True) decorator.

    Args:
        task_func: Async function to execute
        *args: Positional arguments for the function
        task_id: Optional task ID
        total_steps: Expected number of steps
        **kwargs: Keyword arguments for the function

    Returns:
        TaskResult with the outcome

    Example:
        >>> async def analyze_problem(problem: str) -> str:
        ...     # Long-running analysis
        ...     return "Analysis complete"
        ...
        >>> result = await run_reasoning_task(
        ...     analyze_problem,
        ...     "Complex problem to analyze",
        ...     total_steps=5,
        ... )
    """
    manager = get_task_manager()
    task = manager.create_task(task_id=task_id, total_steps=total_steps)
    manager.start_task(task.task_id)

    try:
        result = await task_func(*args, **kwargs)
        manager.complete_task(task.task_id, result=result)
    except asyncio.CancelledError:
        manager.cancel_task(task.task_id)
        raise
    except Exception as e:
        manager.fail_task(task.task_id, error=str(e))
        logger.exception(f"Task {task.task_id} failed: {e}")

    return manager.get_result(task.task_id)  # type: ignore


def register_background_tools(mcp: FastMCP) -> None:
    """Register background task management tools with FastMCP.

    These tools allow clients to check on task progress and retrieve results.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool(
        title="Get Task Progress",
        description="Get the progress of a background reasoning task",
    )
    def task_progress(task_id: str) -> TaskProgress | None:
        """Get the progress of a background task.

        Args:
            task_id: The task ID to check

        Returns:
            TaskProgress with current status, or None if not found
        """
        manager = get_task_manager()
        return manager.get_progress(task_id)

    @mcp.tool(
        title="Get Task Result",
        description="Get the result of a completed background reasoning task",
    )
    def task_result(task_id: str) -> TaskResult | None:
        """Get the result of a completed task.

        Args:
            task_id: The task ID to retrieve results for

        Returns:
            TaskResult with outcome, or None if not found
        """
        manager = get_task_manager()
        return manager.get_result(task_id)

    @mcp.tool(
        title="Cancel Task",
        description="Cancel a running background reasoning task",
    )
    def task_cancel(task_id: str) -> TaskProgress | None:
        """Cancel a running task.

        Args:
            task_id: The task ID to cancel

        Returns:
            TaskProgress showing cancelled status, or None if not found
        """
        manager = get_task_manager()
        manager.cancel_task(task_id)
        return manager.get_progress(task_id)


__all__ = [
    "TaskStatus",
    "TaskResult",
    "TaskProgress",
    "TaskInfo",
    "TaskManager",
    "get_task_manager",
    "run_reasoning_task",
    "register_background_tools",
]
