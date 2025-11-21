"""Background task manager for async model training and other long-running operations."""

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

from recommendation_system.config import get_settings

logger = structlog.get_logger()


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingTask:
    """Represents a background training task."""

    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class BackgroundTaskManager:
    """Manages background tasks for model training and other async operations."""

    def __init__(self, max_workers: int = 4):
        """Initialize task manager.

        Args:
            max_workers: Maximum concurrent background workers.
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TrainingTask] = {}
        self._lock = threading.Lock()
        self._running_count = 0

        logger.info("Background task manager initialized", max_workers=max_workers)

    def submit_task(
        self,
        task_type: str,
        func: Callable,
        *args,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> TrainingTask:
        """Submit a task for background execution.

        Args:
            task_type: Type of task (e.g., 'model_training', 'data_import').
            func: Function to execute.
            *args: Positional arguments for the function.
            metadata: Additional metadata to store with the task.
            **kwargs: Keyword arguments for the function.

        Returns:
            TrainingTask object for tracking.
        """
        task_id = str(uuid.uuid4())
        task = TrainingTask(
            task_id=task_id,
            task_type=task_type,
            metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task_id] = task

        # Submit to executor
        future = self._executor.submit(
            self._execute_task,
            task_id,
            func,
            args,
            kwargs,
        )

        logger.info("Task submitted", task_id=task_id, task_type=task_type)
        return task

    def _execute_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Execute a task and update its status."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self._running_count += 1

        logger.info("Task started", task_id=task_id)

        try:
            # Pass progress callback to the function if it accepts it
            if "progress_callback" in kwargs or hasattr(func, "__code__") and "progress_callback" in func.__code__.co_varnames:
                kwargs["progress_callback"] = lambda p: self._update_progress(task_id, p)

            result = func(*args, **kwargs)

            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 100.0
                task.completed_at = datetime.utcnow()
                self._running_count -= 1

            logger.info("Task completed", task_id=task_id)
            return result

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.utcnow()
                self._running_count -= 1

            logger.error("Task failed", task_id=task_id, error=str(e))
            raise

    def _update_progress(self, task_id: str, progress: float) -> None:
        """Update task progress."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress = min(100.0, max(0.0, progress))

    def get_task(self, task_id: str) -> TrainingTask | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status as dictionary."""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def list_tasks(
        self,
        task_type: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TrainingTask]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())

        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove completed tasks older than max_age_hours."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0

        with self._lock:
            to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and task.completed_at
                and task.completed_at < cutoff
            ]

            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1

        if removed:
            logger.info("Cleaned up old tasks", removed=removed)

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get task manager statistics."""
        with self._lock:
            status_counts = {}
            for task in self._tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_tasks": len(self._tasks),
                "running_tasks": self._running_count,
                "max_workers": self.max_workers,
                "status_counts": status_counts,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager."""
        logger.info("Shutting down task manager", wait=wait)
        self._executor.shutdown(wait=wait)


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        settings = get_settings()
        _task_manager = BackgroundTaskManager(max_workers=settings.workers)
    return _task_manager


def reset_task_manager():
    """Reset the task manager (for testing)."""
    global _task_manager
    if _task_manager:
        _task_manager.shutdown(wait=False)
    _task_manager = None


# Async wrapper for FastAPI
async def submit_training_task(
    training_func: Callable,
    *args,
    **kwargs,
) -> TrainingTask:
    """Submit a training task asynchronously."""
    manager = get_task_manager()
    return manager.submit_task("model_training", training_func, *args, **kwargs)


async def get_training_status(task_id: str) -> dict[str, Any] | None:
    """Get training task status asynchronously."""
    manager = get_task_manager()
    return manager.get_task_status(task_id)
