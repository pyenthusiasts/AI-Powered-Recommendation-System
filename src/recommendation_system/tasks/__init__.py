"""Background task processing for async operations."""

from recommendation_system.tasks.training import (
    BackgroundTaskManager,
    TrainingTask,
    TaskStatus,
    get_task_manager,
)

__all__ = [
    "BackgroundTaskManager",
    "TaskStatus",
    "TrainingTask",
    "get_task_manager",
]
