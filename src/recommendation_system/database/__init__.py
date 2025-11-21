"""Database package for persistent storage."""

from recommendation_system.database.connection import (
    DatabaseManager,
    get_db,
    get_db_manager,
    init_database,
)
from recommendation_system.database.models import Base, InteractionModel, ItemModel, UserModel
from recommendation_system.database.repository import (
    InteractionRepository,
    ItemRepository,
    UserRepository,
)

__all__ = [
    "Base",
    "DatabaseManager",
    "InteractionModel",
    "InteractionRepository",
    "ItemModel",
    "ItemRepository",
    "UserModel",
    "UserRepository",
    "get_db",
    "get_db_manager",
    "init_database",
]
