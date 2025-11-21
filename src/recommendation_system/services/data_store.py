"""In-memory data store for items, users, and interactions."""

import uuid
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from recommendation_system.schemas import (
    Interaction,
    InteractionCreate,
    InteractionType,
    Item,
    ItemCreate,
    ItemType,
    ItemUpdate,
    User,
    UserCreate,
)

logger = structlog.get_logger()


class DataStore:
    """Thread-safe in-memory data store for the recommendation system."""

    def __init__(self):
        self._items: dict[str, dict[str, Any]] = {}
        self._users: dict[str, dict[str, Any]] = {}
        self._interactions: list[dict[str, Any]] = []
        self._user_interactions: dict[str, list[int]] = {}  # user_id -> interaction indices
        self._item_interactions: dict[str, list[int]] = {}  # item_id -> interaction indices

    # ============== Item Operations ==============

    def add_item(self, item: ItemCreate) -> Item:
        """Add a new item to the store."""
        if item.item_id in self._items:
            raise ValueError(f"Item {item.item_id} already exists")

        now = datetime.utcnow()
        item_data = {
            "item_id": item.item_id,
            "title": item.title,
            "item_type": item.item_type,
            "description": item.description,
            "features": item.features,
            "metadata": item.metadata,
            "created_at": now,
            "updated_at": now,
        }
        self._items[item.item_id] = item_data
        self._item_interactions[item.item_id] = []

        logger.info("Item added", item_id=item.item_id, item_type=item.item_type.value)
        return Item(**item_data)

    def get_item(self, item_id: str) -> Item | None:
        """Get an item by ID."""
        item_data = self._items.get(item_id)
        return Item(**item_data) if item_data else None

    def update_item(self, item_id: str, update: ItemUpdate) -> Item | None:
        """Update an existing item."""
        if item_id not in self._items:
            return None

        item_data = self._items[item_id]
        update_dict = update.model_dump(exclude_unset=True)

        for key, value in update_dict.items():
            if value is not None:
                item_data[key] = value

        item_data["updated_at"] = datetime.utcnow()
        return Item(**item_data)

    def delete_item(self, item_id: str) -> bool:
        """Delete an item."""
        if item_id not in self._items:
            return False

        del self._items[item_id]
        del self._item_interactions[item_id]
        logger.info("Item deleted", item_id=item_id)
        return True

    def get_items(
        self,
        item_type: ItemType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Item]:
        """Get items with optional filtering."""
        items = list(self._items.values())

        if item_type:
            items = [i for i in items if i["item_type"] == item_type]

        items = sorted(items, key=lambda x: x["created_at"], reverse=True)
        return [Item(**i) for i in items[offset : offset + limit]]

    def get_all_items(self) -> dict[str, dict[str, Any]]:
        """Get all items as raw dict for model training."""
        return self._items.copy()

    def get_items_by_ids(self, item_ids: list[str]) -> list[Item]:
        """Get multiple items by their IDs."""
        return [Item(**self._items[iid]) for iid in item_ids if iid in self._items]

    # ============== User Operations ==============

    def add_user(self, user: UserCreate) -> User:
        """Add a new user."""
        if user.user_id in self._users:
            raise ValueError(f"User {user.user_id} already exists")

        now = datetime.utcnow()
        user_data = {
            "user_id": user.user_id,
            "preferences": user.preferences,
            "created_at": now,
        }
        self._users[user.user_id] = user_data
        self._user_interactions[user.user_id] = []

        logger.info("User added", user_id=user.user_id)
        return User(**user_data)

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        user_data = self._users.get(user_id)
        return User(**user_data) if user_data else None

    def get_or_create_user(self, user_id: str) -> User:
        """Get existing user or create new one."""
        user = self.get_user(user_id)
        if user:
            return user
        return self.add_user(UserCreate(user_id=user_id))

    def get_all_users(self) -> dict[str, dict[str, Any]]:
        """Get all users."""
        return self._users.copy()

    # ============== Interaction Operations ==============

    def add_interaction(self, interaction: InteractionCreate) -> Interaction:
        """Add a user-item interaction."""
        # Ensure user exists
        self.get_or_create_user(interaction.user_id)

        if interaction.item_id not in self._items:
            raise ValueError(f"Item {interaction.item_id} does not exist")

        interaction_id = str(uuid.uuid4())
        now = datetime.utcnow()

        interaction_data = {
            "interaction_id": interaction_id,
            "user_id": interaction.user_id,
            "item_id": interaction.item_id,
            "interaction_type": interaction.interaction_type,
            "value": interaction.value,
            "timestamp": interaction.timestamp or now,
            "created_at": now,
        }

        idx = len(self._interactions)
        self._interactions.append(interaction_data)

        if interaction.user_id not in self._user_interactions:
            self._user_interactions[interaction.user_id] = []
        self._user_interactions[interaction.user_id].append(idx)

        if interaction.item_id not in self._item_interactions:
            self._item_interactions[interaction.item_id] = []
        self._item_interactions[interaction.item_id].append(idx)

        logger.debug(
            "Interaction added",
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type.value,
        )
        return Interaction(**interaction_data)

    def get_user_interactions(
        self,
        user_id: str,
        interaction_types: list[InteractionType] | None = None,
    ) -> list[Interaction]:
        """Get all interactions for a user."""
        indices = self._user_interactions.get(user_id, [])
        interactions = [self._interactions[i] for i in indices]

        if interaction_types:
            interactions = [
                i for i in interactions if i["interaction_type"] in interaction_types
            ]

        return [Interaction(**i) for i in interactions]

    def get_item_interactions(self, item_id: str) -> list[Interaction]:
        """Get all interactions for an item."""
        indices = self._item_interactions.get(item_id, [])
        return [Interaction(**self._interactions[i]) for i in indices]

    def get_all_interactions(self) -> list[dict[str, Any]]:
        """Get all interactions for model training."""
        return self._interactions.copy()

    def get_user_item_ids(self, user_id: str) -> set[str]:
        """Get set of item IDs a user has interacted with."""
        indices = self._user_interactions.get(user_id, [])
        return {self._interactions[i]["item_id"] for i in indices}

    def build_interaction_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Build user-item interaction matrix for collaborative filtering.

        Returns:
            Tuple of (matrix, user_ids, item_ids)
        """
        user_ids = list(self._users.keys())
        item_ids = list(self._items.keys())

        if not user_ids or not item_ids:
            return np.array([]), [], []

        user_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_idx = {iid: i for i, iid in enumerate(item_ids)}

        matrix = np.zeros((len(user_ids), len(item_ids)))

        # Weight different interaction types
        interaction_weights = {
            InteractionType.VIEW: 0.1,
            InteractionType.CLICK: 0.2,
            InteractionType.LIKE: 0.7,
            InteractionType.DISLIKE: -0.5,
            InteractionType.PURCHASE: 1.0,
            InteractionType.RATING: 1.0,  # Will use actual value
            InteractionType.BOOKMARK: 0.5,
        }

        for interaction in self._interactions:
            uid = interaction["user_id"]
            iid = interaction["item_id"]
            itype = interaction["interaction_type"]

            if uid in user_idx and iid in item_idx:
                ui, ii = user_idx[uid], item_idx[iid]

                if itype == InteractionType.RATING and interaction["value"] is not None:
                    # Normalize rating to 0-1 scale
                    matrix[ui, ii] = interaction["value"] / 10.0
                else:
                    weight = interaction_weights.get(itype, 0.1)
                    matrix[ui, ii] = max(matrix[ui, ii], weight)

        return matrix, user_ids, item_ids

    # ============== Statistics ==============

    def get_stats(self) -> dict[str, Any]:
        """Get data store statistics."""
        items_by_type: dict[str, int] = {}
        for item in self._items.values():
            itype = item["item_type"].value
            items_by_type[itype] = items_by_type.get(itype, 0) + 1

        interactions_by_type: dict[str, int] = {}
        for interaction in self._interactions:
            itype = interaction["interaction_type"].value
            interactions_by_type[itype] = interactions_by_type.get(itype, 0) + 1

        return {
            "total_users": len(self._users),
            "total_items": len(self._items),
            "total_interactions": len(self._interactions),
            "items_by_type": items_by_type,
            "interactions_by_type": interactions_by_type,
        }

    def clear(self):
        """Clear all data (for testing)."""
        self._items.clear()
        self._users.clear()
        self._interactions.clear()
        self._user_interactions.clear()
        self._item_interactions.clear()


# Global singleton instance
_data_store: DataStore | None = None


def get_data_store() -> DataStore:
    """Get the global data store instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store


def reset_data_store():
    """Reset the data store (for testing)."""
    global _data_store
    _data_store = DataStore()
