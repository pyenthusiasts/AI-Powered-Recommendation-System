"""Repository pattern implementation for database operations."""

from datetime import datetime
from typing import Any

import numpy as np
import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from recommendation_system.database.models import (
    InteractionModel,
    ItemModel,
    ModelArtifactModel,
    RecommendationLogModel,
    UserModel,
)
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


class ItemRepository:
    """Repository for item operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, item: ItemCreate) -> Item:
        """Create a new item."""
        db_item = ItemModel(
            item_id=item.item_id,
            title=item.title,
            item_type=item.item_type,
            description=item.description,
            features=item.features,
            metadata_=item.metadata,
        )
        self.session.add(db_item)
        self.session.flush()

        logger.info("Item created", item_id=item.item_id, item_type=item.item_type.value)
        return self._to_schema(db_item)

    def get(self, item_id: str) -> Item | None:
        """Get an item by ID."""
        db_item = self.session.query(ItemModel).filter(
            ItemModel.item_id == item_id,
            ItemModel.is_active == True,
        ).first()
        return self._to_schema(db_item) if db_item else None

    def get_by_ids(self, item_ids: list[str]) -> list[Item]:
        """Get multiple items by IDs."""
        db_items = self.session.query(ItemModel).filter(
            ItemModel.item_id.in_(item_ids),
            ItemModel.is_active == True,
        ).all()
        return [self._to_schema(item) for item in db_items]

    def update(self, item_id: str, update: ItemUpdate) -> Item | None:
        """Update an existing item."""
        db_item = self.session.query(ItemModel).filter(
            ItemModel.item_id == item_id,
            ItemModel.is_active == True,
        ).first()

        if not db_item:
            return None

        update_data = update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                if field == "metadata":
                    setattr(db_item, "metadata_", value)
                else:
                    setattr(db_item, field, value)

        db_item.updated_at = datetime.utcnow()
        self.session.flush()

        return self._to_schema(db_item)

    def delete(self, item_id: str, hard_delete: bool = False) -> bool:
        """Delete an item (soft delete by default)."""
        db_item = self.session.query(ItemModel).filter(
            ItemModel.item_id == item_id
        ).first()

        if not db_item:
            return False

        if hard_delete:
            self.session.delete(db_item)
        else:
            db_item.is_active = False
            db_item.updated_at = datetime.utcnow()

        self.session.flush()
        logger.info("Item deleted", item_id=item_id, hard_delete=hard_delete)
        return True

    def list(
        self,
        item_type: ItemType | None = None,
        limit: int = 100,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> list[Item]:
        """List items with optional filtering."""
        query = self.session.query(ItemModel)

        if not include_inactive:
            query = query.filter(ItemModel.is_active == True)

        if item_type:
            query = query.filter(ItemModel.item_type == item_type)

        query = query.order_by(ItemModel.created_at.desc())
        db_items = query.offset(offset).limit(limit).all()

        return [self._to_schema(item) for item in db_items]

    def get_all_for_training(self) -> dict[str, dict[str, Any]]:
        """Get all active items formatted for model training."""
        db_items = self.session.query(ItemModel).filter(
            ItemModel.is_active == True
        ).all()

        return {
            item.item_id: {
                "item_id": item.item_id,
                "title": item.title,
                "item_type": item.item_type,
                "description": item.description,
                "features": item.features or {},
                "metadata": item.metadata_ or {},
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in db_items
        }

    def count(self, item_type: ItemType | None = None) -> int:
        """Count items."""
        query = self.session.query(func.count(ItemModel.id)).filter(
            ItemModel.is_active == True
        )
        if item_type:
            query = query.filter(ItemModel.item_type == item_type)
        return query.scalar() or 0

    def exists(self, item_id: str) -> bool:
        """Check if item exists."""
        return self.session.query(
            self.session.query(ItemModel).filter(
                ItemModel.item_id == item_id
            ).exists()
        ).scalar()

    def _to_schema(self, db_item: ItemModel) -> Item:
        """Convert database model to Pydantic schema."""
        return Item(
            item_id=db_item.item_id,
            title=db_item.title,
            item_type=db_item.item_type,
            description=db_item.description,
            features=db_item.features or {},
            metadata=db_item.metadata_ or {},
            created_at=db_item.created_at,
            updated_at=db_item.updated_at,
        )


class UserRepository:
    """Repository for user operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, user: UserCreate) -> User:
        """Create a new user."""
        db_user = UserModel(
            user_id=user.user_id,
            preferences=user.preferences,
        )
        self.session.add(db_user)
        self.session.flush()

        logger.info("User created", user_id=user.user_id)
        return self._to_schema(db_user)

    def get(self, user_id: str) -> User | None:
        """Get a user by ID."""
        db_user = self.session.query(UserModel).filter(
            UserModel.user_id == user_id,
            UserModel.is_active == True,
        ).first()
        return self._to_schema(db_user) if db_user else None

    def get_or_create(self, user_id: str) -> User:
        """Get existing user or create new one."""
        user = self.get(user_id)
        if user:
            return user
        return self.create(UserCreate(user_id=user_id))

    def update_last_active(self, user_id: str) -> None:
        """Update user's last active timestamp."""
        self.session.query(UserModel).filter(
            UserModel.user_id == user_id
        ).update({"last_active_at": datetime.utcnow()})
        self.session.flush()

    def get_all_for_training(self) -> dict[str, dict[str, Any]]:
        """Get all active users formatted for model training."""
        db_users = self.session.query(UserModel).filter(
            UserModel.is_active == True
        ).all()

        return {
            user.user_id: {
                "user_id": user.user_id,
                "preferences": user.preferences or {},
                "created_at": user.created_at,
            }
            for user in db_users
        }

    def count(self) -> int:
        """Count active users."""
        return self.session.query(func.count(UserModel.id)).filter(
            UserModel.is_active == True
        ).scalar() or 0

    def exists(self, user_id: str) -> bool:
        """Check if user exists."""
        return self.session.query(
            self.session.query(UserModel).filter(
                UserModel.user_id == user_id
            ).exists()
        ).scalar()

    def _to_schema(self, db_user: UserModel) -> User:
        """Convert database model to Pydantic schema."""
        return User(
            user_id=db_user.user_id,
            preferences=db_user.preferences or {},
            created_at=db_user.created_at,
        )


class InteractionRepository:
    """Repository for interaction operations."""

    # Weights for different interaction types in collaborative filtering
    INTERACTION_WEIGHTS = {
        InteractionType.VIEW: 0.1,
        InteractionType.CLICK: 0.2,
        InteractionType.LIKE: 0.7,
        InteractionType.DISLIKE: -0.5,
        InteractionType.PURCHASE: 1.0,
        InteractionType.RATING: 1.0,
        InteractionType.BOOKMARK: 0.5,
    }

    def __init__(self, session: Session):
        self.session = session

    def create(self, interaction: InteractionCreate) -> Interaction:
        """Create a new interaction."""
        db_interaction = InteractionModel(
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type,
            value=interaction.value,
            timestamp=interaction.timestamp or datetime.utcnow(),
        )
        self.session.add(db_interaction)
        self.session.flush()

        logger.debug(
            "Interaction created",
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type.value,
        )
        return self._to_schema(db_interaction)

    def get_by_user(
        self,
        user_id: str,
        interaction_types: list[InteractionType] | None = None,
        limit: int | None = None,
    ) -> list[Interaction]:
        """Get interactions for a user."""
        query = self.session.query(InteractionModel).filter(
            InteractionModel.user_id == user_id
        )

        if interaction_types:
            query = query.filter(InteractionModel.interaction_type.in_(interaction_types))

        query = query.order_by(InteractionModel.timestamp.desc())

        if limit:
            query = query.limit(limit)

        return [self._to_schema(i) for i in query.all()]

    def get_by_item(self, item_id: str) -> list[Interaction]:
        """Get interactions for an item."""
        db_interactions = self.session.query(InteractionModel).filter(
            InteractionModel.item_id == item_id
        ).order_by(InteractionModel.timestamp.desc()).all()

        return [self._to_schema(i) for i in db_interactions]

    def get_user_item_ids(self, user_id: str) -> set[str]:
        """Get set of item IDs a user has interacted with."""
        result = self.session.query(InteractionModel.item_id).filter(
            InteractionModel.user_id == user_id
        ).distinct().all()

        return {r[0] for r in result}

    def get_all_for_training(self) -> list[dict[str, Any]]:
        """Get all interactions formatted for model training."""
        db_interactions = self.session.query(InteractionModel).all()

        return [
            {
                "interaction_id": i.interaction_id,
                "user_id": i.user_id,
                "item_id": i.item_id,
                "interaction_type": i.interaction_type,
                "value": i.value,
                "timestamp": i.timestamp,
            }
            for i in db_interactions
        ]

    def build_interaction_matrix(
        self,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build user-item interaction matrix for collaborative filtering.

        Returns:
            Tuple of (matrix, user_ids, item_ids)
        """
        # Get unique users and items with interactions
        user_query = self.session.query(InteractionModel.user_id).distinct()
        item_query = self.session.query(InteractionModel.item_id).distinct()

        user_ids = [r[0] for r in user_query.all()]
        item_ids = [r[0] for r in item_query.all()]

        if not user_ids or not item_ids:
            return np.array([]), [], []

        user_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_idx = {iid: i for i, iid in enumerate(item_ids)}

        matrix = np.zeros((len(user_ids), len(item_ids)))

        # Fetch all interactions
        interactions = self.session.query(InteractionModel).all()

        for interaction in interactions:
            uid = interaction.user_id
            iid = interaction.item_id
            itype = interaction.interaction_type

            if uid in user_idx and iid in item_idx:
                ui, ii = user_idx[uid], item_idx[iid]

                if itype == InteractionType.RATING and interaction.value is not None:
                    matrix[ui, ii] = interaction.value / 10.0
                else:
                    weight = self.INTERACTION_WEIGHTS.get(itype, 0.1)
                    matrix[ui, ii] = max(matrix[ui, ii], weight)

        return matrix, user_ids, item_ids

    def count(self, user_id: str | None = None, item_id: str | None = None) -> int:
        """Count interactions with optional filtering."""
        query = self.session.query(func.count(InteractionModel.id))

        if user_id:
            query = query.filter(InteractionModel.user_id == user_id)
        if item_id:
            query = query.filter(InteractionModel.item_id == item_id)

        return query.scalar() or 0

    def get_stats(self) -> dict[str, Any]:
        """Get interaction statistics."""
        total = self.count()

        # Count by type
        type_counts = self.session.query(
            InteractionModel.interaction_type,
            func.count(InteractionModel.id)
        ).group_by(InteractionModel.interaction_type).all()

        by_type = {t.value: c for t, c in type_counts}

        return {
            "total": total,
            "by_type": by_type,
        }

    def _to_schema(self, db_interaction: InteractionModel) -> Interaction:
        """Convert database model to Pydantic schema."""
        return Interaction(
            interaction_id=db_interaction.interaction_id,
            user_id=db_interaction.user_id,
            item_id=db_interaction.item_id,
            interaction_type=db_interaction.interaction_type,
            value=db_interaction.value,
            timestamp=db_interaction.timestamp,
            created_at=db_interaction.created_at,
        )


class ModelArtifactRepository:
    """Repository for model artifact operations."""

    def __init__(self, session: Session):
        self.session = session

    def save_artifact(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        artifact_path: str,
        hyperparameters: dict | None = None,
        metrics: dict | None = None,
        item_count: int = 0,
        user_count: int = 0,
        interaction_count: int = 0,
    ) -> ModelArtifactModel:
        """Save model artifact metadata."""
        artifact = ModelArtifactModel(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            artifact_path=artifact_path,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            item_count=item_count,
            user_count=user_count,
            interaction_count=interaction_count,
        )
        self.session.add(artifact)
        self.session.flush()
        return artifact

    def get_active_model(self, model_name: str) -> ModelArtifactModel | None:
        """Get the currently active model artifact."""
        return self.session.query(ModelArtifactModel).filter(
            ModelArtifactModel.model_name == model_name,
            ModelArtifactModel.is_active == True,
        ).first()

    def set_active_model(self, model_name: str, model_version: str) -> bool:
        """Set a specific model version as active."""
        # Deactivate current active model
        self.session.query(ModelArtifactModel).filter(
            ModelArtifactModel.model_name == model_name,
            ModelArtifactModel.is_active == True,
        ).update({"is_active": False})

        # Activate the specified version
        result = self.session.query(ModelArtifactModel).filter(
            ModelArtifactModel.model_name == model_name,
            ModelArtifactModel.model_version == model_version,
        ).update({"is_active": True})

        self.session.flush()
        return result > 0


class RecommendationLogRepository:
    """Repository for recommendation logging."""

    def __init__(self, session: Session):
        self.session = session

    def log_recommendation(
        self,
        request_id: str,
        user_id: str | None,
        item_id: str | None,
        strategy: str,
        num_requested: int,
        num_returned: int,
        recommended_items: list[str],
        processing_time_ms: float,
        model_version: str | None = None,
        experiment_id: str | None = None,
    ) -> RecommendationLogModel:
        """Log a recommendation request."""
        log = RecommendationLogModel(
            request_id=request_id,
            user_id=user_id,
            item_id=item_id,
            strategy=strategy,
            num_requested=num_requested,
            num_returned=num_returned,
            recommended_items=recommended_items,
            processing_time_ms=processing_time_ms,
            model_version=model_version,
            experiment_id=experiment_id,
        )
        self.session.add(log)
        self.session.flush()
        return log

    def get_user_recommendation_history(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[RecommendationLogModel]:
        """Get recommendation history for a user."""
        return self.session.query(RecommendationLogModel).filter(
            RecommendationLogModel.user_id == user_id
        ).order_by(
            RecommendationLogModel.created_at.desc()
        ).limit(limit).all()
