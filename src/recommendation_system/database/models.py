"""SQLAlchemy database models for persistent storage."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from recommendation_system.schemas import InteractionType, ItemType

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class ItemModel(Base):
    """Database model for items (movies, books, songs, courses, products)."""

    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    item_type = Column(Enum(ItemType), nullable=False, index=True)
    description = Column(Text, nullable=True)
    features = Column(JSON, default=dict)
    metadata_ = Column("metadata", JSON, default=dict)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    interactions = relationship("InteractionModel", back_populates="item", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_items_type_active", "item_type", "is_active"),
        Index("idx_items_created_at", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "item_type": self.item_type,
            "description": self.description,
            "features": self.features or {},
            "metadata": self.metadata_ or {},
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class UserModel(Base):
    """Database model for users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    preferences = Column(JSON, default=dict)
    profile_data = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, index=True)
    last_active_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    interactions = relationship("InteractionModel", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_users_active", "is_active"),
        Index("idx_users_last_active", "last_active_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences or {},
            "profile_data": self.profile_data or {},
            "is_active": self.is_active,
            "last_active_at": self.last_active_at,
            "created_at": self.created_at,
        }


class InteractionModel(Base):
    """Database model for user-item interactions."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    interaction_id = Column(String(36), unique=True, nullable=False, default=generate_uuid, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    item_id = Column(String(100), ForeignKey("items.item_id", ondelete="CASCADE"), nullable=False)
    interaction_type = Column(Enum(InteractionType), nullable=False, index=True)
    value = Column(Float, nullable=True)
    context = Column(JSON, default=dict)  # Session info, device, location, etc.
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("UserModel", back_populates="interactions")
    item = relationship("ItemModel", back_populates="interactions")

    __table_args__ = (
        Index("idx_interactions_user_item", "user_id", "item_id"),
        Index("idx_interactions_user_type", "user_id", "interaction_type"),
        Index("idx_interactions_item_type", "item_id", "interaction_type"),
        Index("idx_interactions_timestamp", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "interaction_type": self.interaction_type,
            "value": self.value,
            "context": self.context or {},
            "timestamp": self.timestamp,
            "created_at": self.created_at,
        }


class ModelArtifactModel(Base):
    """Database model for storing trained model metadata."""

    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    artifact_path = Column(String(500), nullable=False)
    hyperparameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    item_count = Column(Integer, default=0)
    user_count = Column(Integer, default=0)
    interaction_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=False, index=True)
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_version"),
        Index("idx_model_artifacts_active", "model_name", "is_active"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "artifact_path": self.artifact_path,
            "hyperparameters": self.hyperparameters or {},
            "metrics": self.metrics or {},
            "item_count": self.item_count,
            "user_count": self.user_count,
            "interaction_count": self.interaction_count,
            "is_active": self.is_active,
            "trained_at": self.trained_at,
            "created_at": self.created_at,
        }


class RecommendationLogModel(Base):
    """Database model for logging recommendations (for A/B testing and analytics)."""

    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    item_id = Column(String(100), nullable=True)
    strategy = Column(String(50), nullable=False)
    num_requested = Column(Integer, nullable=False)
    num_returned = Column(Integer, nullable=False)
    recommended_items = Column(JSON, default=list)
    processing_time_ms = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    experiment_id = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_rec_logs_user_created", "user_id", "created_at"),
        Index("idx_rec_logs_experiment", "experiment_id", "created_at"),
    )
