"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ItemType(str, Enum):
    """Supported item types for recommendations."""

    MOVIE = "movie"
    BOOK = "book"
    SONG = "song"
    COURSE = "course"
    PRODUCT = "product"


class RecommendationStrategy(str, Enum):
    """Available recommendation strategies."""

    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


# ============== Item Schemas ==============


class ItemBase(BaseModel):
    """Base schema for items."""

    title: str = Field(..., min_length=1, max_length=500)
    item_type: ItemType
    description: str | None = Field(default=None, max_length=5000)
    features: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ItemCreate(ItemBase):
    """Schema for creating a new item."""

    item_id: str = Field(..., min_length=1, max_length=100)


class ItemUpdate(BaseModel):
    """Schema for updating an item."""

    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    features: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class Item(ItemBase):
    """Schema for item response."""

    item_id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============== User Schemas ==============


class UserBase(BaseModel):
    """Base schema for users."""

    preferences: dict[str, Any] = Field(default_factory=dict)


class UserCreate(UserBase):
    """Schema for creating a new user."""

    user_id: str = Field(..., min_length=1, max_length=100)


class User(UserBase):
    """Schema for user response."""

    user_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ============== Interaction Schemas ==============


class InteractionType(str, Enum):
    """Types of user-item interactions."""

    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    DISLIKE = "dislike"
    PURCHASE = "purchase"
    RATING = "rating"
    BOOKMARK = "bookmark"


class InteractionCreate(BaseModel):
    """Schema for creating an interaction."""

    user_id: str = Field(..., min_length=1, max_length=100)
    item_id: str = Field(..., min_length=1, max_length=100)
    interaction_type: InteractionType
    value: float | None = Field(default=None, ge=0, le=10)
    timestamp: datetime | None = None

    @field_validator("value")
    @classmethod
    def validate_rating_value(cls, v: float | None, info) -> float | None:
        if info.data.get("interaction_type") == InteractionType.RATING and v is None:
            raise ValueError("Rating interactions require a value")
        return v


class Interaction(InteractionCreate):
    """Schema for interaction response."""

    interaction_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ============== Recommendation Schemas ==============


class RecommendationRequest(BaseModel):
    """Schema for recommendation request."""

    user_id: str | None = Field(default=None, min_length=1, max_length=100)
    item_id: str | None = Field(default=None, min_length=1, max_length=100)
    item_type: ItemType | None = None
    strategy: RecommendationStrategy = Field(default=RecommendationStrategy.HYBRID)
    num_recommendations: int = Field(default=10, ge=1, le=100)
    exclude_interacted: bool = Field(default=True)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_id", "item_id")
    @classmethod
    def at_least_one_id(cls, v, info):
        # This will be validated at the API level
        return v


class RecommendedItem(BaseModel):
    """Schema for a single recommended item."""

    item_id: str
    title: str
    item_type: ItemType
    score: float = Field(..., ge=0, le=1)
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """Schema for recommendation response."""

    recommendations: list[RecommendedItem]
    strategy_used: RecommendationStrategy
    total_candidates: int
    processing_time_ms: float
    request_id: str


# ============== Training Schemas ==============


class TrainingRequest(BaseModel):
    """Schema for model training request."""

    model_type: RecommendationStrategy
    item_types: list[ItemType] | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainingStatus(BaseModel):
    """Schema for training status response."""

    job_id: str
    status: str
    progress: float = Field(ge=0, le=100)
    started_at: datetime
    completed_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# ============== Health & Stats Schemas ==============


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    uptime_seconds: float
    models_loaded: dict[str, bool]


class StatsResponse(BaseModel):
    """Schema for system statistics response."""

    total_users: int
    total_items: int
    total_interactions: int
    items_by_type: dict[str, int]
    interactions_by_type: dict[str, int]
    model_metrics: dict[str, Any]
