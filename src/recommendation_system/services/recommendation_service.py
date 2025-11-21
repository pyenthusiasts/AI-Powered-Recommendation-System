"""Recommendation service orchestrating models and data."""

import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from recommendation_system.config import get_settings
from recommendation_system.models.hybrid import HybridRecommender
from recommendation_system.schemas import (
    InteractionType,
    ItemType,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationStrategy,
    RecommendedItem,
)
from recommendation_system.services.data_store import DataStore, get_data_store

logger = structlog.get_logger()


class RecommendationService:
    """Service for generating recommendations."""

    def __init__(self, data_store: DataStore | None = None):
        self._data_store = data_store or get_data_store()
        self._recommender = HybridRecommender()
        self._settings = get_settings()

    @property
    def recommender(self) -> HybridRecommender:
        return self._recommender

    @property
    def is_model_trained(self) -> bool:
        return self._recommender.is_trained

    def train_models(
        self,
        model_type: RecommendationStrategy = RecommendationStrategy.HYBRID,
        item_types: list[ItemType] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train recommendation models on current data.

        Args:
            model_type: Which model(s) to train
            item_types: Filter items by type (None = all)
            hyperparameters: Optional model hyperparameters

        Returns:
            Training results and metrics
        """
        hyperparameters = hyperparameters or {}
        results: dict[str, Any] = {"model_type": model_type.value}

        # Get items
        items = self._data_store.get_all_items()
        if item_types:
            items = {
                iid: item
                for iid, item in items.items()
                if item["item_type"] in item_types
            }

        if not items:
            raise ValueError("No items available for training")

        logger.info("Training models", model_type=model_type.value, num_items=len(items))

        # Train content-based model
        if model_type in (RecommendationStrategy.CONTENT_BASED, RecommendationStrategy.HYBRID):
            try:
                content_params = hyperparameters.get("content_based", {})
                content_result = self._recommender.train_content_model(items, **content_params)
                results["content_based"] = content_result
            except Exception as e:
                logger.error("Content-based training failed", error=str(e))
                results["content_based_error"] = str(e)

        # Train collaborative model
        if model_type in (RecommendationStrategy.COLLABORATIVE, RecommendationStrategy.HYBRID):
            try:
                matrix, user_ids, item_ids = self._data_store.build_interaction_matrix()
                if matrix.size > 0:
                    collab_result = self._recommender.train_collaborative_model(
                        matrix, user_ids, item_ids
                    )
                    results["collaborative"] = collab_result
                else:
                    results["collaborative_error"] = "No interactions available"
            except Exception as e:
                logger.error("Collaborative training failed", error=str(e))
                results["collaborative_error"] = str(e)

        return results

    def get_recommendations(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Get recommendations based on request parameters.

        Args:
            request: Recommendation request with parameters

        Returns:
            RecommendationResponse with recommendations
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        logger.info(
            "Processing recommendation request",
            request_id=request_id,
            user_id=request.user_id,
            item_id=request.item_id,
            strategy=request.strategy.value,
        )

        # Validate request
        if not request.user_id and not request.item_id:
            raise ValueError("Either user_id or item_id must be provided")

        # Get user's interacted items for exclusion
        exclude_items: set[str] = set()
        liked_items: list[str] = []

        if request.user_id and request.exclude_interacted:
            exclude_items = self._data_store.get_user_item_ids(request.user_id)

        if request.user_id:
            # Get positively interacted items for content-based
            positive_types = [
                InteractionType.LIKE,
                InteractionType.PURCHASE,
                InteractionType.BOOKMARK,
            ]
            interactions = self._data_store.get_user_interactions(
                request.user_id, positive_types
            )
            liked_items = [i.item_id for i in interactions]

            # Also include high ratings
            rating_interactions = self._data_store.get_user_interactions(
                request.user_id, [InteractionType.RATING]
            )
            liked_items.extend([
                i.item_id for i in rating_interactions
                if i.value and i.value >= 7
            ])

        # Map strategy
        strategy_map = {
            RecommendationStrategy.CONTENT_BASED: "content_based",
            RecommendationStrategy.COLLABORATIVE: "collaborative",
            RecommendationStrategy.HYBRID: "hybrid",
        }

        # Get recommendations
        if self._recommender.is_trained:
            raw_results = self._recommender.get_recommendations(
                user_id=request.user_id,
                item_id=request.item_id,
                liked_items=liked_items,
                n=request.num_recommendations,
                exclude_items=exclude_items,
                strategy=strategy_map[request.strategy],
            )
        else:
            # Fallback: return recent items
            logger.warning("Models not trained, using fallback")
            items = self._data_store.get_items(
                item_type=request.item_type,
                limit=request.num_recommendations,
            )
            raw_results = [
                (item.item_id, 0.5, "Popular item")
                for item in items
                if item.item_id not in exclude_items
            ]

        # Apply filters
        if request.item_type or request.filters:
            filtered_results = []
            for item_id, score, explanation in raw_results:
                item = self._data_store.get_item(item_id)
                if not item:
                    continue

                if request.item_type and item.item_type != request.item_type:
                    continue

                # Apply custom filters
                if request.filters:
                    match = True
                    for key, value in request.filters.items():
                        item_value = item.metadata.get(key) or item.features.get(key)
                        if item_value != value:
                            match = False
                            break
                    if not match:
                        continue

                filtered_results.append((item_id, score, explanation))

            raw_results = filtered_results[:request.num_recommendations]

        # Build response
        recommendations = []
        for item_id, score, explanation in raw_results:
            item = self._data_store.get_item(item_id)
            if item:
                recommendations.append(
                    RecommendedItem(
                        item_id=item.item_id,
                        title=item.title,
                        item_type=item.item_type,
                        score=min(max(score, 0), 1),  # Clamp to 0-1
                        explanation=explanation,
                        metadata=item.metadata,
                    )
                )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Recommendations generated",
            request_id=request_id,
            num_results=len(recommendations),
            processing_time_ms=f"{processing_time:.2f}",
        )

        return RecommendationResponse(
            recommendations=recommendations,
            strategy_used=request.strategy,
            total_candidates=len(raw_results),
            processing_time_ms=processing_time,
            request_id=request_id,
        )

    def save_models(self, directory: Path | None = None):
        """Save trained models to disk."""
        directory = directory or self._settings.model_path
        self._recommender.save(directory)

    def load_models(self, directory: Path | None = None):
        """Load models from disk."""
        directory = directory or self._settings.model_path
        if directory.exists():
            self._recommender.load(directory)
            logger.info("Models loaded from disk")
        else:
            logger.warning("Model directory not found", path=str(directory))


# Global service instance
_recommendation_service: RecommendationService | None = None


def get_recommendation_service() -> RecommendationService:
    """Get the global recommendation service instance."""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service


def reset_recommendation_service():
    """Reset the recommendation service (for testing)."""
    global _recommendation_service
    _recommendation_service = None
