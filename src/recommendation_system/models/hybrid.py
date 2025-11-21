"""Hybrid Recommender combining Content-Based and Collaborative Filtering."""

from pathlib import Path
from typing import Any

import structlog

from recommendation_system.models.collaborative import CollaborativeFilteringModel
from recommendation_system.models.content_based import ContentBasedModel

logger = structlog.get_logger()


class HybridRecommender:
    """Hybrid recommendation system combining multiple strategies.

    This recommender intelligently combines:
    - Content-based filtering (for item similarity)
    - Collaborative filtering (for user preference patterns)

    The combination strategy adapts based on:
    - User history (cold start handling)
    - Model confidence
    - Configurable weights
    """

    def __init__(
        self,
        content_weight: float = 0.4,
        collaborative_weight: float = 0.6,
        cold_start_threshold: int = 5,
    ):
        """Initialize the hybrid recommender.

        Args:
            content_weight: Weight for content-based scores (0-1)
            collaborative_weight: Weight for collaborative scores (0-1)
            cold_start_threshold: Min interactions before using collaborative
        """
        if not 0 <= content_weight <= 1 or not 0 <= collaborative_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")

        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.cold_start_threshold = cold_start_threshold

        self._content_model = ContentBasedModel()
        self._collaborative_model = CollaborativeFilteringModel()

        self._is_content_trained = False
        self._is_collaborative_trained = False

    @property
    def content_model(self) -> ContentBasedModel:
        return self._content_model

    @property
    def collaborative_model(self) -> CollaborativeFilteringModel:
        return self._collaborative_model

    @property
    def is_trained(self) -> bool:
        return self._is_content_trained or self._is_collaborative_trained

    def train_content_model(
        self,
        items: dict[str, dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Train the content-based model."""
        result = self._content_model.train(items, **kwargs)
        self._is_content_trained = True
        return result

    def train_collaborative_model(
        self,
        interaction_matrix,
        user_ids: list[str],
        item_ids: list[str],
    ) -> dict[str, Any]:
        """Train the collaborative filtering model."""
        result = self._collaborative_model.train(interaction_matrix, user_ids, item_ids)
        self._is_collaborative_trained = True
        return result

    def get_recommendations(
        self,
        user_id: str | None = None,
        item_id: str | None = None,
        liked_items: list[str] | None = None,
        n: int = 10,
        exclude_items: set[str] | None = None,
        strategy: str = "hybrid",
    ) -> list[tuple[str, float, str]]:
        """Get recommendations using the specified strategy.

        Args:
            user_id: User ID for personalized recommendations
            item_id: Item ID for similar item recommendations
            liked_items: Items user has liked (for content-based)
            n: Number of recommendations
            exclude_items: Items to exclude
            strategy: 'hybrid', 'content_based', or 'collaborative'

        Returns:
            List of (item_id, score, explanation) tuples
        """
        exclude_items = exclude_items or set()

        # Item-based similarity (content-based)
        if item_id and not user_id:
            return self._get_similar_items(item_id, n, exclude_items)

        # User-based recommendations
        if strategy == "content_based":
            return self._get_content_recommendations(liked_items, n, exclude_items)
        elif strategy == "collaborative":
            return self._get_collaborative_recommendations(user_id, n, exclude_items)
        else:
            return self._get_hybrid_recommendations(
                user_id, liked_items, n, exclude_items
            )

    def _get_similar_items(
        self,
        item_id: str,
        n: int,
        exclude_items: set[str],
    ) -> list[tuple[str, float, str]]:
        """Get items similar to a given item."""
        results = []

        # Try content-based first
        if self._is_content_trained:
            content_results = self._content_model.get_similar_items(item_id, n * 2)
            for iid, score in content_results:
                if iid not in exclude_items:
                    results.append((iid, score, "Similar content features"))

        # Add collaborative if available
        if self._is_collaborative_trained:
            collab_results = self._collaborative_model.get_similar_items(item_id, n * 2)
            result_ids = {r[0] for r in results}

            for iid, score in collab_results:
                if iid not in exclude_items and iid not in result_ids:
                    results.append((iid, score * 0.8, "Users also liked"))

        # Sort by score and return top n
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def _get_content_recommendations(
        self,
        liked_items: list[str] | None,
        n: int,
        exclude_items: set[str],
    ) -> list[tuple[str, float, str]]:
        """Get content-based recommendations."""
        if not self._is_content_trained:
            logger.warning("Content model not trained")
            return []

        if not liked_items:
            return []

        results = self._content_model.get_recommendations_for_user(
            liked_items, n, exclude_items
        )

        return [(iid, score, "Based on your preferences") for iid, score in results]

    def _get_collaborative_recommendations(
        self,
        user_id: str | None,
        n: int,
        exclude_items: set[str],
    ) -> list[tuple[str, float, str]]:
        """Get collaborative filtering recommendations."""
        if not self._is_collaborative_trained:
            logger.warning("Collaborative model not trained")
            return []

        if not user_id:
            return []

        results = self._collaborative_model.get_recommendations_for_user(
            user_id, n, exclude_items
        )

        return [(iid, score, "People like you enjoyed") for iid, score in results]

    def _get_hybrid_recommendations(
        self,
        user_id: str | None,
        liked_items: list[str] | None,
        n: int,
        exclude_items: set[str],
    ) -> list[tuple[str, float, str]]:
        """Get hybrid recommendations combining both strategies."""
        scores: dict[str, dict[str, Any]] = {}

        # Determine weights based on available data
        content_weight = self.content_weight
        collab_weight = self.collaborative_weight

        # Adjust for cold start
        num_liked = len(liked_items) if liked_items else 0
        if num_liked < self.cold_start_threshold:
            # Favor content-based for cold start
            content_weight = 0.8
            collab_weight = 0.2

        # Get content-based scores
        if self._is_content_trained and liked_items:
            content_results = self._content_model.get_recommendations_for_user(
                liked_items, n * 3, exclude_items
            )
            for iid, score in content_results:
                scores[iid] = {
                    "content_score": score,
                    "collab_score": 0,
                    "explanation": "Based on your preferences",
                }

        # Get collaborative scores
        if self._is_collaborative_trained and user_id:
            collab_results = self._collaborative_model.get_recommendations_for_user(
                user_id, n * 3, exclude_items
            )
            for iid, score in collab_results:
                if iid in scores:
                    scores[iid]["collab_score"] = score
                    scores[iid]["explanation"] = "Recommended for you"
                else:
                    scores[iid] = {
                        "content_score": 0,
                        "collab_score": score,
                        "explanation": "People like you enjoyed",
                    }

        # Combine scores
        results = []
        for iid, data in scores.items():
            combined_score = (
                content_weight * data["content_score"]
                + collab_weight * data["collab_score"]
            )
            results.append((iid, combined_score, data["explanation"]))

        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def save(self, directory: Path):
        """Save both models to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._is_content_trained:
            self._content_model.save(directory / "content_model.pkl")

        if self._is_collaborative_trained:
            self._collaborative_model.save(directory / "collaborative_model.pkl")

        # Save config
        import json
        config = {
            "content_weight": self.content_weight,
            "collaborative_weight": self.collaborative_weight,
            "cold_start_threshold": self.cold_start_threshold,
            "is_content_trained": self._is_content_trained,
            "is_collaborative_trained": self._is_collaborative_trained,
        }
        with open(directory / "hybrid_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Hybrid recommender saved", directory=str(directory))

    def load(self, directory: Path):
        """Load models from a directory."""
        directory = Path(directory)

        # Load config
        import json
        config_path = directory / "hybrid_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.content_weight = config.get("content_weight", self.content_weight)
            self.collaborative_weight = config.get("collaborative_weight", self.collaborative_weight)
            self.cold_start_threshold = config.get("cold_start_threshold", self.cold_start_threshold)
            self._is_content_trained = config.get("is_content_trained", False)
            self._is_collaborative_trained = config.get("is_collaborative_trained", False)

        # Load models
        content_path = directory / "content_model.pkl"
        if content_path.exists():
            self._content_model.load(content_path)
            self._is_content_trained = True

        collab_path = directory / "collaborative_model.pkl"
        if collab_path.exists():
            self._collaborative_model.load(collab_path)
            self._is_collaborative_trained = True

        logger.info("Hybrid recommender loaded", directory=str(directory))

    def get_training_info(self) -> dict[str, Any]:
        """Get training information for both models."""
        return {
            "content_based": (
                self._content_model.get_training_info()
                if self._is_content_trained
                else None
            ),
            "collaborative": (
                self._collaborative_model.get_training_info()
                if self._is_collaborative_trained
                else None
            ),
            "weights": {
                "content": self.content_weight,
                "collaborative": self.collaborative_weight,
            },
        }
