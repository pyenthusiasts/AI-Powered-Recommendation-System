"""Collaborative Filtering Model using Matrix Factorization."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


class CollaborativeFilteringModel:
    """Collaborative filtering using matrix factorization (SVD).

    This model learns latent factors for users and items from
    the user-item interaction matrix. It can provide:
    - User-based recommendations (find similar users)
    - Item-based recommendations (find similar items)
    - Personalized recommendations for a user
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        regularization: float = 0.01,
        learning_rate: float = 0.01,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.learning_rate = learning_rate

        self._user_ids: list[str] = []
        self._item_ids: list[str] = []
        self._user_id_to_idx: dict[str, int] = {}
        self._item_id_to_idx: dict[str, int] = {}

        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._user_similarity: np.ndarray | None = None
        self._item_similarity: np.ndarray | None = None
        self._global_mean: float = 0.0

        self._is_trained = False
        self._training_info: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        interaction_matrix: np.ndarray,
        user_ids: list[str],
        item_ids: list[str],
    ) -> dict[str, Any]:
        """Train the collaborative filtering model.

        Args:
            interaction_matrix: User-item interaction matrix (users x items)
            user_ids: List of user IDs (rows)
            item_ids: List of item IDs (columns)

        Returns:
            Training metrics
        """
        if interaction_matrix.size == 0:
            raise ValueError("Empty interaction matrix")

        self._user_ids = user_ids
        self._item_ids = item_ids
        self._user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self._item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

        logger.info(
            "Training collaborative filtering model",
            num_users=len(user_ids),
            num_items=len(item_ids),
            n_factors=self.n_factors,
        )

        # Compute global mean for bias
        mask = interaction_matrix > 0
        if mask.sum() > 0:
            self._global_mean = interaction_matrix[mask].mean()
        else:
            self._global_mean = 0.0

        # Use SVD for matrix factorization
        # Limit factors to matrix dimensions
        n_factors = min(self.n_factors, min(interaction_matrix.shape) - 1, 100)
        if n_factors < 1:
            n_factors = 1

        svd = TruncatedSVD(n_components=n_factors, n_iter=self.n_iterations, random_state=42)

        # Fit SVD to get item factors
        sparse_matrix = csr_matrix(interaction_matrix)
        self._user_factors = svd.fit_transform(sparse_matrix)

        # Item factors are the components
        self._item_factors = svd.components_.T

        # Compute similarity matrices
        self._user_similarity = cosine_similarity(self._user_factors)
        self._item_similarity = cosine_similarity(self._item_factors)

        self._is_trained = True

        # Compute training metrics
        reconstructed = np.dot(self._user_factors, self._item_factors.T)
        mse = np.mean((interaction_matrix[mask] - reconstructed[mask]) ** 2) if mask.sum() > 0 else 0

        self._training_info = {
            "num_users": len(user_ids),
            "num_items": len(item_ids),
            "n_factors": n_factors,
            "explained_variance_ratio": float(sum(svd.explained_variance_ratio_)),
            "reconstruction_mse": float(mse),
            "sparsity": float(1 - mask.sum() / mask.size),
        }

        logger.info(
            "Collaborative filtering model trained",
            explained_variance=f"{self._training_info['explained_variance_ratio']:.2%}",
            mse=f"{mse:.4f}",
        )

        return self._training_info

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict the rating a user would give to an item.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating/score
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if user_id not in self._user_id_to_idx:
            return self._global_mean

        if item_id not in self._item_id_to_idx:
            return self._global_mean

        user_idx = self._user_id_to_idx[user_id]
        item_idx = self._item_id_to_idx[item_id]

        score = np.dot(self._user_factors[user_idx], self._item_factors[item_idx])
        return float(np.clip(score, 0, 1))

    def get_recommendations_for_user(
        self,
        user_id: str,
        n: int = 10,
        exclude_items: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Get personalized recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            List of (item_id, predicted_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        exclude_items = exclude_items or set()

        if user_id not in self._user_id_to_idx:
            # Cold start: return popular items
            logger.debug("Cold start user", user_id=user_id)
            return self._get_popular_items(n, exclude_items)

        user_idx = self._user_id_to_idx[user_id]

        # Compute predicted scores for all items
        scores = np.dot(self._user_factors[user_idx], self._item_factors.T)

        # Exclude items
        for item_id in exclude_items:
            if item_id in self._item_id_to_idx:
                scores[self._item_id_to_idx[item_id]] = -np.inf

        # Get top-n items
        top_indices = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                # Normalize score to 0-1
                normalized_score = float(np.clip(scores[idx], 0, 1))
                results.append((self._item_ids[idx], normalized_score))

        return results

    def get_similar_users(
        self,
        user_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """Find users similar to a given user.

        Args:
            user_id: Source user ID
            n: Number of similar users

        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if user_id not in self._user_id_to_idx:
            return []

        user_idx = self._user_id_to_idx[user_id]
        similarities = self._user_similarity[user_idx].copy()
        similarities[user_idx] = -1  # Exclude self

        top_indices = np.argsort(similarities)[::-1][:n]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self._user_ids[idx], float(similarities[idx])))

        return results

    def get_similar_items(
        self,
        item_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """Find items similar to a given item (based on user interactions).

        Args:
            item_id: Source item ID
            n: Number of similar items

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if item_id not in self._item_id_to_idx:
            return []

        item_idx = self._item_id_to_idx[item_id]
        similarities = self._item_similarity[item_idx].copy()
        similarities[item_idx] = -1  # Exclude self

        top_indices = np.argsort(similarities)[::-1][:n]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self._item_ids[idx], float(similarities[idx])))

        return results

    def _get_popular_items(
        self,
        n: int,
        exclude_items: set[str],
    ) -> list[tuple[str, float]]:
        """Get popular items for cold start users."""
        # Use item factor norms as a proxy for popularity
        item_scores = np.linalg.norm(self._item_factors, axis=1)

        for item_id in exclude_items:
            if item_id in self._item_id_to_idx:
                item_scores[self._item_id_to_idx[item_id]] = -np.inf

        top_indices = np.argsort(item_scores)[::-1][:n]

        results = []
        max_score = item_scores[top_indices[0]] if len(top_indices) > 0 else 1
        for idx in top_indices:
            if item_scores[idx] > -np.inf:
                # Normalize
                score = float(item_scores[idx] / max_score) if max_score > 0 else 0
                results.append((self._item_ids[idx], score))

        return results

    def save(self, path: Path):
        """Save model to disk."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "user_ids": self._user_ids,
            "item_ids": self._item_ids,
            "user_id_to_idx": self._user_id_to_idx,
            "item_id_to_idx": self._item_id_to_idx,
            "user_factors": self._user_factors,
            "item_factors": self._item_factors,
            "user_similarity": self._user_similarity,
            "item_similarity": self._item_similarity,
            "global_mean": self._global_mean,
            "training_info": self._training_info,
            "config": {
                "n_factors": self.n_factors,
                "n_iterations": self.n_iterations,
                "regularization": self.regularization,
                "learning_rate": self.learning_rate,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Collaborative filtering model saved", path=str(path))

    def load(self, path: Path):
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self._user_ids = model_data["user_ids"]
        self._item_ids = model_data["item_ids"]
        self._user_id_to_idx = model_data["user_id_to_idx"]
        self._item_id_to_idx = model_data["item_id_to_idx"]
        self._user_factors = model_data["user_factors"]
        self._item_factors = model_data["item_factors"]
        self._user_similarity = model_data["user_similarity"]
        self._item_similarity = model_data["item_similarity"]
        self._global_mean = model_data["global_mean"]
        self._training_info = model_data["training_info"]

        config = model_data.get("config", {})
        self.n_factors = config.get("n_factors", self.n_factors)
        self.n_iterations = config.get("n_iterations", self.n_iterations)

        self._is_trained = True
        logger.info("Collaborative filtering model loaded", path=str(path))

    def get_training_info(self) -> dict[str, Any]:
        """Get training information."""
        return self._training_info.copy()
