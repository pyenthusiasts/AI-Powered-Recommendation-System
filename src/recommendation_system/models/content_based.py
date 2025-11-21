"""Content-Based Filtering Model using TF-IDF and cosine similarity."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logger = structlog.get_logger()


class ContentBasedModel:
    """Content-based recommendation model using item features.

    This model computes item similarities based on:
    - Text features (title, description) using TF-IDF
    - Categorical features using one-hot encoding
    - Numerical features normalized to 0-1 range

    All features are combined and cosine similarity is used
    to find similar items.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

        self._tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
        )
        self._scaler = MinMaxScaler()
        self._encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

        self._item_ids: list[str] = []
        self._item_id_to_idx: dict[str, int] = {}
        self._feature_matrix: np.ndarray | None = None
        self._similarity_matrix: np.ndarray | None = None
        self._is_trained = False

        # Training metadata
        self._training_info: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        items: dict[str, dict[str, Any]],
        text_fields: list[str] | None = None,
        categorical_fields: list[str] | None = None,
        numerical_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train the content-based model on item data.

        Args:
            items: Dict mapping item_id to item data
            text_fields: Fields to use for TF-IDF (default: title, description)
            categorical_fields: Categorical feature fields
            numerical_fields: Numerical feature fields

        Returns:
            Training metrics and info
        """
        if not items:
            raise ValueError("No items provided for training")

        text_fields = text_fields or ["title", "description"]
        categorical_fields = categorical_fields or ["item_type"]
        numerical_fields = numerical_fields or []

        self._item_ids = list(items.keys())
        self._item_id_to_idx = {iid: idx for idx, iid in enumerate(self._item_ids)}

        logger.info(
            "Training content-based model",
            num_items=len(items),
            text_fields=text_fields,
            categorical_fields=categorical_fields,
        )

        feature_matrices = []

        # 1. Text features with TF-IDF
        text_corpus = []
        for item_id in self._item_ids:
            item = items[item_id]
            text_parts = []
            for field in text_fields:
                value = item.get(field) or item.get("features", {}).get(field, "")
                if value:
                    text_parts.append(str(value))
            text_corpus.append(" ".join(text_parts))

        if any(text_corpus):
            tfidf_matrix = self._tfidf.fit_transform(text_corpus)
            feature_matrices.append(tfidf_matrix)
            logger.debug("TF-IDF features", shape=tfidf_matrix.shape)

        # 2. Categorical features
        cat_data = []
        for item_id in self._item_ids:
            item = items[item_id]
            row = []
            for field in categorical_fields:
                value = item.get(field)
                if hasattr(value, "value"):  # Enum
                    value = value.value
                row.append(str(value) if value else "unknown")
            cat_data.append(row)

        if cat_data and categorical_fields:
            cat_matrix = self._encoder.fit_transform(cat_data)
            feature_matrices.append(cat_matrix)
            logger.debug("Categorical features", shape=cat_matrix.shape)

        # 3. Numerical features
        num_data = []
        for item_id in self._item_ids:
            item = items[item_id]
            row = []
            for field in numerical_fields:
                value = item.get(field) or item.get("features", {}).get(field, 0)
                try:
                    row.append(float(value))
                except (TypeError, ValueError):
                    row.append(0.0)
            num_data.append(row)

        if num_data and numerical_fields and any(any(r) for r in num_data):
            num_array = np.array(num_data)
            num_scaled = self._scaler.fit_transform(num_array)
            feature_matrices.append(csr_matrix(num_scaled))
            logger.debug("Numerical features", shape=num_scaled.shape)

        # Combine all features
        if not feature_matrices:
            raise ValueError("No features could be extracted from items")

        from scipy.sparse import hstack

        if len(feature_matrices) == 1:
            combined = feature_matrices[0]
        else:
            combined = hstack(feature_matrices)

        # Convert to dense if small enough, otherwise keep sparse
        if combined.shape[0] * combined.shape[1] < 1_000_000:
            self._feature_matrix = combined.toarray()
        else:
            self._feature_matrix = combined

        # Compute similarity matrix
        self._similarity_matrix = cosine_similarity(self._feature_matrix)
        self._is_trained = True

        self._training_info = {
            "num_items": len(items),
            "feature_dimensions": self._feature_matrix.shape[1],
            "text_fields": text_fields,
            "categorical_fields": categorical_fields,
            "numerical_fields": numerical_fields,
        }

        logger.info(
            "Content-based model trained",
            num_items=len(items),
            feature_dims=self._feature_matrix.shape[1],
        )

        return self._training_info

    def get_similar_items(
        self,
        item_id: str,
        n: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Get items most similar to a given item.

        Args:
            item_id: Source item ID
            n: Number of similar items to return
            exclude_self: Whether to exclude the source item

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if item_id not in self._item_id_to_idx:
            logger.warning("Item not in model", item_id=item_id)
            return []

        idx = self._item_id_to_idx[item_id]
        similarities = self._similarity_matrix[idx]

        # Get top similar items
        if exclude_self:
            similarities = similarities.copy()
            similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:n]

        results = []
        for i in top_indices:
            if similarities[i] > 0:
                results.append((self._item_ids[i], float(similarities[i])))

        return results

    def get_recommendations_for_user(
        self,
        liked_item_ids: list[str],
        n: int = 10,
        exclude_items: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Get recommendations based on items a user has liked.

        Aggregates similarity scores from all liked items.

        Args:
            liked_item_ids: Items the user has positively interacted with
            n: Number of recommendations
            exclude_items: Items to exclude from results

        Returns:
            List of (item_id, aggregated_score) tuples
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if not liked_item_ids:
            return []

        exclude_items = exclude_items or set()
        exclude_items.update(liked_item_ids)

        # Aggregate similarities from all liked items
        scores = np.zeros(len(self._item_ids))

        valid_liked = [iid for iid in liked_item_ids if iid in self._item_id_to_idx]
        if not valid_liked:
            return []

        for item_id in valid_liked:
            idx = self._item_id_to_idx[item_id]
            scores += self._similarity_matrix[idx]

        # Average the scores
        scores /= len(valid_liked)

        # Exclude items
        for item_id in exclude_items:
            if item_id in self._item_id_to_idx:
                scores[self._item_id_to_idx[item_id]] = -1

        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append((self._item_ids[i], float(scores[i])))

        return results

    def save(self, path: Path):
        """Save the trained model to disk."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "item_ids": self._item_ids,
            "item_id_to_idx": self._item_id_to_idx,
            "feature_matrix": self._feature_matrix,
            "similarity_matrix": self._similarity_matrix,
            "training_info": self._training_info,
            "tfidf": self._tfidf,
            "scaler": self._scaler,
            "encoder": self._encoder,
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Content-based model saved", path=str(path))

    def load(self, path: Path):
        """Load a trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self._item_ids = model_data["item_ids"]
        self._item_id_to_idx = model_data["item_id_to_idx"]
        self._feature_matrix = model_data["feature_matrix"]
        self._similarity_matrix = model_data["similarity_matrix"]
        self._training_info = model_data["training_info"]
        self._tfidf = model_data["tfidf"]
        self._scaler = model_data["scaler"]
        self._encoder = model_data["encoder"]

        config = model_data.get("config", {})
        self.max_features = config.get("max_features", self.max_features)
        self.ngram_range = config.get("ngram_range", self.ngram_range)
        self.min_df = config.get("min_df", self.min_df)

        self._is_trained = True
        logger.info("Content-based model loaded", path=str(path))

    def get_training_info(self) -> dict[str, Any]:
        """Get information about the trained model."""
        return self._training_info.copy()
