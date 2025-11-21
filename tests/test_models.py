"""Tests for the recommendation models."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from recommendation_system.models.content_based import ContentBasedModel
from recommendation_system.models.collaborative import CollaborativeFilteringModel
from recommendation_system.models.hybrid import HybridRecommender
from recommendation_system.schemas import ItemType


@pytest.fixture
def sample_items_dict():
    """Sample items as dictionary for model training."""
    return {
        "item_1": {
            "title": "The Matrix",
            "description": "A sci-fi action film about virtual reality",
            "item_type": ItemType.MOVIE,
            "features": {"genre": "Sci-Fi", "year": 1999},
        },
        "item_2": {
            "title": "Inception",
            "description": "A sci-fi thriller about dreams and reality",
            "item_type": ItemType.MOVIE,
            "features": {"genre": "Sci-Fi", "year": 2010},
        },
        "item_3": {
            "title": "The Dark Knight",
            "description": "An action superhero film about Batman",
            "item_type": ItemType.MOVIE,
            "features": {"genre": "Action", "year": 2008},
        },
        "item_4": {
            "title": "Pulp Fiction",
            "description": "A crime drama with multiple storylines",
            "item_type": ItemType.MOVIE,
            "features": {"genre": "Crime", "year": 1994},
        },
        "item_5": {
            "title": "Interstellar",
            "description": "A sci-fi film about space exploration",
            "item_type": ItemType.MOVIE,
            "features": {"genre": "Sci-Fi", "year": 2014},
        },
    }


@pytest.fixture
def sample_interaction_matrix():
    """Sample interaction matrix for collaborative filtering."""
    # 5 users x 5 items
    matrix = np.array([
        [1.0, 0.8, 0.0, 0.0, 0.9],  # User 1 likes sci-fi
        [0.9, 0.7, 0.0, 0.0, 0.8],  # User 2 likes sci-fi
        [0.0, 0.2, 0.9, 0.8, 0.1],  # User 3 likes action/crime
        [0.0, 0.0, 0.8, 0.9, 0.0],  # User 4 likes action/crime
        [0.5, 0.5, 0.5, 0.5, 0.5],  # User 5 likes everything
    ])
    user_ids = ["user_1", "user_2", "user_3", "user_4", "user_5"]
    item_ids = ["item_1", "item_2", "item_3", "item_4", "item_5"]
    return matrix, user_ids, item_ids


class TestContentBasedModel:
    """Tests for content-based filtering model."""

    def test_train_model(self, sample_items_dict):
        """Test model training."""
        model = ContentBasedModel()
        result = model.train(sample_items_dict)

        assert model.is_trained
        assert result["num_items"] == 5
        assert result["feature_dimensions"] > 0

    def test_train_empty_items_raises(self):
        """Test that training with no items raises error."""
        model = ContentBasedModel()

        with pytest.raises(ValueError, match="No items provided"):
            model.train({})

    def test_get_similar_items(self, sample_items_dict):
        """Test getting similar items."""
        model = ContentBasedModel()
        model.train(sample_items_dict)

        similar = model.get_similar_items("item_1", n=3)

        assert len(similar) <= 3
        assert all(isinstance(item, tuple) for item in similar)
        assert all(len(item) == 2 for item in similar)
        # item_1 (Matrix) should be similar to other sci-fi
        similar_ids = [item[0] for item in similar]
        assert "item_2" in similar_ids or "item_5" in similar_ids

    def test_get_similar_items_exclude_self(self, sample_items_dict):
        """Test that similar items excludes the source item."""
        model = ContentBasedModel()
        model.train(sample_items_dict)

        similar = model.get_similar_items("item_1", n=5, exclude_self=True)

        assert "item_1" not in [item[0] for item in similar]

    def test_get_recommendations_for_user(self, sample_items_dict):
        """Test user recommendations based on liked items."""
        model = ContentBasedModel()
        model.train(sample_items_dict)

        # User liked item_1 (Matrix) and item_2 (Inception)
        recs = model.get_recommendations_for_user(
            liked_item_ids=["item_1", "item_2"],
            n=3,
            exclude_items={"item_1", "item_2"},
        )

        assert len(recs) <= 3
        # Should recommend item_5 (Interstellar) as it's also sci-fi
        rec_ids = [r[0] for r in recs]
        assert "item_1" not in rec_ids
        assert "item_2" not in rec_ids

    def test_get_similar_items_unknown_item(self, sample_items_dict):
        """Test handling of unknown item."""
        model = ContentBasedModel()
        model.train(sample_items_dict)

        similar = model.get_similar_items("unknown_item")
        assert similar == []

    def test_model_not_trained_error(self):
        """Test that using untrained model raises error."""
        model = ContentBasedModel()

        with pytest.raises(RuntimeError, match="not trained"):
            model.get_similar_items("item_1")

    def test_save_and_load(self, sample_items_dict, tmp_path):
        """Test model save and load."""
        model = ContentBasedModel()
        model.train(sample_items_dict)

        # Save
        save_path = tmp_path / "content_model.pkl"
        model.save(save_path)

        # Load into new model
        model2 = ContentBasedModel()
        model2.load(save_path)

        assert model2.is_trained
        similar1 = model.get_similar_items("item_1", n=3)
        similar2 = model2.get_similar_items("item_1", n=3)
        assert similar1 == similar2


class TestCollaborativeFilteringModel:
    """Tests for collaborative filtering model."""

    def test_train_model(self, sample_interaction_matrix):
        """Test model training."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        result = model.train(matrix, user_ids, item_ids)

        assert model.is_trained
        assert result["num_users"] == 5
        assert result["num_items"] == 5
        assert "explained_variance_ratio" in result

    def test_train_empty_matrix_raises(self):
        """Test that training with empty matrix raises error."""
        model = CollaborativeFilteringModel()

        with pytest.raises(ValueError, match="Empty interaction matrix"):
            model.train(np.array([]), [], [])

    def test_predict_rating(self, sample_interaction_matrix):
        """Test rating prediction."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        # User 1 should have high prediction for item_5 (similar to their liked items)
        rating = model.predict_rating("user_1", "item_5")
        assert 0 <= rating <= 1

    def test_get_recommendations_for_user(self, sample_interaction_matrix):
        """Test user recommendations."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        recs = model.get_recommendations_for_user(
            "user_1",
            n=3,
            exclude_items={"item_1", "item_2", "item_5"},
        )

        assert len(recs) <= 3
        rec_ids = [r[0] for r in recs]
        assert "item_1" not in rec_ids
        assert "item_2" not in rec_ids
        assert "item_5" not in rec_ids

    def test_get_similar_users(self, sample_interaction_matrix):
        """Test finding similar users."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        similar = model.get_similar_users("user_1", n=3)

        assert len(similar) <= 3
        # User 2 should be similar to User 1 (both like sci-fi)
        similar_ids = [s[0] for s in similar]
        assert "user_1" not in similar_ids  # Exclude self

    def test_get_similar_items(self, sample_interaction_matrix):
        """Test finding similar items based on user patterns."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        similar = model.get_similar_items("item_1", n=3)

        assert len(similar) <= 3
        similar_ids = [s[0] for s in similar]
        assert "item_1" not in similar_ids

    def test_cold_start_user(self, sample_interaction_matrix):
        """Test recommendations for unknown user (cold start)."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        recs = model.get_recommendations_for_user("new_user", n=3)

        # Should still return some recommendations (popular items)
        assert len(recs) > 0

    def test_save_and_load(self, sample_interaction_matrix, tmp_path):
        """Test model save and load."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        model = CollaborativeFilteringModel(n_factors=3)
        model.train(matrix, user_ids, item_ids)

        # Save
        save_path = tmp_path / "collab_model.pkl"
        model.save(save_path)

        # Load
        model2 = CollaborativeFilteringModel()
        model2.load(save_path)

        assert model2.is_trained
        rating1 = model.predict_rating("user_1", "item_1")
        rating2 = model2.predict_rating("user_1", "item_1")
        assert abs(rating1 - rating2) < 0.001


class TestHybridRecommender:
    """Tests for hybrid recommender."""

    def test_train_both_models(self, sample_items_dict, sample_interaction_matrix):
        """Test training both underlying models."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        hybrid = HybridRecommender()

        # Train content model
        content_result = hybrid.train_content_model(sample_items_dict)
        assert "num_items" in content_result

        # Train collaborative model
        collab_result = hybrid.train_collaborative_model(matrix, user_ids, item_ids)
        assert "num_users" in collab_result

        assert hybrid.is_trained

    def test_hybrid_recommendations(self, sample_items_dict, sample_interaction_matrix):
        """Test hybrid recommendations combining both strategies."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        hybrid = HybridRecommender()
        hybrid.train_content_model(sample_items_dict)
        hybrid.train_collaborative_model(matrix, user_ids, item_ids)

        recs = hybrid.get_recommendations(
            user_id="user_1",
            liked_items=["item_1", "item_2"],
            n=3,
            strategy="hybrid",
        )

        assert len(recs) <= 3
        assert all(len(r) == 3 for r in recs)  # (item_id, score, explanation)

    def test_content_only_recommendations(self, sample_items_dict):
        """Test content-based only recommendations."""
        hybrid = HybridRecommender()
        hybrid.train_content_model(sample_items_dict)

        recs = hybrid.get_recommendations(
            liked_items=["item_1"],
            n=3,
            strategy="content_based",
        )

        assert len(recs) <= 3

    def test_similar_items(self, sample_items_dict, sample_interaction_matrix):
        """Test getting similar items via hybrid."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        hybrid = HybridRecommender()
        hybrid.train_content_model(sample_items_dict)
        hybrid.train_collaborative_model(matrix, user_ids, item_ids)

        similar = hybrid.get_recommendations(
            item_id="item_1",
            n=3,
        )

        assert len(similar) <= 3

    def test_save_and_load(self, sample_items_dict, sample_interaction_matrix, tmp_path):
        """Test hybrid model save and load."""
        matrix, user_ids, item_ids = sample_interaction_matrix
        hybrid = HybridRecommender()
        hybrid.train_content_model(sample_items_dict)
        hybrid.train_collaborative_model(matrix, user_ids, item_ids)

        # Save
        hybrid.save(tmp_path)

        # Load
        hybrid2 = HybridRecommender()
        hybrid2.load(tmp_path)

        assert hybrid2.is_trained

        # Compare recommendations
        recs1 = hybrid.get_recommendations(item_id="item_1", n=3)
        recs2 = hybrid2.get_recommendations(item_id="item_1", n=3)
        assert [r[0] for r in recs1] == [r[0] for r in recs2]
