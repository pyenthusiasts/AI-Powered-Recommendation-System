"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health and monitoring endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "recommendation_requests_total" in response.text

    def test_stats_endpoint(self, client, sample_items):
        """Test stats endpoint."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_items" in data
        assert data["total_items"] == 5


class TestItemEndpoints:
    """Tests for item CRUD endpoints."""

    def test_create_item(self, client):
        """Test creating a new item."""
        item_data = {
            "item_id": "new_movie",
            "title": "New Movie",
            "item_type": "movie",
            "description": "A brand new movie",
            "features": {"genre": "Comedy"},
            "metadata": {"director": "Someone"},
        }
        response = client.post("/items", json=item_data)

        assert response.status_code == 201
        data = response.json()
        assert data["item_id"] == "new_movie"
        assert data["title"] == "New Movie"

    def test_create_duplicate_item(self, client, sample_items):
        """Test creating duplicate item returns 409."""
        item_data = {
            "item_id": "movie_1",
            "title": "Duplicate",
            "item_type": "movie",
        }
        response = client.post("/items", json=item_data)

        assert response.status_code == 409

    def test_get_item(self, client, sample_items):
        """Test getting an item by ID."""
        response = client.get("/items/movie_1")

        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == "movie_1"
        assert data["title"] == "The Matrix"

    def test_get_item_not_found(self, client):
        """Test getting nonexistent item returns 404."""
        response = client.get("/items/nonexistent")

        assert response.status_code == 404

    def test_update_item(self, client, sample_items):
        """Test updating an item."""
        update_data = {"title": "The Matrix Updated"}
        response = client.patch("/items/movie_1", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "The Matrix Updated"

    def test_delete_item(self, client, sample_items):
        """Test deleting an item."""
        response = client.delete("/items/movie_1")

        assert response.status_code == 204

        # Verify deleted
        response = client.get("/items/movie_1")
        assert response.status_code == 404

    def test_list_items(self, client, sample_items):
        """Test listing items."""
        response = client.get("/items")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_list_items_filtered(self, client, sample_items):
        """Test listing items with type filter."""
        response = client.get("/items?item_type=movie")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert all(item["item_type"] == "movie" for item in data)

    def test_list_items_paginated(self, client, sample_items):
        """Test pagination."""
        response = client.get("/items?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_bulk_create_items(self, client):
        """Test bulk item creation."""
        items = [
            {"item_id": f"bulk_{i}", "title": f"Bulk Item {i}", "item_type": "movie"}
            for i in range(3)
        ]
        response = client.post("/items/bulk", json=items)

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 3


class TestUserEndpoints:
    """Tests for user endpoints."""

    def test_create_user(self, client):
        """Test creating a user."""
        user_data = {"user_id": "new_user", "preferences": {"theme": "dark"}}
        response = client.post("/users", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == "new_user"

    def test_get_user(self, client, sample_users):
        """Test getting a user."""
        response = client.get("/users/user_1")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_1"

    def test_get_user_not_found(self, client):
        """Test getting nonexistent user."""
        response = client.get("/users/nonexistent")

        assert response.status_code == 404


class TestInteractionEndpoints:
    """Tests for interaction endpoints."""

    def test_create_interaction(self, client, sample_items):
        """Test creating an interaction."""
        interaction_data = {
            "user_id": "test_user",
            "item_id": "movie_1",
            "interaction_type": "like",
        }
        response = client.post("/interactions", json=interaction_data)

        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["item_id"] == "movie_1"

    def test_create_interaction_with_rating(self, client, sample_items):
        """Test creating a rating interaction."""
        interaction_data = {
            "user_id": "test_user",
            "item_id": "movie_1",
            "interaction_type": "rating",
            "value": 8.5,
        }
        response = client.post("/interactions", json=interaction_data)

        assert response.status_code == 201
        data = response.json()
        assert data["value"] == 8.5

    def test_create_interaction_invalid_item(self, client):
        """Test interaction with invalid item."""
        interaction_data = {
            "user_id": "user",
            "item_id": "nonexistent",
            "interaction_type": "like",
        }
        response = client.post("/interactions", json=interaction_data)

        assert response.status_code == 400

    def test_get_user_interactions(self, client, sample_interactions):
        """Test getting user interactions."""
        response = client.get("/users/user_1/interactions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_bulk_create_interactions(self, client, sample_items):
        """Test bulk interaction creation."""
        interactions = [
            {"user_id": "bulk_user", "item_id": f"movie_{i}", "interaction_type": "view"}
            for i in range(1, 4)
        ]
        response = client.post("/interactions/bulk", json=interactions)

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 3


class TestRecommendationEndpoints:
    """Tests for recommendation endpoints."""

    def test_get_recommendations_for_user(self, client, sample_interactions):
        """Test getting recommendations for a user."""
        # First train models
        client.post("/train")

        response = client.get("/recommendations/user/user_1?n=3")

        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "strategy_used" in data
        assert "processing_time_ms" in data

    def test_get_similar_items(self, client, sample_interactions):
        """Test getting similar items."""
        # Train models
        client.post("/train")

        response = client.get("/recommendations/item/movie_1/similar?n=3")

        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data

    def test_post_recommendations(self, client, sample_interactions):
        """Test POST recommendations with full options."""
        client.post("/train")

        request_data = {
            "user_id": "user_1",
            "strategy": "hybrid",
            "num_recommendations": 5,
            "exclude_interacted": True,
        }
        response = client.post("/recommendations", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) <= 5

    def test_recommendations_missing_params(self, client):
        """Test recommendations without required params."""
        response = client.post("/recommendations", json={})

        assert response.status_code == 400


class TestTrainingEndpoints:
    """Tests for training endpoints."""

    def test_train_models(self, client, sample_interactions):
        """Test training models."""
        response = client.post("/train")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "result" in data

    def test_train_specific_model(self, client, sample_interactions):
        """Test training specific model type."""
        response = client.post("/train", json={"model_type": "content_based"})

        assert response.status_code == 200
        data = response.json()
        assert "content_based" in data["result"]

    def test_train_no_items(self, client):
        """Test training with no items."""
        response = client.post("/train")

        assert response.status_code == 400

    def test_save_models(self, client, sample_interactions):
        """Test saving models."""
        client.post("/train")
        response = client.post("/models/save")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_load_models(self, client, sample_interactions):
        """Test loading models."""
        # First train and save
        client.post("/train")
        client.post("/models/save")

        # Then load
        response = client.post("/models/load")

        assert response.status_code == 200
