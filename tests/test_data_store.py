"""Tests for the data store module."""

import pytest

from recommendation_system.schemas import (
    InteractionCreate,
    InteractionType,
    ItemCreate,
    ItemType,
    ItemUpdate,
    UserCreate,
)


class TestDataStoreItems:
    """Tests for item operations."""

    def test_add_item(self, data_store):
        """Test adding a new item."""
        item = ItemCreate(
            item_id="test_item",
            title="Test Item",
            item_type=ItemType.MOVIE,
            description="A test item",
        )
        result = data_store.add_item(item)

        assert result.item_id == "test_item"
        assert result.title == "Test Item"
        assert result.item_type == ItemType.MOVIE

    def test_add_duplicate_item_raises_error(self, data_store):
        """Test that adding a duplicate item raises ValueError."""
        item = ItemCreate(
            item_id="duplicate",
            title="Item 1",
            item_type=ItemType.MOVIE,
        )
        data_store.add_item(item)

        with pytest.raises(ValueError, match="already exists"):
            data_store.add_item(item)

    def test_get_item(self, data_store, sample_items):
        """Test getting an item by ID."""
        item = data_store.get_item("movie_1")

        assert item is not None
        assert item.title == "The Matrix"

    def test_get_nonexistent_item(self, data_store):
        """Test getting a nonexistent item returns None."""
        item = data_store.get_item("nonexistent")
        assert item is None

    def test_update_item(self, data_store, sample_items):
        """Test updating an item."""
        update = ItemUpdate(title="The Matrix Reloaded")
        result = data_store.update_item("movie_1", update)

        assert result is not None
        assert result.title == "The Matrix Reloaded"

    def test_delete_item(self, data_store, sample_items):
        """Test deleting an item."""
        result = data_store.delete_item("movie_1")
        assert result is True

        item = data_store.get_item("movie_1")
        assert item is None

    def test_get_items_with_filter(self, data_store, sample_items):
        """Test getting items with type filter."""
        movies = data_store.get_items(item_type=ItemType.MOVIE)

        assert len(movies) == 3
        assert all(item.item_type == ItemType.MOVIE for item in movies)

    def test_get_items_pagination(self, data_store, sample_items):
        """Test pagination of items."""
        items = data_store.get_items(limit=2, offset=0)
        assert len(items) == 2

        items2 = data_store.get_items(limit=2, offset=2)
        assert len(items2) == 2


class TestDataStoreUsers:
    """Tests for user operations."""

    def test_add_user(self, data_store):
        """Test adding a new user."""
        user = UserCreate(user_id="new_user", preferences={"lang": "en"})
        result = data_store.add_user(user)

        assert result.user_id == "new_user"
        assert result.preferences == {"lang": "en"}

    def test_get_user(self, data_store, sample_users):
        """Test getting a user."""
        user = data_store.get_user("user_1")

        assert user is not None
        assert user.preferences["favorite_genre"] == "Sci-Fi"

    def test_get_or_create_user(self, data_store):
        """Test get or create user."""
        # First call creates
        user1 = data_store.get_or_create_user("auto_user")
        assert user1.user_id == "auto_user"

        # Second call gets existing
        user2 = data_store.get_or_create_user("auto_user")
        assert user2.user_id == user1.user_id


class TestDataStoreInteractions:
    """Tests for interaction operations."""

    def test_add_interaction(self, data_store, sample_items):
        """Test adding an interaction."""
        interaction = InteractionCreate(
            user_id="new_user",
            item_id="movie_1",
            interaction_type=InteractionType.LIKE,
        )
        result = data_store.add_interaction(interaction)

        assert result.user_id == "new_user"
        assert result.item_id == "movie_1"
        assert result.interaction_type == InteractionType.LIKE

    def test_add_interaction_creates_user(self, data_store, sample_items):
        """Test that adding interaction creates user if not exists."""
        interaction = InteractionCreate(
            user_id="auto_created",
            item_id="movie_1",
            interaction_type=InteractionType.VIEW,
        )
        data_store.add_interaction(interaction)

        user = data_store.get_user("auto_created")
        assert user is not None

    def test_add_interaction_invalid_item(self, data_store):
        """Test that interaction with invalid item raises error."""
        interaction = InteractionCreate(
            user_id="user",
            item_id="nonexistent",
            interaction_type=InteractionType.VIEW,
        )

        with pytest.raises(ValueError, match="does not exist"):
            data_store.add_interaction(interaction)

    def test_get_user_interactions(self, data_store, sample_interactions):
        """Test getting user interactions."""
        interactions = data_store.get_user_interactions("user_1")

        assert len(interactions) == 3
        assert all(i.user_id == "user_1" for i in interactions)

    def test_get_user_interactions_filtered(self, data_store, sample_interactions):
        """Test getting filtered user interactions."""
        interactions = data_store.get_user_interactions(
            "user_1",
            interaction_types=[InteractionType.LIKE],
        )

        assert len(interactions) == 2
        assert all(i.interaction_type == InteractionType.LIKE for i in interactions)

    def test_get_user_item_ids(self, data_store, sample_interactions):
        """Test getting user's interacted item IDs."""
        item_ids = data_store.get_user_item_ids("user_1")

        assert "movie_1" in item_ids
        assert "movie_2" in item_ids
        assert "book_1" in item_ids

    def test_build_interaction_matrix(self, data_store, sample_interactions):
        """Test building interaction matrix."""
        matrix, user_ids, item_ids = data_store.build_interaction_matrix()

        assert matrix.shape[0] == len(user_ids)
        assert matrix.shape[1] == len(item_ids)
        assert "user_1" in user_ids
        assert "movie_1" in item_ids


class TestDataStoreStats:
    """Tests for statistics."""

    def test_get_stats(self, data_store, sample_interactions):
        """Test getting statistics."""
        stats = data_store.get_stats()

        assert stats["total_users"] == 3
        assert stats["total_items"] == 5
        assert stats["total_interactions"] == 7
        assert "movie" in stats["items_by_type"]
        assert "book" in stats["items_by_type"]
