"""Pytest fixtures for the recommendation system tests."""

import pytest
from fastapi.testclient import TestClient

from recommendation_system.api.app import create_app
from recommendation_system.config import Settings
from recommendation_system.services.data_store import DataStore, reset_data_store
from recommendation_system.services.recommendation_service import (
    RecommendationService,
    reset_recommendation_service,
)


@pytest.fixture
def settings(tmp_path):
    """Create test settings."""
    return Settings(
        app_env="development",
        debug=True,
        api_key_enabled=False,
        model_path=tmp_path / "models",  # Use temp dir to avoid loading stale models
    )


@pytest.fixture
def data_store():
    """Create a fresh data store for each test."""
    reset_data_store()
    from recommendation_system.services.data_store import get_data_store
    store = get_data_store()
    yield store
    store.clear()


@pytest.fixture
def recommendation_service(data_store):
    """Create a recommendation service with fresh data store."""
    reset_recommendation_service()
    service = RecommendationService(data_store)
    return service


@pytest.fixture
def client(settings, data_store):
    """Create a test client."""
    # Also reset recommendation service to ensure it uses the fresh data store
    reset_recommendation_service()
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_items(data_store):
    """Create sample items for testing."""
    from recommendation_system.schemas import ItemCreate, ItemType

    items = [
        ItemCreate(
            item_id="movie_1",
            title="The Matrix",
            item_type=ItemType.MOVIE,
            description="A sci-fi action film about reality",
            features={"genre": "Sci-Fi", "year": 1999},
            metadata={"director": "Wachowskis"},
        ),
        ItemCreate(
            item_id="movie_2",
            title="Inception",
            item_type=ItemType.MOVIE,
            description="A sci-fi thriller about dreams",
            features={"genre": "Sci-Fi", "year": 2010},
            metadata={"director": "Christopher Nolan"},
        ),
        ItemCreate(
            item_id="movie_3",
            title="The Dark Knight",
            item_type=ItemType.MOVIE,
            description="A superhero action film",
            features={"genre": "Action", "year": 2008},
            metadata={"director": "Christopher Nolan"},
        ),
        ItemCreate(
            item_id="book_1",
            title="Dune",
            item_type=ItemType.BOOK,
            description="A sci-fi epic about desert planet",
            features={"genre": "Science Fiction", "year": 1965},
            metadata={"author": "Frank Herbert"},
        ),
        ItemCreate(
            item_id="book_2",
            title="1984",
            item_type=ItemType.BOOK,
            description="A dystopian novel about totalitarianism",
            features={"genre": "Science Fiction", "year": 1949},
            metadata={"author": "George Orwell"},
        ),
    ]

    created = []
    for item in items:
        created.append(data_store.add_item(item))

    return created


@pytest.fixture
def sample_users(data_store):
    """Create sample users for testing."""
    from recommendation_system.schemas import UserCreate

    users = [
        UserCreate(user_id="user_1", preferences={"favorite_genre": "Sci-Fi"}),
        UserCreate(user_id="user_2", preferences={"favorite_genre": "Action"}),
        UserCreate(user_id="user_3", preferences={}),
    ]

    created = []
    for user in users:
        created.append(data_store.add_user(user))

    return created


@pytest.fixture
def sample_interactions(data_store, sample_items, sample_users):
    """Create sample interactions for testing."""
    from recommendation_system.schemas import InteractionCreate, InteractionType

    interactions = [
        # User 1 likes sci-fi
        InteractionCreate(
            user_id="user_1",
            item_id="movie_1",
            interaction_type=InteractionType.LIKE,
        ),
        InteractionCreate(
            user_id="user_1",
            item_id="movie_2",
            interaction_type=InteractionType.LIKE,
        ),
        InteractionCreate(
            user_id="user_1",
            item_id="book_1",
            interaction_type=InteractionType.RATING,
            value=9.0,
        ),
        # User 2 likes action
        InteractionCreate(
            user_id="user_2",
            item_id="movie_3",
            interaction_type=InteractionType.LIKE,
        ),
        InteractionCreate(
            user_id="user_2",
            item_id="movie_1",
            interaction_type=InteractionType.VIEW,
        ),
        # User 3 mixed
        InteractionCreate(
            user_id="user_3",
            item_id="movie_1",
            interaction_type=InteractionType.RATING,
            value=8.0,
        ),
        InteractionCreate(
            user_id="user_3",
            item_id="book_2",
            interaction_type=InteractionType.PURCHASE,
        ),
    ]

    created = []
    for interaction in interactions:
        created.append(data_store.add_interaction(interaction))

    return created
