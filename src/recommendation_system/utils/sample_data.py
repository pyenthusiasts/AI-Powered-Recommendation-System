"""Sample data generator for DEVELOPMENT and TESTING only.

WARNING: This module generates synthetic test data and should NOT be used in production.
For production use, import real data using:
- /data/import API endpoint
- CSVDataLoader or JSONDataLoader from utils.data_loader
- Direct database import via the repository layer

See utils/data_loader.py for production data loading utilities.
"""

import random
from datetime import datetime, timedelta

from recommendation_system.schemas import (
    InteractionCreate,
    InteractionType,
    ItemCreate,
    ItemType,
)


def _generate_test_items(
    item_type: ItemType,
    n: int,
    prefix: str,
    categories: list[str],
) -> list[ItemCreate]:
    """Generate test items for a specific type.

    Args:
        item_type: The type of items to generate.
        n: Number of items to generate.
        prefix: Prefix for item IDs.
        categories: List of category names for the items.

    Returns:
        List of ItemCreate objects.
    """
    items = []
    for i in range(n):
        category = random.choice(categories)
        items.append(ItemCreate(
            item_id=f"{prefix}_{i+1}",
            title=f"Test {item_type.value.title()} {i+1}",
            item_type=item_type,
            description=f"Test {item_type.value} item in {category} category for development/testing.",
            features={
                "category": category,
                "year": random.randint(2000, 2024),
                "rating": round(random.uniform(3.0, 5.0), 1),
            },
            metadata={
                "source": "test_data_generator",
                "generated_at": datetime.utcnow().isoformat(),
            },
        ))
    return items


def generate_test_movies(n: int = 50) -> list[ItemCreate]:
    """Generate test movie data for development."""
    categories = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller"]
    return _generate_test_items(ItemType.MOVIE, n, "test_movie", categories)


def generate_test_books(n: int = 50) -> list[ItemCreate]:
    """Generate test book data for development."""
    categories = ["Fiction", "Non-Fiction", "Mystery", "Fantasy", "Science", "Biography"]
    return _generate_test_items(ItemType.BOOK, n, "test_book", categories)


def generate_test_songs(n: int = 50) -> list[ItemCreate]:
    """Generate test song data for development."""
    categories = ["Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical", "R&B"]
    return _generate_test_items(ItemType.SONG, n, "test_song", categories)


def generate_test_courses(n: int = 50) -> list[ItemCreate]:
    """Generate test course data for development."""
    categories = ["Programming", "Data Science", "Web Development", "Business", "Design"]
    return _generate_test_items(ItemType.COURSE, n, "test_course", categories)


def generate_test_interactions(
    user_ids: list[str],
    item_ids: list[str],
    n_per_user: int = 20,
) -> list[InteractionCreate]:
    """Generate random user-item interactions for testing.

    Args:
        user_ids: List of user IDs to generate interactions for.
        item_ids: List of item IDs available for interaction.
        n_per_user: Average number of interactions per user.

    Returns:
        List of InteractionCreate objects.
    """
    interactions = []
    interaction_types = list(InteractionType)

    for user_id in user_ids:
        # Each user interacts with a subset of items
        n_items = min(n_per_user, len(item_ids))
        sampled_items = random.sample(item_ids, n_items)

        for item_id in sampled_items:
            itype = random.choice(interaction_types)
            value = None

            if itype == InteractionType.RATING:
                value = round(random.uniform(1, 10), 1)

            timestamp = datetime.utcnow() - timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 24),
            )

            interactions.append(InteractionCreate(
                user_id=user_id,
                item_id=item_id,
                interaction_type=itype,
                value=value,
                timestamp=timestamp,
            ))

    return interactions


def load_test_data(
    data_store,
    n_items_per_type: int = 25,
    n_users: int = 50,
    interactions_per_user: int = 15,
) -> dict:
    """Load test data into the data store for development/testing.

    WARNING: This is for DEVELOPMENT and TESTING only.
    Do NOT use in production - use the data loader utilities instead.

    Args:
        data_store: DataStore instance to load data into.
        n_items_per_type: Number of items to generate per category.
        n_users: Number of test users to create.
        interactions_per_user: Average interactions per user.

    Returns:
        Dictionary with statistics about loaded data.
    """
    from recommendation_system.schemas import UserCreate

    # Generate items for each type
    all_items = []
    all_items.extend(generate_test_movies(n_items_per_type))
    all_items.extend(generate_test_books(n_items_per_type))
    all_items.extend(generate_test_songs(n_items_per_type))
    all_items.extend(generate_test_courses(n_items_per_type))

    # Add items to store
    items_created = 0
    for item in all_items:
        try:
            data_store.add_item(item)
            items_created += 1
        except ValueError:
            pass  # Item already exists

    # Create test users
    user_ids = [f"test_user_{i+1}" for i in range(n_users)]
    users_created = 0
    for user_id in user_ids:
        try:
            data_store.add_user(UserCreate(user_id=user_id))
            users_created += 1
        except ValueError:
            pass  # User already exists

    # Generate and add interactions
    item_ids = [item.item_id for item in all_items]
    interactions = generate_test_interactions(user_ids, item_ids, interactions_per_user)

    interactions_created = 0
    for interaction in interactions:
        try:
            data_store.add_interaction(interaction)
            interactions_created += 1
        except ValueError:
            pass  # Invalid interaction

    return {
        "items_created": items_created,
        "users_created": users_created,
        "interactions_created": interactions_created,
        "warning": "Test data loaded - DO NOT USE IN PRODUCTION",
    }


# Legacy aliases for backward compatibility - DEPRECATED
generate_movies = generate_test_movies
generate_books = generate_test_books
generate_songs = generate_test_songs
generate_courses = generate_test_courses
generate_interactions = generate_test_interactions
load_sample_data = load_test_data
