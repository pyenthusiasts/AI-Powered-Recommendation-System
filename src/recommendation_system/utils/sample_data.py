"""Sample data generator for testing and demos."""

import random
from datetime import datetime, timedelta

from recommendation_system.schemas import (
    InteractionCreate,
    InteractionType,
    ItemCreate,
    ItemType,
)


def generate_movies(n: int = 50) -> list[ItemCreate]:
    """Generate sample movie data."""
    genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller", "Animation"]
    directors = ["Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "Quentin Tarantino",
                 "Denis Villeneuve", "Greta Gerwig", "Jordan Peele", "Bong Joon-ho"]

    movies = []
    for i in range(n):
        genre = random.choice(genres)
        movies.append(ItemCreate(
            item_id=f"movie_{i+1}",
            title=f"The {random.choice(['Great', 'Amazing', 'Incredible', 'Dark', 'Lost'])} "
                  f"{random.choice(['Adventure', 'Journey', 'Secret', 'Night', 'Dream'])} {i+1}",
            item_type=ItemType.MOVIE,
            description=f"A {genre.lower()} film about extraordinary events and memorable characters.",
            features={
                "genre": genre,
                "year": random.randint(1990, 2024),
                "duration_minutes": random.randint(90, 180),
                "rating": round(random.uniform(5.0, 9.5), 1),
            },
            metadata={
                "director": random.choice(directors),
                "language": random.choice(["English", "Spanish", "French", "Korean", "Japanese"]),
                "country": random.choice(["USA", "UK", "France", "South Korea", "Japan"]),
            },
        ))
    return movies


def generate_books(n: int = 50) -> list[ItemCreate]:
    """Generate sample book data."""
    genres = ["Fiction", "Non-Fiction", "Mystery", "Fantasy", "Science Fiction", "Biography", "Self-Help"]
    authors = ["Stephen King", "J.K. Rowling", "George R.R. Martin", "Agatha Christie",
               "Isaac Asimov", "Malcolm Gladwell", "Michelle Obama", "Yuval Noah Harari"]

    books = []
    for i in range(n):
        genre = random.choice(genres)
        books.append(ItemCreate(
            item_id=f"book_{i+1}",
            title=f"The {random.choice(['Hidden', 'Forgotten', 'Last', 'First', 'Secret'])} "
                  f"{random.choice(['Kingdom', 'Path', 'Truth', 'Legacy', 'Code'])}",
            item_type=ItemType.BOOK,
            description=f"A compelling {genre.lower()} book that explores deep themes.",
            features={
                "genre": genre,
                "year": random.randint(1950, 2024),
                "pages": random.randint(150, 800),
                "rating": round(random.uniform(3.5, 5.0), 1),
            },
            metadata={
                "author": random.choice(authors),
                "publisher": random.choice(["Penguin", "HarperCollins", "Simon & Schuster", "Random House"]),
                "format": random.choice(["Hardcover", "Paperback", "eBook", "Audiobook"]),
            },
        ))
    return books


def generate_songs(n: int = 50) -> list[ItemCreate]:
    """Generate sample song data."""
    genres = ["Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical", "R&B", "Country"]
    artists = ["Taylor Swift", "Ed Sheeran", "Drake", "Beyonce", "The Weeknd",
               "Dua Lipa", "Bad Bunny", "BTS", "Adele", "Post Malone"]

    songs = []
    for i in range(n):
        genre = random.choice(genres)
        songs.append(ItemCreate(
            item_id=f"song_{i+1}",
            title=f"{random.choice(['Midnight', 'Summer', 'Electric', 'Golden', 'Neon'])} "
                  f"{random.choice(['Dreams', 'Love', 'Lights', 'Heart', 'Sky'])}",
            item_type=ItemType.SONG,
            description=f"A {genre.lower()} track with catchy melodies.",
            features={
                "genre": genre,
                "year": random.randint(2000, 2024),
                "duration_seconds": random.randint(180, 300),
                "bpm": random.randint(80, 160),
            },
            metadata={
                "artist": random.choice(artists),
                "album": f"Album {random.randint(1, 10)}",
                "explicit": random.choice([True, False]),
            },
        ))
    return songs


def generate_courses(n: int = 50) -> list[ItemCreate]:
    """Generate sample course data."""
    topics = ["Programming", "Data Science", "Machine Learning", "Web Development",
              "Business", "Design", "Marketing", "Finance"]
    platforms = ["Coursera", "Udemy", "edX", "LinkedIn Learning", "Pluralsight"]
    levels = ["Beginner", "Intermediate", "Advanced"]

    courses = []
    for i in range(n):
        topic = random.choice(topics)
        courses.append(ItemCreate(
            item_id=f"course_{i+1}",
            title=f"{random.choice(['Complete', 'Ultimate', 'Modern', 'Practical'])} "
                  f"{topic} {random.choice(['Masterclass', 'Bootcamp', 'Fundamentals', 'Guide'])}",
            item_type=ItemType.COURSE,
            description=f"Learn {topic.lower()} with hands-on projects and real-world examples.",
            features={
                "topic": topic,
                "level": random.choice(levels),
                "duration_hours": random.randint(5, 100),
                "rating": round(random.uniform(3.5, 5.0), 1),
            },
            metadata={
                "platform": random.choice(platforms),
                "instructor": f"Professor {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}",
                "certificate": random.choice([True, False]),
                "price": round(random.uniform(0, 200), 2),
            },
        ))
    return courses


def generate_interactions(
    user_ids: list[str],
    item_ids: list[str],
    n_per_user: int = 20,
) -> list[InteractionCreate]:
    """Generate random user-item interactions."""
    interactions = []
    interaction_types = list(InteractionType)

    for user_id in user_ids:
        # Each user interacts with a subset of items
        sampled_items = random.sample(item_ids, min(n_per_user, len(item_ids)))

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


def load_sample_data(
    data_store,
    n_items: int = 50,
    n_users: int = 100,
    interactions_per_user: int = 20,
):
    """Load sample data into the data store.

    Args:
        data_store: DataStore instance
        n_items: Number of items per category
        n_users: Number of users to create
        interactions_per_user: Average interactions per user
    """
    from recommendation_system.schemas import UserCreate

    # Generate items
    all_items = []
    all_items.extend(generate_movies(n_items))
    all_items.extend(generate_books(n_items))
    all_items.extend(generate_songs(n_items))
    all_items.extend(generate_courses(n_items))

    # Add items to store
    for item in all_items:
        try:
            data_store.add_item(item)
        except ValueError:
            pass  # Item already exists

    # Create users
    user_ids = [f"user_{i+1}" for i in range(n_users)]
    for user_id in user_ids:
        try:
            data_store.add_user(UserCreate(user_id=user_id))
        except ValueError:
            pass  # User already exists

    # Generate and add interactions
    item_ids = [item.item_id for item in all_items]
    interactions = generate_interactions(user_ids, item_ids, interactions_per_user)

    for interaction in interactions:
        try:
            data_store.add_interaction(interaction)
        except ValueError:
            pass  # Invalid interaction

    return {
        "items_created": len(all_items),
        "users_created": len(user_ids),
        "interactions_created": len(interactions),
    }
