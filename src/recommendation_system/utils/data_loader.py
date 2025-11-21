"""Production data loading utilities.

This module provides utilities for loading item and interaction data from various sources
including CSV files, JSON files, and external APIs. It replaces the hardcoded sample data
with configurable data sources suitable for production use.
"""

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import structlog

from recommendation_system.schemas import (
    InteractionCreate,
    InteractionType,
    ItemCreate,
    ItemType,
    UserCreate,
)

logger = structlog.get_logger()


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_items(self) -> Generator[ItemCreate, None, None]:
        """Load items from data source."""
        pass

    @abstractmethod
    def load_users(self) -> Generator[UserCreate, None, None]:
        """Load users from data source."""
        pass

    @abstractmethod
    def load_interactions(self) -> Generator[InteractionCreate, None, None]:
        """Load interactions from data source."""
        pass


class CSVDataLoader(DataLoader):
    """Load data from CSV files."""

    def __init__(
        self,
        items_path: Path | str | None = None,
        users_path: Path | str | None = None,
        interactions_path: Path | str | None = None,
    ):
        """Initialize CSV data loader.

        Args:
            items_path: Path to items CSV file.
            users_path: Path to users CSV file.
            interactions_path: Path to interactions CSV file.
        """
        self.items_path = Path(items_path) if items_path else None
        self.users_path = Path(users_path) if users_path else None
        self.interactions_path = Path(interactions_path) if interactions_path else None

    def load_items(self) -> Generator[ItemCreate, None, None]:
        """Load items from CSV file.

        Expected CSV columns:
        - item_id: Unique identifier
        - title: Item title
        - item_type: One of movie, book, song, course, product
        - description: Optional description
        - features: JSON string of features dict
        - metadata: JSON string of metadata dict
        """
        if not self.items_path or not self.items_path.exists():
            logger.warning("Items CSV not found", path=str(self.items_path))
            return

        with open(self.items_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    yield ItemCreate(
                        item_id=row["item_id"],
                        title=row["title"],
                        item_type=ItemType(row["item_type"]),
                        description=row.get("description"),
                        features=json.loads(row.get("features", "{}")),
                        metadata=json.loads(row.get("metadata", "{}")),
                    )
                except Exception as e:
                    logger.error("Failed to parse item row", row=row, error=str(e))

    def load_users(self) -> Generator[UserCreate, None, None]:
        """Load users from CSV file.

        Expected CSV columns:
        - user_id: Unique identifier
        - preferences: JSON string of preferences dict
        """
        if not self.users_path or not self.users_path.exists():
            logger.warning("Users CSV not found", path=str(self.users_path))
            return

        with open(self.users_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    yield UserCreate(
                        user_id=row["user_id"],
                        preferences=json.loads(row.get("preferences", "{}")),
                    )
                except Exception as e:
                    logger.error("Failed to parse user row", row=row, error=str(e))

    def load_interactions(self) -> Generator[InteractionCreate, None, None]:
        """Load interactions from CSV file.

        Expected CSV columns:
        - user_id: User identifier
        - item_id: Item identifier
        - interaction_type: One of view, click, like, dislike, purchase, rating, bookmark
        - value: Optional numeric value (required for rating)
        - timestamp: Optional ISO timestamp
        """
        if not self.interactions_path or not self.interactions_path.exists():
            logger.warning("Interactions CSV not found", path=str(self.interactions_path))
            return

        with open(self.interactions_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = None
                    if row.get("timestamp"):
                        timestamp = datetime.fromisoformat(row["timestamp"])

                    value = None
                    if row.get("value"):
                        value = float(row["value"])

                    yield InteractionCreate(
                        user_id=row["user_id"],
                        item_id=row["item_id"],
                        interaction_type=InteractionType(row["interaction_type"]),
                        value=value,
                        timestamp=timestamp,
                    )
                except Exception as e:
                    logger.error("Failed to parse interaction row", row=row, error=str(e))


class JSONDataLoader(DataLoader):
    """Load data from JSON files."""

    def __init__(
        self,
        items_path: Path | str | None = None,
        users_path: Path | str | None = None,
        interactions_path: Path | str | None = None,
    ):
        """Initialize JSON data loader."""
        self.items_path = Path(items_path) if items_path else None
        self.users_path = Path(users_path) if users_path else None
        self.interactions_path = Path(interactions_path) if interactions_path else None

    def load_items(self) -> Generator[ItemCreate, None, None]:
        """Load items from JSON file (array of objects)."""
        if not self.items_path or not self.items_path.exists():
            return

        with open(self.items_path, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                try:
                    yield ItemCreate(
                        item_id=item["item_id"],
                        title=item["title"],
                        item_type=ItemType(item["item_type"]),
                        description=item.get("description"),
                        features=item.get("features", {}),
                        metadata=item.get("metadata", {}),
                    )
                except Exception as e:
                    logger.error("Failed to parse item", item=item, error=str(e))

    def load_users(self) -> Generator[UserCreate, None, None]:
        """Load users from JSON file."""
        if not self.users_path or not self.users_path.exists():
            return

        with open(self.users_path, encoding="utf-8") as f:
            data = json.load(f)
            for user in data:
                try:
                    yield UserCreate(
                        user_id=user["user_id"],
                        preferences=user.get("preferences", {}),
                    )
                except Exception as e:
                    logger.error("Failed to parse user", user=user, error=str(e))

    def load_interactions(self) -> Generator[InteractionCreate, None, None]:
        """Load interactions from JSON file."""
        if not self.interactions_path or not self.interactions_path.exists():
            return

        with open(self.interactions_path, encoding="utf-8") as f:
            data = json.load(f)
            for interaction in data:
                try:
                    timestamp = None
                    if interaction.get("timestamp"):
                        timestamp = datetime.fromisoformat(interaction["timestamp"])

                    yield InteractionCreate(
                        user_id=interaction["user_id"],
                        item_id=interaction["item_id"],
                        interaction_type=InteractionType(interaction["interaction_type"]),
                        value=interaction.get("value"),
                        timestamp=timestamp,
                    )
                except Exception as e:
                    logger.error("Failed to parse interaction", interaction=interaction, error=str(e))


class DatabaseExportLoader(DataLoader):
    """Load data exported from external databases (e.g., PostgreSQL dump)."""

    def __init__(self, export_dir: Path | str):
        """Initialize database export loader."""
        self.export_dir = Path(export_dir)
        self._json_loader = JSONDataLoader(
            items_path=self.export_dir / "items.json",
            users_path=self.export_dir / "users.json",
            interactions_path=self.export_dir / "interactions.json",
        )

    def load_items(self) -> Generator[ItemCreate, None, None]:
        return self._json_loader.load_items()

    def load_users(self) -> Generator[UserCreate, None, None]:
        return self._json_loader.load_users()

    def load_interactions(self) -> Generator[InteractionCreate, None, None]:
        return self._json_loader.load_interactions()


class DataImporter:
    """Import data from various sources into the recommendation system."""

    def __init__(self, data_store):
        """Initialize data importer.

        Args:
            data_store: DataStore or database session for storing data.
        """
        self.data_store = data_store

    def import_from_loader(
        self,
        loader: DataLoader,
        batch_size: int = 1000,
        skip_errors: bool = True,
    ) -> dict[str, Any]:
        """Import data from a data loader.

        Args:
            loader: DataLoader instance.
            batch_size: Number of records to process before committing.
            skip_errors: Whether to skip individual record errors.

        Returns:
            Dictionary with import statistics.
        """
        stats = {
            "items_imported": 0,
            "items_skipped": 0,
            "users_imported": 0,
            "users_skipped": 0,
            "interactions_imported": 0,
            "interactions_skipped": 0,
            "errors": [],
        }

        # Import items
        logger.info("Importing items...")
        for item in loader.load_items():
            try:
                self.data_store.add_item(item)
                stats["items_imported"] += 1
            except ValueError as e:
                stats["items_skipped"] += 1
                if not skip_errors:
                    raise
            except Exception as e:
                stats["errors"].append(f"Item {item.item_id}: {str(e)}")
                if not skip_errors:
                    raise

        # Import users
        logger.info("Importing users...")
        for user in loader.load_users():
            try:
                self.data_store.add_user(user)
                stats["users_imported"] += 1
            except ValueError:
                stats["users_skipped"] += 1
            except Exception as e:
                stats["errors"].append(f"User {user.user_id}: {str(e)}")
                if not skip_errors:
                    raise

        # Import interactions
        logger.info("Importing interactions...")
        for interaction in loader.load_interactions():
            try:
                self.data_store.add_interaction(interaction)
                stats["interactions_imported"] += 1
            except ValueError:
                stats["interactions_skipped"] += 1
            except Exception as e:
                stats["errors"].append(
                    f"Interaction {interaction.user_id}-{interaction.item_id}: {str(e)}"
                )
                if not skip_errors:
                    raise

        logger.info("Import completed", **stats)
        return stats

    def import_from_csv(
        self,
        items_path: str | None = None,
        users_path: str | None = None,
        interactions_path: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method to import from CSV files."""
        loader = CSVDataLoader(
            items_path=items_path,
            users_path=users_path,
            interactions_path=interactions_path,
        )
        return self.import_from_loader(loader, **kwargs)

    def import_from_json(
        self,
        items_path: str | None = None,
        users_path: str | None = None,
        interactions_path: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convenience method to import from JSON files."""
        loader = JSONDataLoader(
            items_path=items_path,
            users_path=users_path,
            interactions_path=interactions_path,
        )
        return self.import_from_loader(loader, **kwargs)


def create_sample_csv_files(output_dir: Path | str, n_items: int = 100, n_users: int = 50):
    """Create sample CSV files for testing data import.

    This creates properly structured CSV files that can be used as templates
    for production data import.
    """
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample items
    items_path = output_dir / "items.csv"
    with open(items_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "title", "item_type", "description", "features", "metadata"])

        item_types = ["movie", "book", "song", "course", "product"]
        for i in range(n_items):
            item_type = random.choice(item_types)
            features = json.dumps({
                "category": f"category_{random.randint(1, 10)}",
                "year": random.randint(2000, 2024),
                "rating": round(random.uniform(1, 5), 1),
            })
            metadata = json.dumps({"source": "sample_data"})
            writer.writerow([
                f"item_{i+1}",
                f"Sample {item_type.title()} {i+1}",
                item_type,
                f"Description for {item_type} item {i+1}",
                features,
                metadata,
            ])

    # Sample users
    users_path = output_dir / "users.csv"
    with open(users_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "preferences"])

        for i in range(n_users):
            preferences = json.dumps({
                "preferred_types": random.sample(item_types, k=random.randint(1, 3)),
            })
            writer.writerow([f"user_{i+1}", preferences])

    # Sample interactions
    interactions_path = output_dir / "interactions.csv"
    with open(interactions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "interaction_type", "value", "timestamp"])

        interaction_types = ["view", "click", "like", "rating", "purchase"]
        for i in range(n_users):
            # Each user has some interactions
            n_interactions = random.randint(5, 20)
            for _ in range(n_interactions):
                item_id = f"item_{random.randint(1, n_items)}"
                itype = random.choice(interaction_types)
                value = round(random.uniform(1, 10), 1) if itype == "rating" else ""
                timestamp = datetime.now().isoformat()
                writer.writerow([f"user_{i+1}", item_id, itype, value, timestamp])

    logger.info(
        "Sample CSV files created",
        output_dir=str(output_dir),
        items=n_items,
        users=n_users,
    )

    return {
        "items_path": str(items_path),
        "users_path": str(users_path),
        "interactions_path": str(interactions_path),
    }
