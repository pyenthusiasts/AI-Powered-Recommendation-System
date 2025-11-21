# AI-Powered Recommendation System

A production-ready recommendation system with content-based and collaborative filtering algorithms, served via FastAPI.

## Features

- **Content-Based Filtering**: Recommends items similar to those a user has liked based on item features (text, categories, numerical attributes)
- **Collaborative Filtering**: Matrix factorization (SVD) to find patterns in user-item interactions
- **Hybrid Recommendations**: Combines both approaches with configurable weights
- **Multi-Domain Support**: Movies, books, songs, courses, and custom product types
- **REST API**: Full-featured FastAPI service with OpenAPI documentation
- **Production Ready**: Docker support, health checks, Prometheus metrics, structured logging

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run the API server
python -m recommendation_system.cli serve --reload

# Or use uvicorn directly
uvicorn recommendation_system.api.app:app --reload
```

The API will be available at http://localhost:8000

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker

```bash
# Build and run
docker-compose up -d

# With monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

## API Usage

### 1. Add Items

```bash
# Add a movie
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "movie_1",
    "title": "The Matrix",
    "item_type": "movie",
    "description": "A sci-fi action film about virtual reality",
    "features": {"genre": "Sci-Fi", "year": 1999},
    "metadata": {"director": "Wachowskis"}
  }'

# Bulk add items
curl -X POST http://localhost:8000/items/bulk \
  -H "Content-Type: application/json" \
  -d '[
    {"item_id": "movie_2", "title": "Inception", "item_type": "movie", "features": {"genre": "Sci-Fi"}},
    {"item_id": "movie_3", "title": "The Dark Knight", "item_type": "movie", "features": {"genre": "Action"}}
  ]'
```

### 2. Record Interactions

```bash
# Record a user liking an item
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "item_id": "movie_1",
    "interaction_type": "like"
  }'

# Record a rating
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "item_id": "movie_2",
    "interaction_type": "rating",
    "value": 8.5
  }'
```

### 3. Train Models

```bash
# Train all models (hybrid)
curl -X POST http://localhost:8000/train

# Train specific model type
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "content_based"}'
```

### 4. Get Recommendations

```bash
# Get recommendations for a user
curl "http://localhost:8000/recommendations/user/user_123?n=10"

# Get similar items
curl "http://localhost:8000/recommendations/item/movie_1/similar?n=5"

# Advanced recommendation request
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "strategy": "hybrid",
    "num_recommendations": 10,
    "item_type": "movie",
    "exclude_interacted": true
  }'
```

## Recommendation Strategies

### Content-Based Filtering
Best for:
- New users with few interactions (cold start)
- When item features are rich and meaningful
- Finding items similar to a specific item

Uses TF-IDF on text fields and one-hot encoding for categories.

### Collaborative Filtering
Best for:
- Users with interaction history
- Discovering diverse recommendations
- Finding items liked by similar users

Uses SVD matrix factorization on the user-item interaction matrix.

### Hybrid (Default)
Combines both approaches with configurable weights. Automatically handles cold start by favoring content-based for new users.

## Item Types

- `movie`: Films with genre, year, director
- `book`: Books with genre, author, pages
- `song`: Music with genre, artist, BPM
- `course`: Educational courses with topic, level, duration
- `product`: Generic products

## Interaction Types

- `view`: User viewed the item
- `click`: User clicked on the item
- `like`: User liked/favorited the item
- `dislike`: User disliked the item
- `purchase`: User purchased/enrolled
- `rating`: User rated (0-10 scale)
- `bookmark`: User saved for later

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/production) | development |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |
| `REDIS_URL` | Redis URL for caching | None |
| `API_KEY_ENABLED` | Enable API key auth | false |
| `API_KEY` | API key value | - |
| `MODEL_PATH` | Path to save/load models | ./models |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/recommendation_system --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Project Structure

```
.
├── src/recommendation_system/
│   ├── api/           # FastAPI application
│   │   └── app.py     # Main API endpoints
│   ├── models/        # ML models
│   │   ├── content_based.py    # TF-IDF + cosine similarity
│   │   ├── collaborative.py    # SVD matrix factorization
│   │   └── hybrid.py           # Combined recommender
│   ├── services/      # Business logic
│   │   ├── data_store.py       # In-memory data store
│   │   └── recommendation_service.py
│   ├── utils/         # Utilities
│   │   ├── sample_data.py      # Sample data generator
│   │   └── logging.py          # Structured logging
│   ├── config.py      # Configuration
│   ├── schemas.py     # Pydantic models
│   └── cli.py         # CLI commands
├── tests/             # Test suite
├── Dockerfile         # Container image
├── docker-compose.yml # Docker orchestration
└── pyproject.toml     # Python package config
```

## Performance Tips

1. **Batch Operations**: Use bulk endpoints for adding items/interactions
2. **Model Training**: Retrain periodically (e.g., daily) as new data arrives
3. **Caching**: Enable Redis for caching recommendations
4. **Monitoring**: Use `/metrics` endpoint with Prometheus

## License

MIT
