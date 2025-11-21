# AI-Powered Recommendation System

A fully production-ready recommendation engine with content-based filtering, collaborative filtering, and hybrid recommendation strategies. Built with FastAPI, PostgreSQL, Redis, and scikit-learn.

## Features

### Core Recommendation Algorithms
- **Content-Based Filtering**: TF-IDF vectorization + cosine similarity on item features
- **Collaborative Filtering**: SVD matrix factorization for user-item patterns
- **Hybrid Recommendations**: Intelligent combination with cold-start handling

### Production Infrastructure
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **Caching**: Redis-backed caching with intelligent invalidation
- **Rate Limiting**: Token bucket rate limiter (Redis or in-memory)
- **Async Tasks**: Background task queue for model training
- **API Security**: API key authentication, CORS configuration
- **Monitoring**: Prometheus metrics, structured logging, health checks
- **Docker**: Multi-stage builds, docker-compose with full stack

### Multi-Domain Support
- Movies, Books, Songs, Courses, and Generic Products
- 7 interaction types: view, click, like, dislike, purchase, rating, bookmark

## Quick Start

### Option 1: Docker (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/your-org/AI-Powered-Recommendation-System.git
cd AI-Powered-Recommendation-System

# Copy environment configuration
cp .env.example .env
# Edit .env and set your API_KEY and SECRET_KEY

# Start all services (API + PostgreSQL + Redis)
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run database migrations
alembic upgrade head

# Start the API server (development mode)
uvicorn recommendation_system.api.app:app --reload

# Or use the CLI
python -m recommendation_system.cli serve --reload
```

The API will be available at http://localhost:8000

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## API Usage

### Authentication

For protected endpoints, include your API key in the header:
```bash
-H "X-API-Key: your-api-key"
```

### 1. Add Items

```bash
# Add a single item
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "item_id": "movie_001",
    "title": "The Matrix",
    "item_type": "movie",
    "description": "A sci-fi action film about virtual reality",
    "features": {"genre": "Sci-Fi", "year": 1999, "rating": 8.7},
    "metadata": {"director": "Wachowskis", "runtime": 136}
  }'

# Bulk add items
curl -X POST http://localhost:8000/items/bulk \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '[
    {"item_id": "movie_002", "title": "Inception", "item_type": "movie", "features": {"genre": "Sci-Fi", "year": 2010}},
    {"item_id": "movie_003", "title": "The Dark Knight", "item_type": "movie", "features": {"genre": "Action", "year": 2008}}
  ]'
```

### 2. Record Interactions

```bash
# Record a like
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "user_id": "user_001",
    "item_id": "movie_001",
    "interaction_type": "like"
  }'

# Record a rating (0-10 scale)
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "user_id": "user_001",
    "item_id": "movie_002",
    "interaction_type": "rating",
    "value": 8.5
  }'
```

### 3. Train Models

```bash
# Train all models (hybrid - default)
curl -X POST http://localhost:8000/train \
  -H "X-API-Key: your-api-key"

# Train specific model type
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"model_type": "content_based"}'

# Async training (returns task ID)
curl -X POST "http://localhost:8000/train?async_mode=true" \
  -H "X-API-Key: your-api-key"

# Check training status
curl http://localhost:8000/tasks/{task_id} \
  -H "X-API-Key: your-api-key"
```

### 4. Get Recommendations

```bash
# Get recommendations for a user
curl "http://localhost:8000/recommendations/user/user_001?n=10" \
  -H "X-API-Key: your-api-key"

# Get similar items
curl "http://localhost:8000/recommendations/item/movie_001/similar?n=5" \
  -H "X-API-Key: your-api-key"

# Advanced recommendation request
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "user_id": "user_001",
    "strategy": "hybrid",
    "num_recommendations": 10,
    "item_type": "movie",
    "exclude_interacted": true
  }'
```

### 5. Import Data

```bash
# Import from CSV files
curl -X POST "http://localhost:8000/data/import?items_path=/data/items.csv&format=csv" \
  -H "X-API-Key: your-api-key"

# Import from JSON files
curl -X POST "http://localhost:8000/data/import?items_path=/data/items.json&format=json" \
  -H "X-API-Key: your-api-key"
```

## Configuration

See `.env.example` for all configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/staging/production) | development |
| `DATABASE_URL` | Database connection URL | sqlite:///./recommendation_system.db |
| `REDIS_URL` | Redis URL for caching/rate limiting | None |
| `API_KEY_ENABLED` | Enable API key authentication | true |
| `API_KEY` | API key value (min 32 chars for production) | - |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | true |
| `RATE_LIMIT_REQUESTS` | Requests per window | 100 |
| `MODEL_AUTO_SAVE` | Auto-save models after training | true |

### Production Configuration

For production deployments:

1. **Use PostgreSQL** instead of SQLite:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/recommendations
   ```

2. **Enable Redis** for caching and rate limiting:
   ```
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Set strong secrets**:
   ```
   API_KEY=your-32-character-minimum-api-key
   SECRET_KEY=your-32-character-minimum-secret
   ```

4. **Restrict CORS origins**:
   ```
   CORS_ORIGINS=["https://yourdomain.com"]
   ```

## Data Import

### CSV Format

Create CSV files with these columns:

**items.csv**:
```csv
item_id,title,item_type,description,features,metadata
movie_001,The Matrix,movie,A sci-fi film,"{""genre"":""Sci-Fi""}","{""director"":""Wachowskis""}"
```

**users.csv**:
```csv
user_id,preferences
user_001,"{""favorite_genres"":[""Sci-Fi"",""Action""]}"
```

**interactions.csv**:
```csv
user_id,item_id,interaction_type,value,timestamp
user_001,movie_001,rating,8.5,2024-01-15T10:30:00
```

### Programmatic Import

```python
from recommendation_system.utils.data_loader import CSVDataLoader, DataImporter
from recommendation_system.services.data_store import get_data_store

# Load from CSV
loader = CSVDataLoader(
    items_path="data/items.csv",
    users_path="data/users.csv",
    interactions_path="data/interactions.csv"
)

importer = DataImporter(get_data_store())
stats = importer.import_from_loader(loader)
print(stats)
```

## Project Structure

```
AI-Powered-Recommendation-System/
├── src/recommendation_system/
│   ├── api/
│   │   └── app.py              # FastAPI application (30+ endpoints)
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── connection.py       # Database connection manager
│   │   └── repository.py       # Repository pattern implementation
│   ├── models/
│   │   ├── content_based.py    # TF-IDF content filtering
│   │   ├── collaborative.py    # SVD collaborative filtering
│   │   └── hybrid.py           # Combined recommender
│   ├── middleware/
│   │   ├── cache.py            # Redis caching layer
│   │   └── rate_limiter.py     # Rate limiting middleware
│   ├── services/
│   │   ├── data_store.py       # In-memory data store
│   │   └── recommendation_service.py
│   ├── tasks/
│   │   └── training.py         # Background task manager
│   ├── utils/
│   │   ├── data_loader.py      # Production data loaders
│   │   ├── sample_data.py      # Test data generator
│   │   └── logging.py          # Structured logging
│   ├── config.py               # Configuration management
│   ├── schemas.py              # Pydantic models
│   └── cli.py                  # CLI commands
├── alembic/                    # Database migrations
├── tests/                      # Test suite (70+ tests)
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Full stack deployment
├── prometheus.yml              # Prometheus configuration
└── requirements.txt            # Python dependencies
```

## API Endpoints

### Monitoring
- `GET /health` - Health check with model status
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics
- `GET /stats` - System statistics

### Items
- `POST /items` - Create item
- `POST /items/bulk` - Bulk create items
- `GET /items` - List items
- `GET /items/{item_id}` - Get item
- `PATCH /items/{item_id}` - Update item
- `DELETE /items/{item_id}` - Delete item

### Users
- `POST /users` - Create user
- `GET /users` - List users
- `GET /users/{user_id}` - Get user

### Interactions
- `POST /interactions` - Record interaction
- `POST /interactions/bulk` - Bulk record interactions
- `GET /users/{user_id}/interactions` - Get user interactions

### Recommendations
- `POST /recommendations` - Get recommendations (full control)
- `GET /recommendations/user/{user_id}` - User recommendations
- `GET /recommendations/item/{item_id}/similar` - Similar items

### Training
- `POST /train` - Train models
- `GET /tasks/{task_id}` - Get task status
- `GET /tasks` - List tasks
- `POST /models/save` - Save models
- `POST /models/load` - Load models

### Admin
- `POST /cache/clear` - Clear cache
- `GET /cache/stats` - Cache statistics
- `POST /data/import` - Import data

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src/recommendation_system --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with verbose output
pytest -v -s
```

## Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# View migration history
alembic history
```

## Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes:
- `recommendation_requests_total` - Request counts by endpoint/status
- `recommendation_request_latency_seconds` - Request latency histogram
- `model_training_total` - Training job counts

### Health Checks

- `/health/live` - Returns 200 if the service is running
- `/health/ready` - Returns 200 if models are trained and ready
- `/health` - Detailed health status with uptime and model info

## Performance Tips

1. **Use PostgreSQL** for production with proper connection pooling
2. **Enable Redis** caching for frequently requested recommendations
3. **Batch Operations**: Use bulk endpoints for large imports
4. **Async Training**: Use `async_mode=true` for large datasets
5. **Model Persistence**: Enable `MODEL_AUTO_SAVE` to preserve trained models

## License

MIT
