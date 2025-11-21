"""FastAPI application for the recommendation system."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from recommendation_system import __version__
from recommendation_system.config import Settings, get_settings
from recommendation_system.schemas import (
    HealthResponse,
    Interaction,
    InteractionCreate,
    Item,
    ItemCreate,
    ItemType,
    ItemUpdate,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationStrategy,
    StatsResponse,
    TrainingRequest,
    TrainingStatus,
    User,
    UserCreate,
)
from recommendation_system.services.data_store import DataStore, get_data_store
from recommendation_system.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)

logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "recommendation_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

# Track startup time
_startup_time: datetime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _startup_time
    _startup_time = datetime.utcnow()

    # Try to load existing models
    service = get_recommendation_service()
    settings = get_settings()
    try:
        service.load_models(settings.model_path)
    except Exception as e:
        logger.info("No existing models to load", reason=str(e))

    logger.info("Application started", version=__version__)
    yield
    logger.info("Application shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="AI-Powered Recommendation System",
        description="Content-based and collaborative filtering recommendations for movies, books, songs, courses, and more.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    _setup_routes(app, settings)

    return app


def _setup_routes(app: FastAPI, settings: Settings):
    """Set up all API routes."""

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(
        api_key: str | None = Security(api_key_header),
    ) -> bool:
        """Verify API key if enabled."""
        if not settings.api_key_enabled:
            return True
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True

    # ============== Health & Monitoring ==============

    @app.get("/health", response_model=HealthResponse, tags=["monitoring"])
    async def health_check():
        """Health check endpoint."""
        service = get_recommendation_service()
        uptime = (datetime.utcnow() - _startup_time).total_seconds() if _startup_time else 0

        return HealthResponse(
            status="healthy",
            version=__version__,
            uptime_seconds=uptime,
            models_loaded={
                "content_based": service.recommender._is_content_trained,
                "collaborative": service.recommender._is_collaborative_trained,
            },
        )

    @app.get("/metrics", tags=["monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type="text/plain")

    @app.get("/stats", response_model=StatsResponse, tags=["monitoring"])
    async def get_stats(
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Get system statistics."""
        stats = data_store.get_stats()
        stats["model_metrics"] = service.recommender.get_training_info()
        return StatsResponse(**stats)

    # ============== Items ==============

    @app.post("/items", response_model=Item, status_code=201, tags=["items"])
    async def create_item(
        item: ItemCreate,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Create a new item."""
        try:
            return data_store.add_item(item)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/items/{item_id}", response_model=Item, tags=["items"])
    async def get_item(
        item_id: str,
        data_store: DataStore = Depends(get_data_store),
    ):
        """Get an item by ID."""
        item = data_store.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.patch("/items/{item_id}", response_model=Item, tags=["items"])
    async def update_item(
        item_id: str,
        update: ItemUpdate,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Update an existing item."""
        item = data_store.update_item(item_id, update)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.delete("/items/{item_id}", status_code=204, tags=["items"])
    async def delete_item(
        item_id: str,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Delete an item."""
        if not data_store.delete_item(item_id):
            raise HTTPException(status_code=404, detail="Item not found")

    @app.get("/items", response_model=list[Item], tags=["items"])
    async def list_items(
        item_type: ItemType | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        data_store: DataStore = Depends(get_data_store),
    ):
        """List items with optional filtering."""
        return data_store.get_items(item_type=item_type, limit=limit, offset=offset)

    @app.post("/items/bulk", response_model=list[Item], status_code=201, tags=["items"])
    async def create_items_bulk(
        items: list[ItemCreate],
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Create multiple items in bulk."""
        created = []
        errors = []

        for item in items:
            try:
                created.append(data_store.add_item(item))
            except ValueError as e:
                errors.append({"item_id": item.item_id, "error": str(e)})

        if errors:
            logger.warning("Bulk creation had errors", errors=errors)

        return created

    # ============== Users ==============

    @app.post("/users", response_model=User, status_code=201, tags=["users"])
    async def create_user(
        user: UserCreate,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Create a new user."""
        try:
            return data_store.add_user(user)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/users/{user_id}", response_model=User, tags=["users"])
    async def get_user(
        user_id: str,
        data_store: DataStore = Depends(get_data_store),
    ):
        """Get a user by ID."""
        user = data_store.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    # ============== Interactions ==============

    @app.post("/interactions", response_model=Interaction, status_code=201, tags=["interactions"])
    async def create_interaction(
        interaction: InteractionCreate,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Record a user-item interaction."""
        try:
            return data_store.add_interaction(interaction)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/interactions/bulk", response_model=list[Interaction], status_code=201, tags=["interactions"])
    async def create_interactions_bulk(
        interactions: list[InteractionCreate],
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Record multiple interactions in bulk."""
        created = []
        for interaction in interactions:
            try:
                created.append(data_store.add_interaction(interaction))
            except ValueError:
                pass  # Skip invalid interactions
        return created

    @app.get("/users/{user_id}/interactions", response_model=list[Interaction], tags=["interactions"])
    async def get_user_interactions(
        user_id: str,
        data_store: DataStore = Depends(get_data_store),
    ):
        """Get all interactions for a user."""
        return data_store.get_user_interactions(user_id)

    # ============== Recommendations ==============

    @app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
    async def get_recommendations(
        request: RecommendationRequest,
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Get personalized recommendations."""
        start = time.time()

        try:
            response = service.get_recommendations(request)
            REQUEST_COUNT.labels(endpoint="recommendations", status="success").inc()
            return response
        except ValueError as e:
            REQUEST_COUNT.labels(endpoint="recommendations", status="error").inc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            REQUEST_LATENCY.labels(endpoint="recommendations").observe(time.time() - start)

    @app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
    async def get_recommendations_for_user(
        user_id: str,
        n: int = Query(default=10, ge=1, le=100),
        strategy: RecommendationStrategy = Query(default=RecommendationStrategy.HYBRID),
        item_type: ItemType | None = None,
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Get recommendations for a specific user (convenience endpoint)."""
        request = RecommendationRequest(
            user_id=user_id,
            num_recommendations=n,
            strategy=strategy,
            item_type=item_type,
        )
        return service.get_recommendations(request)

    @app.get("/recommendations/item/{item_id}/similar", response_model=RecommendationResponse, tags=["recommendations"])
    async def get_similar_items(
        item_id: str,
        n: int = Query(default=10, ge=1, le=100),
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Get items similar to a given item."""
        if not data_store.get_item(item_id):
            raise HTTPException(status_code=404, detail="Item not found")

        request = RecommendationRequest(
            item_id=item_id,
            num_recommendations=n,
            strategy=RecommendationStrategy.CONTENT_BASED,
        )
        return service.get_recommendations(request)

    # ============== Training ==============

    @app.post("/train", response_model=dict[str, Any], tags=["training"])
    async def train_models(
        request: TrainingRequest | None = None,
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Train recommendation models."""
        request = request or TrainingRequest(model_type=RecommendationStrategy.HYBRID)

        try:
            result = service.train_models(
                model_type=request.model_type,
                item_types=request.item_types,
                hyperparameters=request.hyperparameters,
            )
            return {"status": "success", "result": result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/models/save", tags=["training"])
    async def save_models(
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Save trained models to disk."""
        try:
            service.save_models()
            return {"status": "success", "message": "Models saved"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/models/load", tags=["training"])
    async def load_models(
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Load models from disk."""
        try:
            service.load_models()
            return {"status": "success", "message": "Models loaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Create default application
app = create_app()
