"""FastAPI application for the recommendation system - Production Ready."""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from recommendation_system import __version__
from recommendation_system.config import Settings, get_settings
from recommendation_system.middleware.cache import CacheManager, get_cache_manager
from recommendation_system.middleware.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitMiddleware,
)
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
from recommendation_system.tasks import (
    BackgroundTaskManager,
    TaskStatus,
    get_task_manager,
)

logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "recommendation_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
ACTIVE_REQUESTS = Counter(
    "recommendation_active_requests",
    "Currently active requests",
)
MODEL_TRAINING_COUNT = Counter(
    "model_training_total",
    "Total model training runs",
    ["model_type", "status"],
)

# Track startup time
_startup_time: datetime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _startup_time
    _startup_time = datetime.utcnow()

    settings = get_settings()

    # Validate production settings
    if settings.is_production:
        errors = settings.validate_production_settings()
        if errors:
            for error in errors:
                logger.warning("Production config issue", issue=error)

    # Initialize database if using persistent storage
    try:
        from recommendation_system.database import init_database
        init_database(settings.database_url, create_tables=True)
        logger.info("Database initialized")
    except ImportError:
        logger.info("Using in-memory data store")
    except Exception as e:
        logger.warning("Database initialization failed, using in-memory store", error=str(e))

    # Try to load existing models
    service = get_recommendation_service()
    try:
        service.load_models(settings.model_path)
        logger.info("Models loaded from disk")
    except Exception as e:
        logger.info("No existing models to load", reason=str(e))

    # Initialize cache
    cache = get_cache_manager()
    logger.info("Cache initialized", type="redis" if cache.is_redis_connected else "memory")

    logger.info(
        "Application started",
        version=__version__,
        environment=settings.app_env,
        debug=settings.debug,
    )

    yield

    # Cleanup
    task_manager = get_task_manager()
    task_manager.shutdown(wait=True)
    logger.info("Application shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="AI-Powered Recommendation System",
        description="""
A production-ready recommendation engine supporting:
- **Content-Based Filtering**: TF-IDF and cosine similarity
- **Collaborative Filtering**: SVD matrix factorization
- **Hybrid Recommendations**: Intelligent combination of both approaches

Supports movies, books, songs, courses, and generic products.
        """,
        version=__version__,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.effective_cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    if settings.rate_limit_enabled:
        rate_limiter = RateLimiter(
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window,
            burst_limit=settings.rate_limit_burst,
            redis_url=settings.redis_url,
        )
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter,
            exclude_paths=["/health", "/metrics", "/docs", "/openapi.json", "/redoc"],
        )

    # Exception handlers
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Set up routes
    _setup_routes(app, settings)

    return app


def _setup_routes(app: FastAPI, settings: Settings):
    """Set up all API routes."""

    api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

    async def verify_api_key(
        request: Request,
        api_key: str | None = Security(api_key_header),
    ) -> bool:
        """Verify API key if enabled."""
        if not settings.api_key_enabled:
            return True

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if api_key != settings.api_key:
            logger.warning("Invalid API key attempt", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return True

    # ============== Health & Monitoring ==============

    @app.get("/health", response_model=HealthResponse, tags=["monitoring"])
    async def health_check():
        """Health check endpoint for load balancers and orchestrators."""
        service = get_recommendation_service()
        cache = get_cache_manager()
        uptime = (datetime.utcnow() - _startup_time).total_seconds() if _startup_time else 0

        # Check database health
        db_healthy = True
        try:
            from recommendation_system.database import get_db_manager
            db_healthy = get_db_manager().health_check()
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if db_healthy else "degraded",
            version=__version__,
            uptime_seconds=uptime,
            models_loaded={
                "content_based": service.recommender._is_content_trained,
                "collaborative": service.recommender._is_collaborative_trained,
            },
        )

    @app.get("/health/ready", tags=["monitoring"])
    async def readiness_check():
        """Readiness probe - checks if app is ready to serve traffic."""
        service = get_recommendation_service()
        is_ready = (
            service.recommender._is_content_trained
            or service.recommender._is_collaborative_trained
        )

        if not is_ready:
            # Still ready, just no models trained yet
            return {"status": "ready", "models_trained": False}

        return {"status": "ready", "models_trained": True}

    @app.get("/health/live", tags=["monitoring"])
    async def liveness_check():
        """Liveness probe - checks if app is alive."""
        return {"status": "alive"}

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
        """Get comprehensive system statistics."""
        cache = get_cache_manager()
        task_manager = get_task_manager()

        stats = data_store.get_stats()
        stats["model_metrics"] = service.recommender.get_training_info()
        stats["cache_stats"] = cache.get_stats()
        stats["task_stats"] = task_manager.get_stats()

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
            created = data_store.add_item(item)
            REQUEST_COUNT.labels(endpoint="items", method="POST", status="success").inc()

            # Invalidate related caches
            cache = get_cache_manager()
            cache.delete_pattern("rec:recs:*")

            return created
        except ValueError as e:
            REQUEST_COUNT.labels(endpoint="items", method="POST", status="conflict").inc()
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

        # Invalidate caches
        cache = get_cache_manager()
        cache.invalidate_item_cache(item_id)

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

        # Invalidate caches
        cache = get_cache_manager()
        cache.invalidate_item_cache(item_id)

    @app.get("/items", response_model=list[Item], tags=["items"])
    async def list_items(
        item_type: ItemType | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        data_store: DataStore = Depends(get_data_store),
    ):
        """List items with optional filtering and pagination."""
        return data_store.get_items(item_type=item_type, limit=limit, offset=offset)

    @app.post("/items/bulk", response_model=dict[str, Any], status_code=201, tags=["items"])
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
                created.append(data_store.add_item(item).item_id)
            except ValueError as e:
                errors.append({"item_id": item.item_id, "error": str(e)})

        if errors:
            logger.warning("Bulk creation had errors", error_count=len(errors))

        # Invalidate caches
        cache = get_cache_manager()
        cache.delete_pattern("rec:recs:*")

        return {
            "created": len(created),
            "errors": len(errors),
            "created_ids": created,
            "error_details": errors if errors else None,
        }

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

    @app.get("/users", tags=["users"])
    async def list_users(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """List users with pagination."""
        users = data_store.get_all_users()
        user_list = list(users.values())[offset : offset + limit]
        return {"users": user_list, "total": len(users)}

    # ============== Interactions ==============

    @app.post("/interactions", response_model=Interaction, status_code=201, tags=["interactions"])
    async def create_interaction(
        interaction: InteractionCreate,
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Record a user-item interaction."""
        try:
            created = data_store.add_interaction(interaction)

            # Invalidate user's cached recommendations
            cache = get_cache_manager()
            cache.invalidate_user_cache(interaction.user_id)

            return created
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/interactions/bulk", response_model=dict[str, Any], status_code=201, tags=["interactions"])
    async def create_interactions_bulk(
        interactions: list[InteractionCreate],
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Record multiple interactions in bulk."""
        created = 0
        errors = 0
        user_ids = set()

        for interaction in interactions:
            try:
                data_store.add_interaction(interaction)
                created += 1
                user_ids.add(interaction.user_id)
            except ValueError:
                errors += 1

        # Invalidate caches for affected users
        cache = get_cache_manager()
        for user_id in user_ids:
            cache.invalidate_user_cache(user_id)

        return {"created": created, "errors": errors}

    @app.get("/users/{user_id}/interactions", response_model=list[Interaction], tags=["interactions"])
    async def get_user_interactions(
        user_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Get all interactions for a user."""
        interactions = data_store.get_user_interactions(user_id)
        return interactions[:limit]

    # ============== Recommendations ==============

    @app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
    async def get_recommendations(
        request: RecommendationRequest,
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Get personalized recommendations with full control over parameters."""
        start = time.time()

        # Check cache first
        cache = get_cache_manager()
        settings = get_settings()

        if settings.cache_enabled:
            cached = cache.get_cached_recommendations(
                user_id=request.user_id,
                item_id=request.item_id,
                strategy=request.strategy.value,
                num=request.num_recommendations,
            )
            if cached:
                REQUEST_COUNT.labels(endpoint="recommendations", method="POST", status="cache_hit").inc()
                return RecommendationResponse(**cached)

        try:
            response = service.get_recommendations(request)
            REQUEST_COUNT.labels(endpoint="recommendations", method="POST", status="success").inc()

            # Cache the result
            if settings.cache_enabled:
                cache.cache_recommendations(
                    user_id=request.user_id,
                    item_id=request.item_id,
                    strategy=request.strategy.value,
                    num=request.num_recommendations,
                    recommendations=response.model_dump(),
                )

            return response
        except ValueError as e:
            REQUEST_COUNT.labels(endpoint="recommendations", method="POST", status="error").inc()
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
        async_mode: bool = Query(default=False, description="Run training asynchronously"),
        _: bool = Depends(verify_api_key),
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        """Train recommendation models."""
        request = request or TrainingRequest(model_type=RecommendationStrategy.HYBRID)
        settings = get_settings()

        def do_training():
            return service.train_models(
                model_type=request.model_type,
                item_types=request.item_types,
                hyperparameters=request.hyperparameters,
            )

        if async_mode and settings.training_async_enabled:
            # Submit as background task
            task_manager = get_task_manager()
            task = task_manager.submit_task(
                "model_training",
                do_training,
                metadata={"model_type": request.model_type.value},
            )
            MODEL_TRAINING_COUNT.labels(model_type=request.model_type.value, status="submitted").inc()

            return {
                "status": "submitted",
                "task_id": task.task_id,
                "message": "Training started in background. Use /tasks/{task_id} to check status.",
            }
        else:
            try:
                result = do_training()

                # Auto-save models
                if settings.model_auto_save:
                    service.save_models()

                # Invalidate all recommendation caches
                cache = get_cache_manager()
                cache.clear()

                MODEL_TRAINING_COUNT.labels(model_type=request.model_type.value, status="success").inc()
                return {"status": "success", "result": result}
            except ValueError as e:
                MODEL_TRAINING_COUNT.labels(model_type=request.model_type.value, status="failed").inc()
                raise HTTPException(status_code=400, detail=str(e))

    @app.get("/tasks/{task_id}", tags=["training"])
    async def get_task_status(
        task_id: str,
        _: bool = Depends(verify_api_key),
    ):
        """Get status of a background task."""
        task_manager = get_task_manager()
        status = task_manager.get_task_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        return status

    @app.get("/tasks", tags=["training"])
    async def list_tasks(
        task_type: str | None = None,
        status: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        _: bool = Depends(verify_api_key),
    ):
        """List background tasks."""
        task_manager = get_task_manager()

        task_status = None
        if status:
            try:
                task_status = TaskStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        tasks = task_manager.list_tasks(task_type=task_type, status=task_status, limit=limit)
        return {"tasks": [t.to_dict() for t in tasks]}

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

    # ============== Cache Management ==============

    @app.post("/cache/clear", tags=["admin"])
    async def clear_cache(
        _: bool = Depends(verify_api_key),
    ):
        """Clear all cached recommendations."""
        cache = get_cache_manager()
        cache.clear()
        return {"status": "success", "message": "Cache cleared"}

    @app.get("/cache/stats", tags=["admin"])
    async def cache_stats(
        _: bool = Depends(verify_api_key),
    ):
        """Get cache statistics."""
        cache = get_cache_manager()
        return cache.get_stats()

    # ============== Data Import ==============

    @app.post("/data/import", tags=["admin"])
    async def import_data(
        items_path: str | None = Query(default=None),
        users_path: str | None = Query(default=None),
        interactions_path: str | None = Query(default=None),
        format: str = Query(default="csv", description="File format: csv or json"),
        _: bool = Depends(verify_api_key),
        data_store: DataStore = Depends(get_data_store),
    ):
        """Import data from external files."""
        from recommendation_system.utils.data_loader import DataImporter

        importer = DataImporter(data_store)

        try:
            if format == "csv":
                result = importer.import_from_csv(
                    items_path=items_path,
                    users_path=users_path,
                    interactions_path=interactions_path,
                )
            elif format == "json":
                result = importer.import_from_json(
                    items_path=items_path,
                    users_path=users_path,
                    interactions_path=interactions_path,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Create default application
app = create_app()
