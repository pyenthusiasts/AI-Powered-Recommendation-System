"""Command-line interface for the recommendation system."""

import argparse
import sys

import uvicorn

from recommendation_system.config import get_settings


def train():
    """Train recommendation models from command line."""
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--model-type",
        choices=["content_based", "collaborative", "hybrid"],
        default="hybrid",
        help="Type of model to train",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for trained models",
    )

    args = parser.parse_args()

    from recommendation_system.services.recommendation_service import get_recommendation_service
    from recommendation_system.schemas import RecommendationStrategy

    service = get_recommendation_service()

    strategy_map = {
        "content_based": RecommendationStrategy.CONTENT_BASED,
        "collaborative": RecommendationStrategy.COLLABORATIVE,
        "hybrid": RecommendationStrategy.HYBRID,
    }

    print(f"Training {args.model_type} model...")

    try:
        result = service.train_models(model_type=strategy_map[args.model_type])
        print("Training completed successfully!")
        print(f"Results: {result}")

        if args.output:
            from pathlib import Path
            service.save_models(Path(args.output))
            print(f"Models saved to {args.output}")

    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


def serve():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="Start the recommendation API server")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    settings = get_settings()

    host = args.host or settings.host
    port = args.port or settings.port
    workers = args.workers or settings.workers

    print(f"Starting recommendation API server on {host}:{port}")

    uvicorn.run(
        "recommendation_system.api.app:app",
        host=host,
        port=port,
        workers=1 if args.reload else workers,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    serve()
