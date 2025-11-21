"""Application configuration using Pydantic Settings."""

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="recommendation-system")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)

    # Database
    database_url: str = Field(
        default="sqlite:///./recommendation_system.db",
        description="Database connection URL (PostgreSQL, MySQL, or SQLite)",
    )
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    database_pool_timeout: int = Field(default=30)

    # Redis (caching and rate limiting)
    redis_url: str | None = Field(default=None)
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_enabled: bool = Field(default=True)

    # Paths
    model_path: Path = Field(default=Path("./models"))
    data_path: Path = Field(default=Path("./data"))
    log_path: Path = Field(default=Path("./logs"))

    # API Security
    api_key_enabled: bool = Field(default=True)
    api_key: str = Field(default="")
    api_key_header: str = Field(default="X-API-Key")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=True)

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Window in seconds")
    rate_limit_burst: int = Field(default=20, description="Burst limit")

    # Model Training
    training_batch_size: int = Field(default=1000)
    training_async_enabled: bool = Field(default=True)
    model_auto_save: bool = Field(default=True)
    model_version_limit: int = Field(default=5, description="Max model versions to keep")

    # Recommendation Settings
    default_num_recommendations: int = Field(default=10)
    max_recommendations: int = Field(default=100)
    cold_start_threshold: int = Field(default=5)
    content_weight: float = Field(default=0.4)
    collaborative_weight: float = Field(default=0.6)

    # Feature Flags
    enable_recommendation_logging: bool = Field(default=True)
    enable_ab_testing: bool = Field(default=False)
    enable_metrics: bool = Field(default=True)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str, info) -> str:
        """Ensure API key is set in production."""
        # This is called during validation, can't access other fields easily
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def effective_cors_origins(self) -> list[str]:
        """Get CORS origins based on environment."""
        if self.is_production:
            return [o for o in self.cors_origins if o != "*"]
        return self.cors_origins

    def validate_production_settings(self) -> list[str]:
        """Validate settings for production readiness."""
        errors = []

        if self.is_production:
            if not self.api_key or len(self.api_key) < 32:
                errors.append("API_KEY must be at least 32 characters in production")

            if self.debug:
                errors.append("DEBUG must be False in production")

            if "*" in self.cors_origins:
                errors.append("CORS_ORIGINS should not contain '*' in production")

            if self.database_url.startswith("sqlite"):
                errors.append("SQLite is not recommended for production; use PostgreSQL")

            if not self.redis_url:
                errors.append("REDIS_URL should be set for production caching")

        return errors


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache():
    """Clear the settings cache (for testing)."""
    get_settings.cache_clear()
