"""Application configuration management."""

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1, le=300)


class RedisConfig(BaseModel):
    """Redis configuration settings."""

    url: str = Field(..., description="Redis connection URL")
    max_connections: int = Field(default=10, ge=1, le=100)
    socket_timeout: int = Field(default=5, ge=1, le=60)


class SecurityConfig(BaseModel):
    """Security-related configuration."""

    secret_key: str = Field(..., min_length=32)
    session_timeout_minutes: int = Field(default=60, ge=5, le=480)
    max_sessions_per_user: int = Field(default=5, ge=1, le=20)
    rate_limit_per_minute: int = Field(default=10, ge=1, le=100)


class TherapyConfig(BaseModel):
    """Therapy-specific configuration."""

    safety_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_conversation_length: int = Field(default=50, ge=5, le=200)
    session_summary_interval: int = Field(default=10, ge=5, le=30)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "test_key"),
        min_length=1,
    )

    database: DatabaseConfig = Field(default=None)  # type: ignore[assignment]
    redis: RedisConfig = Field(default=None)  # type: ignore[assignment]
    security: SecurityConfig = Field(default=None)  # type: ignore[assignment]
    therapy: TherapyConfig = Field(default=None)  # type: ignore[assignment]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @field_validator("database", mode="before")
    @classmethod
    def parse_database_config(cls, v: Any) -> DatabaseConfig:
        if v is None:
            database_url = os.getenv(
                "DATABASE_URL", "sqlite+aiosqlite:///:memory:"
            )  # Default to in-memory SQLite for testing
            return DatabaseConfig(url=database_url)
        if isinstance(v, dict):
            return DatabaseConfig(**v)
        return v

    @field_validator("redis", mode="before")
    @classmethod
    def parse_redis_config(cls, v: Any) -> RedisConfig:
        if v is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            return RedisConfig(url=redis_url)
        if isinstance(v, dict):
            return RedisConfig(**v)
        return v

    @field_validator("security", mode="before")
    @classmethod
    def parse_security_config(cls, v: Any) -> SecurityConfig:
        if v is None:
            secret_key = os.getenv(
                "SECRET_KEY",
                "test_secret_key_for_development_only_replace_in_production",
            )
            if len(secret_key) < 32:
                secret_key = secret_key.ljust(32, "x")  # Pad for testing
            return SecurityConfig(
                secret_key=secret_key,
                session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
                max_sessions_per_user=int(os.getenv("MAX_SESSIONS_PER_USER", "5")),
                rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
            )
        if isinstance(v, dict):
            return SecurityConfig(**v)
        return v

    @field_validator("therapy", mode="before")
    @classmethod
    def parse_therapy_config(cls, v: Any) -> TherapyConfig:
        if v is None:
            return TherapyConfig(
                safety_threshold=float(os.getenv("SAFETY_THRESHOLD", "0.8")),
                max_conversation_length=int(os.getenv("MAX_CONVERSATION_LENGTH", "50")),
                session_summary_interval=int(
                    os.getenv("SESSION_SUMMARY_INTERVAL", "10")
                ),
            )
        if isinstance(v, dict):
            return TherapyConfig(**v)
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    # Type ignore needed as validators will parse these fields from environment
    return Settings()  # type: ignore[call-arg]
