"""Configuration management for EvalX CLI."""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """CLI settings loaded from environment variables."""

    evalx_service_url: str = "http://localhost:8000"
    evalx_bearer_token: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


def get_settings() -> Settings:
    """Get CLI settings."""
    return Settings()
