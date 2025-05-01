"""Configuration management for the therapeutic agent."""

from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "Virtual Therapist"
    debug: bool = False
    database_url: str = "sqlite:///./therapy.db"
    
    class Config:
        env_file = ".env"

settings = Settings()
