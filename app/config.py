"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    waqi_api_token: str = "demo"
    owm_api_token: str = ""
    jwt_secret: str = "change_me_to_a_long_random_string_at_least_32_characters"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080  # 7 days
    cors_origins: str = "*"

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()