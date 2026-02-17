from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    api_secret: str

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
