"""Runtime configuration. Values come from the environment (`.env` in dev).

No secrets are hard-coded. Missing keys degrade gracefully:
- `GEMINI_API_KEY` unset  -> deterministic regex extractor is used.
- `SD_CONTROLNET_URL` unset -> SD client returns the control image as a stub.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    cors_origins: str = "http://localhost:3000"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    sd_controlnet_url: str = ""
    sd_timeout_seconds: int = 180

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
