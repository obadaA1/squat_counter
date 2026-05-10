from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_root: Path
    cors_origins: tuple[str, ...]
    max_video_bytes: int
    environment: str


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings(
        model_root=Path(os.getenv("SQUAT_MODEL_ROOT", "/models/squat/current")),
        cors_origins=_csv_env(
            "SQUAT_CORS_ORIGINS",
            "https://obadaalsehli.com,http://localhost:3000,http://127.0.0.1:3000",
        ),
        max_video_bytes=int(os.getenv("SQUAT_MAX_VIDEO_BYTES", str(50 * 1024 * 1024))),
        environment=os.getenv("SQUAT_ENV", "production"),
    )

