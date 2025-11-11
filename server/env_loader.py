from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _candidate_env_paths(explicit_path: Optional[Path]) -> Iterable[Path]:
    """Yield possible .env locations in priority order."""
    if explicit_path:
        yield explicit_path

    script_dir = Path(__file__).resolve().parent
    yield script_dir / ".env"
    yield script_dir.parent / ".env"


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Populate os.environ with values read from .env files, without overriding existing keys."""
    for path in _candidate_env_paths(env_path):
        if not path or not path.is_file():
            continue

        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def get_env_str(key: str, default: str) -> str:
    """Return a string env var with a default fallback."""
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """Return an integer env var with validation."""
    value = os.getenv(key)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer, got '{value}'") from exc
