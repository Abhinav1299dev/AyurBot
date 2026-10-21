"""Central configuration for AyurBot.

Every setting has a safe default. Any setting can be overridden with an
environment variable or a line in a local ``.env`` file (see ``.env.example``).
Real environment variables always win over values in ``.env``.

Invalid values never crash the app: they are reported as warnings and the
default is used instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables are never overwritten. Returns the values
    that were applied. Blank lines and ``#`` comments are ignored.
    """
    path = path or PROJECT_ROOT / ".env"
    applied: dict[str, str] = {}
    if not path.is_file():
        return applied
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return applied
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    kb_path: Path = PROJECT_ROOT / "data" / "herbs.json"
    host: str = "127.0.0.1"
    port: int = 7860
    port_search_range: int = 10
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: Path | None = None
    disable_transformer: bool = False
    offline: bool = False
    device: str = "cpu"
    top_k: int = 3
    min_semantic_score: float = 0.30
    min_lexical_score: float = 0.25
    max_query_chars: int = 1000
    max_history_messages: int = 40
    concurrency_limit: int = 4
    open_browser: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int, lo: int, hi: int, warnings: list[str]) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        warnings.append(f"{name}={raw!r} is not a whole number; using {default}.")
        return default
    if not lo <= value <= hi:
        warnings.append(f"{name}={value} is outside {lo}-{hi}; using {default}.")
        return default
    return value


def _env_float(name: str, default: float, lo: float, hi: float, warnings: list[str]) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        warnings.append(f"{name}={raw!r} is not a number; using {default}.")
        return default
    if not lo <= value <= hi:
        warnings.append(f"{name}={value} is outside {lo}-{hi}; using {default}.")
        return default
    return value


def _env_bool(name: str, default: bool, warnings: list[str]) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    warnings.append(f"{name}={raw!r} is not true/false; using {default}.")
    return default


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw.strip()).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_settings(use_dotenv: bool = True) -> Settings:
    """Build settings from defaults, ``.env`` and environment variables."""
    if use_dotenv:
        load_dotenv()
    warnings: list[str] = []
    defaults = Settings()
    return Settings(
        kb_path=_env_path("AYURBOT_KB_PATH", defaults.kb_path) or defaults.kb_path,
        host=_env_str("AYURBOT_HOST", defaults.host),
        port=_env_int("AYURBOT_PORT", defaults.port, 1024, 65535, warnings),
        port_search_range=_env_int("AYURBOT_PORT_SEARCH_RANGE", defaults.port_search_range, 1, 100, warnings),
        model_name=_env_str("AYURBOT_MODEL_NAME", defaults.model_name),
        model_cache_dir=_env_path("AYURBOT_MODEL_CACHE_DIR", defaults.model_cache_dir),
        disable_transformer=_env_bool("AYURBOT_DISABLE_TRANSFORMER", defaults.disable_transformer, warnings),
        offline=_env_bool("AYURBOT_OFFLINE", defaults.offline, warnings),
        device=_env_str("AYURBOT_DEVICE", defaults.device),
        top_k=_env_int("AYURBOT_TOP_K", defaults.top_k, 1, 10, warnings),
        min_semantic_score=_env_float("AYURBOT_MIN_SEMANTIC_SCORE", defaults.min_semantic_score, 0.0, 1.0, warnings),
        min_lexical_score=_env_float("AYURBOT_MIN_LEXICAL_SCORE", defaults.min_lexical_score, 0.0, 1.0, warnings),
        max_query_chars=_env_int("AYURBOT_MAX_QUERY_CHARS", defaults.max_query_chars, 50, 10000, warnings),
        max_history_messages=_env_int("AYURBOT_MAX_HISTORY_MESSAGES", defaults.max_history_messages, 2, 500, warnings),
        concurrency_limit=_env_int("AYURBOT_CONCURRENCY_LIMIT", defaults.concurrency_limit, 1, 32, warnings),
        open_browser=_env_bool("AYURBOT_OPEN_BROWSER", defaults.open_browser, warnings),
        warnings=tuple(warnings),
    )
