from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # Profiles: offline -> dependency-light deterministic stack; max -> transformer ensemble.
    use_transformers: bool = _env_bool("AEGIS_USE_TRANSFORMERS", False)
    max_profile: bool = _env_bool("AEGIS_MAX_PROFILE", False)
    sentiment_models: tuple[str, ...] = _env_list(
        "AEGIS_SENTIMENT_MODELS",
        "ProsusAI/finbert,yiyanghkust/finbert-tone,soleimanian/financial-roberta-large-sentiment",
    )
    event_model: str = os.getenv(
        "AEGIS_EVENT_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )
    model_device: int = int(os.getenv("AEGIS_MODEL_DEVICE", "-1"))
    request_timeout: float = float(os.getenv("AEGIS_REQUEST_TIMEOUT", "15"))
    user_agent: str = os.getenv(
        "AEGIS_USER_AGENT", "AegisSignal-X/1.0 research-prototype contact=demo@example.com"
    )
    reddit_subreddits: str = os.getenv("AEGIS_REDDIT_SUBREDDITS", "stocks+investing+wallstreetbets")
    news_lookback: str = os.getenv("AEGIS_NEWS_LOOKBACK", "6h")
    gdelt_max_records: int = int(os.getenv("AEGIS_GDELT_MAX_RECORDS", "75"))
    risk_alert_threshold: float = float(os.getenv("AEGIS_ALERT_THRESHOLD", "7.0"))
    sqlite_path: str = os.getenv("AEGIS_SQLITE_PATH", "outputs/aegis.db")
    impact_model_path: str = os.getenv("AEGIS_IMPACT_MODEL_PATH", "")
    impact_tail_blend: float = float(os.getenv("AEGIS_IMPACT_TAIL_BLEND", "0.45"))
    market_benchmark: str = os.getenv("AEGIS_MARKET_BENCHMARK", "SPY")
    market_intraday_interval: str = os.getenv("AEGIS_MARKET_INTERVAL", "5m")
    market_intraday_range: str = os.getenv("AEGIS_MARKET_RANGE", "5d")
    hawkes_half_life_minutes: float = float(os.getenv("AEGIS_HAWKES_HALF_LIFE_MIN", "45"))
    graph_diffusion: float = float(os.getenv("AEGIS_GRAPH_DIFFUSION", "0.30"))
    source_min_quality: float = float(os.getenv("AEGIS_SOURCE_MIN_QUALITY", "0.20"))
    random_seed: int = int(os.getenv("AEGIS_RANDOM_SEED", "42"))


SETTINGS = Settings()
