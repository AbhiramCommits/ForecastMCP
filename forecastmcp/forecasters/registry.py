"""Forecaster registry driven by ``configs/models.yaml`` and ``configs/costs.yaml``."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from forecastmcp.config import CONFIGS_DIR, is_offline, read_yaml
from forecastmcp.forecasters.base import BaseForecaster
from forecastmcp.forecasters.baselines import PromoSeasonalNaive, StatsForecaster
from forecastmcp.forecasters.chronos_fm import ChronosForecaster
from forecastmcp.forecasters.finetuned import FinetunedChronosForecaster
from forecastmcp.forecasters.moirai_fm import MoiraiForecaster
from forecastmcp.forecasters.timesfm_fm import TimesFMForecaster

MODELS_PATH = CONFIGS_DIR / "models.yaml"
COSTS_PATH = CONFIGS_DIR / "costs.yaml"

#: Models permitted when ``FORECASTMCP_OFFLINE=1`` (all CPU-cheap).
OFFLINE_ALLOWED = {"seasonal_naive", "autoets", "autotheta", "seasonal_naive_cov", "chronos",
                   "chronos-lora-v1"}


@lru_cache(maxsize=1)
def model_configs() -> dict[str, dict[str, Any]]:
    """Read and cache the model registry file."""
    payload = read_yaml(MODELS_PATH, default=None)
    if not payload or "models" not in payload:
        raise FileNotFoundError(f"model registry not found at {MODELS_PATH}")
    return dict(payload["models"])


@lru_cache(maxsize=1)
def cost_config() -> dict[str, Any]:
    """Read and cache the cost model."""
    return read_yaml(COSTS_PATH, default={}) or {}


def _instantiate(name: str, config: dict[str, Any]) -> BaseForecaster:
    adapter = config.get("adapter", name)
    if adapter == "baselines":
        if name == "seasonal_naive_cov":
            return PromoSeasonalNaive(name, config)
        return StatsForecaster(name, config, kind=name)
    if adapter == "chronos":
        return ChronosForecaster(name, config)
    if adapter == "finetuned":
        return FinetunedChronosForecaster(name, config)
    if adapter == "timesfm":
        return TimesFMForecaster(name, config)
    if adapter == "moirai":
        return MoiraiForecaster(name, config)
    raise ValueError(f"unknown adapter {adapter!r} for model {name!r}")


def get_forecaster(name: str) -> BaseForecaster:
    """Instantiate a registered forecaster by name.

    Raises ``KeyError`` for unknown names and ``RuntimeError`` when the model is
    disabled or blocked by offline mode. Availability itself is *not* checked
    here so callers can report ``available: false`` instead of crashing.
    """
    configs = model_configs()
    if name not in configs:
        raise KeyError(f"unknown model {name!r}; available: {sorted(configs)}")
    config = configs[name]
    if config.get("enabled", True) is False:
        raise RuntimeError(f"model {name!r} is disabled in configs/models.yaml")
    if is_offline() and name not in OFFLINE_ALLOWED:
        raise RuntimeError(f"model {name!r} is not permitted in FORECASTMCP_OFFLINE=1 mode")
    return _instantiate(name, config)


def availability(name: str) -> tuple[bool, str]:
    """Return availability without raising, including offline/disabled reasons."""
    configs = model_configs()
    if name not in configs:
        return False, f"unknown model {name!r}"
    if configs[name].get("enabled", True) is False:
        return False, "disabled in configs/models.yaml"
    if is_offline() and name not in OFFLINE_ALLOWED:
        return False, "not permitted in FORECASTMCP_OFFLINE=1 mode"
    try:
        return get_forecaster(name).available()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def list_forecasters(include_disabled: bool = False) -> list[dict[str, Any]]:
    """Return registry metadata plus live availability for every model."""
    rows = []
    for name, config in model_configs().items():
        enabled = config.get("enabled", True) is not False
        if not enabled and not include_disabled:
            continue
        ok, reason = availability(name)
        rows.append(
            {
                "name": name,
                "adapter": config.get("adapter"),
                "checkpoint": config.get("checkpoint"),
                "revision": config.get("revision"),
                "context_length": config.get("context_length"),
                "device": config.get("device"),
                "cost_per_call_usd": config.get("cost_per_call_usd"),
                "n_params": config.get("n_params"),
                "supports_covariates": bool(config.get("supports_covariates", False)),
                "enabled": enabled,
                "available": ok,
                "reason": reason,
                "training_domain": config.get("training_domain"),
                "description": config.get("description"),
            }
        )
    return rows


def model_card(name: str) -> dict[str, Any]:
    """Build a model card with availability, size, context and training domain."""
    configs = model_configs()
    if name not in configs:
        raise KeyError(f"unknown model {name!r}")
    config = configs[name]
    ok, reason = availability(name)
    card = {
        "name": name,
        "available": ok,
        "reason": reason,
        "adapter": config.get("adapter"),
        "checkpoint": config.get("checkpoint"),
        "revision": config.get("revision"),
        "context_length": config.get("context_length"),
        "n_params": config.get("n_params"),
        "supports_covariates": bool(config.get("supports_covariates", False)),
        "training_domain": config.get("training_domain"),
        "description": config.get("description"),
        "cost_per_call_usd": config.get("cost_per_call_usd"),
        "device": config.get("device"),
    }
    if config.get("adapter") == "finetuned":
        import json
        from pathlib import Path

        from forecastmcp.config import ROOT

        card_path = Path(config.get("checkpoint", ""))
        if not card_path.is_absolute():
            card_path = ROOT / card_path
        card_path = card_path / "model_card.json"
        if card_path.exists():
            card.update({"finetune": json.loads(card_path.read_text())})
    return card


def estimate_cost(name: str, latency_ms: float, device: str = "cpu") -> float:
    """Estimated USD cost for one call: max(config estimate, device-time price)."""
    costs = cost_config()
    configured = float((costs.get("forecasters") or {}).get(name, 0.0))
    per_second = float((costs.get("device_second_usd") or {}).get(device, 0.0))
    measured = per_second * max(latency_ms, 0.0) / 1000.0
    return round(max(configured, measured), 8)


def available_names() -> list[str]:
    """Names of models that are currently usable."""
    return [row["name"] for row in list_forecasters() if row["available"]]
