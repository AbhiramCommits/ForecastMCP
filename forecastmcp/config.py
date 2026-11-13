"""Central configuration for ForecastMCP.

All paths are derived from the repository root (or ``FORECASTMCP_HOME`` when set),
so the CLI behaves identically from any working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FORECASTMCP_HOME", Path(__file__).resolve().parents[1])).resolve()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = Path(os.environ.get("FORECASTMCP_DB", DATA_DIR / "forecastmcp.duckdb"))

CONFIGS_DIR = ROOT / "configs"
ARTIFACTS_DIR = Path(os.environ.get("FORECASTMCP_ARTIFACTS", ROOT / "artifacts"))
RESULTS_DIR = ROOT / "results"
FINETUNE_DIR = ARTIFACTS_DIR / "finetune"
RUNS_DIR = ARTIFACTS_DIR / "runs"

DEFAULT_SEED = int(os.environ.get("FORECASTMCP_SEED", "42"))

#: Deterministic synthetic generator size (200 SKUs x daily history).
SYNTHETIC_N_SERIES = int(os.environ.get("FORECASTMCP_SYNTHETIC_SERIES", "200"))

#: Default rolling-origin holdout geometry. Frozen in configs/holdout.yaml.
DEFAULT_HORIZON = 28
DEFAULT_N_WINDOWS = 6


def is_offline() -> bool:
    """Return True when ``FORECASTMCP_OFFLINE=1`` is set.

    Offline mode forces the deterministic synthetic generator for data loading and
    restricts the registry to CPU-cheap models.
    """
    return os.environ.get("FORECASTMCP_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}


def ensure_dirs() -> None:
    """Create the writable directories the pipeline needs."""
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, RESULTS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path | str, default: Any = None) -> Any:
    """Read a YAML file, returning ``default`` when it does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(path: Path | str, payload: Any) -> None:
    """Write a YAML file atomically-ish (single write, deterministic key order)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata about a supported dataset."""

    name: str
    kind: str  # "synthetic" | "gluonts"
    freq: str
    horizon: int = DEFAULT_HORIZON
    n_windows: int = DEFAULT_N_WINDOWS
    max_series: int | None = None
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "synthetic_supply": DatasetSpec(
        name="synthetic_supply",
        kind="synthetic",
        freq="D",
        horizon=28,
        n_windows=6,
        max_series=SYNTHETIC_N_SERIES,
        description="200 seeded SKU-level daily demand series with promo spikes and lead time.",
    ),
    "m4_daily": DatasetSpec(
        name="m4_daily",
        kind="gluonts",
        freq="D",
        horizon=28,
        n_windows=6,
        max_series=int(os.environ.get("FORECASTMCP_M4_SERIES", "50")),
        description="M4 daily competition series (deterministic head subset).",
    ),
    "electricity": DatasetSpec(
        name="electricity",
        kind="gluonts",
        freq="h",
        horizon=48,
        n_windows=6,
        max_series=int(os.environ.get("FORECASTMCP_ELEC_SERIES", "50")),
        description="Hourly electricity load (capacity signal), deterministic head subset.",
    ),
    "traffic": DatasetSpec(
        name="traffic",
        kind="gluonts",
        freq="h",
        horizon=48,
        n_windows=6,
        max_series=int(os.environ.get("FORECASTMCP_TRAFFIC_SERIES", "50")),
        description="Hourly road occupancy (capacity signal), deterministic head subset.",
    ),
}


def dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset specification, raising a helpful error when unknown."""
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASET_SPECS)}") from exc
