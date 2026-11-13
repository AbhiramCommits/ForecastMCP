"""Dataset loading and calendar feature derivation.

Long-format frames always expose ``dataset, series_id, timestamp, target`` plus
covariate columns. The synthetic generator is the only code path required for
offline/CI runs; GluonTS datasets are downloaded once into ``data/raw``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from forecastmcp.config import (
    PROCESSED_DIR,
    RAW_DIR,
    dataset_spec,
    ensure_dirs,
    is_offline,
)
from forecastmcp.data.synthetic import generate_synthetic_supply


class DatasetUnavailable(RuntimeError):
    """Raised when a requested dataset cannot be downloaded or decoded."""


def add_calendar_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Append deterministic calendar covariates to a long frame.

    Daily data gets day-of-week/month, weekend flag, annual and weekly Fourier
    terms. Hourly data additionally gets ``hour`` and a daily Fourier pair.
    The frame must contain a ``timestamp`` column.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["timestamp"] = ts
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    annual_angle = 2 * np.pi * ts.dt.dayofyear / 365.25
    out["sin_annual_1"] = np.sin(annual_angle)
    out["cos_annual_1"] = np.cos(annual_angle)
    out["sin_annual_2"] = np.sin(2 * annual_angle)
    out["cos_annual_2"] = np.cos(2 * annual_angle)
    weekly_angle = 2 * np.pi * ts.dt.dayofweek / 7.0
    out["sin_week_1"] = np.sin(weekly_angle)
    out["cos_week_1"] = np.cos(weekly_angle)

    if freq.lower().startswith(("h", "t", "min")):
        out["hour"] = ts.dt.hour
        daily_angle = 2 * np.pi * ts.dt.hour / 24.0
        out["sin_day_1"] = np.sin(daily_angle)
        out["cos_day_1"] = np.cos(daily_angle)

    if "promo" not in out.columns:
        out["promo"] = 0.0
    return out


def _cache_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.parquet"


def _from_gluonts(name: str, max_series: int | None) -> pd.DataFrame:
    """Materialise the deterministic head of a GluonTS dataset."""
    try:
        from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
    except ImportError as exc:  # pragma: no cover - gluonts is a core dependency
        raise DatasetUnavailable(f"gluonts not importable: {exc}") from exc

    try:
        dataset = gluonts_get_dataset(name, path=RAW_DIR / "gluonts")
    except Exception as exc:  # network / disk / upstream failures are recorded verbatim
        raise DatasetUnavailable(f"get_dataset({name!r}) failed: {type(exc).__name__}: {exc}") from exc

    spec = dataset_spec(name)
    frames: list[pd.DataFrame] = []
    for i, item in enumerate(dataset.train):
        if max_series is not None and i >= max_series:
            break
        target = np.asarray(item["target"], dtype=float)
        start = item["start"]
        try:
            index = pd.period_range(start=start, periods=len(target), freq=start.freq)
            timestamps = index.to_timestamp()
        except Exception:
            timestamps = pd.date_range(pd.Timestamp(start.start_time), periods=len(target),
                                       freq=spec.freq)
        series_id = str(item.get("item_id", f"{name}-{i:04d}"))
        frames.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "timestamp": timestamps,
                    "target": target,
                    "promo": 0.0,
                }
            )
        )
    if not frames:
        raise DatasetUnavailable(f"GluonTS dataset {name!r} produced no training series")
    df = pd.concat(frames, ignore_index=True)
    df["series_id"] = pd.Categorical(df["series_id"], categories=list(dict.fromkeys(df["series_id"])))
    df["series_id"] = df["series_id"].astype(str)
    return df


def load_dataset(
    name: str,
    max_series: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load a supported dataset as a long-format frame.

    Parameters
    ----------
    name:
        One of ``synthetic_supply``, ``m4_daily``, ``electricity``, ``traffic``.
    max_series:
        Optional cap on the number of series (deterministic head subset).
    use_cache:
        Read/write the processed parquet cache under ``data/processed``.

    In offline mode every request resolves to the deterministic synthetic panel.
    """
    ensure_dirs()
    if is_offline() and name != "synthetic_supply":
        warnings.warn(
            f"FORECASTMCP_OFFLINE=1: using synthetic_supply instead of {name!r}",
            stacklevel=2,
        )
        name = "synthetic_supply"

    spec = dataset_spec(name)
    max_series = spec.max_series if max_series is None else max_series
    cache = _cache_path(name)
    cache_ok = cache.exists() and (max_series is None or spec.max_series == max_series)

    if use_cache and cache_ok:
        df = pd.read_parquet(cache)
        df["dataset"] = name
        return df

    if spec.kind == "synthetic":
        df = generate_synthetic_supply(n_series=max_series or 200)
    else:
        df = _from_gluonts(name, max_series)

    df = add_calendar_features(df, spec.freq)
    df["dataset"] = name
    ordered = ["dataset", "series_id", "timestamp", "target"]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    if use_cache:
        df.to_parquet(cache, index=False)
    return df


def infer_freq(df: pd.DataFrame) -> str:
    """Infer a pandas frequency alias from one series' timestamps."""
    ts = pd.Series(pd.to_datetime(df["timestamp"]).drop_duplicates().sort_values())
    if len(ts) < 3:
        return "D"
    delta = ts.diff().dropna().median()
    if delta <= pd.Timedelta(hours=1):
        return "h"
    if delta <= pd.Timedelta(days=1):
        return "D"
    return "D"


def series_ids(df: pd.DataFrame) -> list[str]:
    """Return series identifiers in stable order."""
    return list(dict.fromkeys(df["series_id"].astype(str)))


def get_series(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Return one series sorted by timestamp, raising ``KeyError`` if absent."""
    sub = df[df["series_id"].astype(str) == str(series_id)]
    if sub.empty:
        raise KeyError(f"series_id {series_id!r} not found in dataset")
    return sub.sort_values("timestamp").reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Compute series count, frequency, span and observation count."""
    ts = pd.to_datetime(df["timestamp"])
    return {
        "dataset": str(df["dataset"].iloc[0]) if "dataset" in df.columns else "unknown",
        "n_series": int(df["series_id"].nunique()),
        "n_observations": int(len(df)),
        "freq": infer_freq(df),
        "start": str(ts.min().date()),
        "end": str(ts.max().date()),
        "target_mean": float(df["target"].mean()),
        "target_std": float(df["target"].std()),
        "covariates": [c for c in df.columns if c not in
                       ("dataset", "series_id", "timestamp", "target")],
    }
