"""Series-level bootstrap confidence intervals for metrics and paired deltas."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42


def series_means(
    frame: pd.DataFrame, metric: str, value_col: str = "value", model: str | None = None
) -> pd.Series:
    """Average a metric over windows for each series (the bootstrap unit)."""
    sub = frame[frame["metric"] == metric]
    if model is not None:
        sub = sub[sub["model"] == model]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("series_id")[value_col].mean()


def bootstrap_ci(
    series_values: np.ndarray | pd.Series,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of per-series values."""
    values = np.asarray(series_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_series": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n_series": int(values.size),
    }


def paired_bootstrap_delta(
    frame: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Paired series-level bootstrap of ``mean(a) - mean(b)`` for one metric.

    Positive deltas mean model A is worse for error metrics (WQL/CRPS/MASE).
    ``crosses_zero`` is True when the 95% CI includes zero (no significant
    difference at that level).
    """
    sub = frame[frame["metric"] == metric]
    a = sub[sub["model"] == model_a].groupby("series_id")["value"].mean()
    b = sub[sub["model"] == model_b].groupby("series_id")["value"].mean()
    joined = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if joined.empty:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "crosses_zero": True,
            "n_series": 0,
        }
    deltas = (joined["a"] - joined["b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    means = deltas[idx].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "delta": float(deltas.mean()),
        "ci_low": low,
        "ci_high": high,
        "crosses_zero": bool(low <= 0 <= high),
        "n_series": int(deltas.size),
    }


def run_to_run_variance(values: list[float]) -> dict[str, float]:
    """Mean and standard deviation across repeated trials."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}
