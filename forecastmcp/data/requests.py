"""Build forecaster requests from the frozen holdout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastmcp.data.splits import Window, holdout_windows
from forecastmcp.forecasters.base import DEFAULT_QUANTILES, ForecastRequest

COVARIATE_COLUMNS = ("promo", "lead_time_days", "unit_price")


@dataclass
class WindowRequest:
    """A request plus the ground truth it will be scored against."""

    request: ForecastRequest
    actual: list[float]
    dates: list[str]
    window: Window

    @property
    def horizon(self) -> int:
        return self.request.horizon


def build_window_request(
    df: pd.DataFrame,
    dataset: str,
    series_id: str,
    horizon: int = 28,
    window_id: int | None = None,
    covariates: bool = True,
    quantiles: list[float] | None = None,
) -> WindowRequest:
    """Create a ForecastRequest ending at a frozen window's train boundary.

    ``window_id`` defaults to the last frozen window (the most recent origin).
    Raises ``KeyError``/``ValueError`` for unknown series, windows, or missing
    history so MCP tools can return a typed error.
    """
    windows = holdout_windows(dataset)
    if window_id is None:
        window = windows[-1]
    else:
        window = next((w for w in windows if w.window_id == int(window_id)), None)
        if window is None:
            raise ValueError(f"window {window_id} is not in the frozen holdout for {dataset}")
    series = df[df["series_id"].astype(str) == str(series_id)].sort_values("timestamp")
    if series.empty:
        raise KeyError(f"series_id {series_id!r} not found in {dataset}")
    ts = pd.to_datetime(series["timestamp"])
    train = series[ts <= window.train_end_ts]
    test = series[(ts >= window.test_start_ts) & (ts <= window.test_end_ts)]
    if train.empty or test.empty:
        raise ValueError(f"series {series_id!r} has no observations for window {window.window_id}")
    horizon = max(1, min(int(horizon), len(test)))
    cov_cols = [c for c in COVARIATE_COLUMNS if c in series.columns]
    past = {c: train[c].to_numpy(dtype=float).tolist() for c in cov_cols}
    future = {c: test[c].to_numpy(dtype=float).tolist()[:horizon] for c in cov_cols}
    request = ForecastRequest(
        series_id=str(series_id),
        context=train["target"].to_numpy(dtype=float).tolist(),
        horizon=horizon,
        quantile_levels=list(quantiles or DEFAULT_QUANTILES),
        timestamps=[pd.Timestamp(v).to_pydatetime() for v in train["timestamp"]],
        freq=window.freq,
        past_covariates=past if covariates and past else None,
        future_covariates=future if covariates and future else None,
        seed=42,
    )
    return WindowRequest(
        request=request,
        actual=test["target"].to_numpy(dtype=float).tolist()[:horizon],
        dates=[str(v) for v in test["timestamp"].dt.strftime("%Y-%m-%d").tolist()[:horizon]],
        window=window,
    )
