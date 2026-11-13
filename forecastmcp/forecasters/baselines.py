"""Classical statistical baselines with proper quantiles.

StatsForecast models provide model-native two-sided intervals; those are turned
into a quantile function by linear interpolation over the interval anchors plus
the point forecast as the median. When no native intervals exist the adapter
falls back to empirical in-sample residual quantiles.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from forecastmcp.forecasters.base import (
    BaseForecaster,
    ForecastRequest,
    ForecastResult,
    check_import,
)

ANCHOR_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def _season_length(freq: str) -> int:
    f = freq.lower()
    if f.startswith("h"):
        return 24
    if f.startswith(("d", "b")):
        return 7
    return 1


@lru_cache(maxsize=1)
def _statsforecast_available() -> tuple[bool, str]:
    ok, reason = check_import("statsforecast")
    if not ok:
        return False, reason
    return True, "statsforecast importable"


def _make_model(kind: str, season: int):
    from statsforecast.models import AutoETS, AutoTheta, SeasonalNaive

    if kind == "seasonal_naive":
        return SeasonalNaive(season_length=season)
    if kind == "autoets":
        return AutoETS(season_length=season, model="ZZZ")
    if kind == "autotheta":
        return AutoTheta(season_length=season, decomposition_type="multiplicative")
    raise ValueError(f"unknown baseline kind {kind!r}")


def _dates_for(req: ForecastRequest, freq: str) -> pd.DatetimeIndex:
    if req.timestamps is not None:
        return pd.DatetimeIndex(pd.to_datetime(req.timestamps))
    return pd.date_range(end=pd.Timestamp("2024-12-31"), periods=len(req.context), freq=freq)


def _interpolate_quantiles(
    point: np.ndarray,
    anchor_q: list[float],
    anchor_v: list[list[float]],
    levels: list[float],
    horizon: int,
) -> np.ndarray:
    """Interpolate a quantile function per forecast step from interval anchors."""
    out = np.zeros((len(levels), horizon), dtype=float)
    for step in range(horizon):
        q = np.asarray(anchor_q, dtype=float)
        v = np.asarray([row[step] for row in anchor_v], dtype=float)
        good = np.isfinite(v)
        if not np.any(good):
            raise ValueError("no finite interval anchors")
        q, v = q[good], v[good]
        order = np.argsort(q)
        q, v = q[order], v[order]
        # Drop duplicate quantile anchors (e.g. repeated point at 0.5).
        uniq, idx = np.unique(q, return_index=True)
        v = v[idx]
        if len(uniq) < 2:
            out[:, step] = v[0]
        else:
            out[:, step] = np.interp(levels, uniq, v)
    return out


def _residual_quantiles(
    req: ForecastRequest, point: np.ndarray, levels: list[float], season: int
) -> np.ndarray:
    """Empirical seasonal-naive residual quantiles around a point forecast."""
    y = np.asarray(req.context, dtype=float)
    resid = y[season:] - y[:-season] if len(y) > season + 5 else np.diff(y)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 5:
        scale = float(np.nanstd(y) or 1.0)
        from scipy.stats import norm

        offsets = norm.ppf(levels) * scale
    else:
        offsets = np.quantile(resid, levels)
    return np.asarray(point)[None, :] + np.asarray(offsets)[:, None]


class StatsForecaster(BaseForecaster):
    """One StatsForecast model exposed through the Forecaster protocol."""

    supports_covariates = False

    def __init__(self, name: str, config: dict[str, Any] | None = None, kind: str | None = None):
        super().__init__(name, config)
        self.kind = kind or name

    def available(self) -> tuple[bool, str]:
        """Check that StatsForecast imports."""
        return _statsforecast_available()

    def _run(self, reqs: list[ForecastRequest]) -> tuple[pd.DataFrame, float, int]:
        from statsforecast import StatsForecast

        season = _season_length(reqs[0].freq)
        frames = []
        for req in reqs:
            dates = _dates_for(req, req.freq)
            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": req.series_id,
                        "ds": dates,
                        "y": np.asarray(req.context, dtype=float),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        model = _make_model(self.kind, season)
        sf = StatsForecast(
            models=[model],
            freq=reqs[0].freq,
            n_jobs=1,
            fallback_model=_make_model("seasonal_naive", season),
        )
        started = time.perf_counter()
        out = sf.forecast(df=df, h=reqs[0].horizon, level=ANCHOR_LEVELS)
        elapsed = time.perf_counter() - started
        return out, elapsed, season

    def predict_many(self, reqs: list[ForecastRequest]) -> list[ForecastResult]:
        """Fit once for the whole panel and return one result per request."""
        if not reqs:
            return []
        out, elapsed, season = self._run(reqs)
        alias = _make_model(self.kind, season).alias
        results: list[ForecastResult] = []
        for req in reqs:
            sub = out[out["unique_id"] == req.series_id].sort_values("ds").head(req.horizon)
            point = sub[alias].to_numpy(dtype=float)
            anchor_q: list[float] = []
            anchor_v: list[list[float]] = []
            for level in ANCHOR_LEVELS:
                lo_col, hi_col = f"{alias}-lo-{level}", f"{alias}-hi-{level}"
                if lo_col in sub.columns and hi_col in sub.columns:
                    lo = sub[lo_col].to_numpy(dtype=float)
                    hi = sub[hi_col].to_numpy(dtype=float)
                    if np.any(np.isfinite(lo)):
                        anchor_q.extend([0.5 - level / 200, 0.5 + level / 200])
                        anchor_v.extend([lo, hi])
            warnings: list[str] = []
            if anchor_q:
                anchor_q.append(0.5)
                anchor_v.append(point)
                quantiles = _interpolate_quantiles(
                    point, anchor_q, anchor_v, req.quantile_levels, req.horizon
                )
            else:
                quantiles = _residual_quantiles(req, point, req.quantile_levels, season)
                warnings.append("model produced no native intervals; empirical residual quantiles")
            if req.past_covariates or req.future_covariates:
                warnings.append(
                    f"{self.name} does not consume covariates; inputs ignored "
                    "(used_covariates=false)"
                )
            results.append(
                self.finalize(
                    req,
                    point,
                    quantiles,
                    time.perf_counter(),
                    device="cpu",
                    n_params=0,
                    warnings=warnings,
                    used_covariates=False,
                    metadata={
                        "family": "statsforecast",
                        "season_length": season,
                        "amortized_over_batch": len(reqs),
                        "fit_seconds": round(elapsed, 4),
                    },
                )
            )
            results[-1].latency_ms = round(elapsed * 1000.0 / max(len(reqs), 1), 3)
        return results

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Forecast one series through the batched implementation."""
        return self.predict_many([req])[0]


class PromoSeasonalNaive(BaseForecaster):
    """Seasonal naive adjusted by a promotion uplift learned from history.

    This is the covariate arm of the benchmark: it is a real, if simple, model
    that uses the ``promo`` covariate. It exists to measure whether covariates
    help at all before foundation-model covariate support is considered.
    """

    supports_covariates = True

    def available(self) -> tuple[bool, str]:
        """No optional dependency beyond numpy/pandas."""
        return True, "ok"

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Apply the learned promo lift to a seasonal-naive forecast."""
        started = time.perf_counter()
        season = _season_length(req.freq)
        y = np.asarray(req.context, dtype=float)
        promo_past = (req.past_covariates or {}).get("promo")
        future_promo = (req.future_covariates or {}).get("promo")
        warnings: list[str] = []
        used = False
        lift = 1.0
        if promo_past is not None and future_promo is not None:
            promo_arr = np.asarray(promo_past, dtype=float) > 0.5
            if promo_arr.sum() >= 5 and (~promo_arr).sum() >= 5:
                base_mean = float(np.mean(y[~promo_arr]))
                promo_mean = float(np.mean(y[promo_arr]))
                if base_mean > 0:
                    lift = max(0.2, min(5.0, promo_mean / base_mean))
                used = True
            else:
                warnings.append(
                    "too few promo/non-promo observations to estimate a lift; covariate ignored"
                )
            future_arr = np.asarray(future_promo, dtype=float)
        else:
            warnings.append("promo covariates missing; falling back to plain seasonal naive")
            future_arr = np.zeros(req.horizon, dtype=float)

        m = min(season, len(y))
        base = np.array([y[len(y) - m + (i % m)] for i in range(req.horizon)], dtype=float)
        point = base * (1 + (lift - 1) * future_arr)

        # Residuals from the promo-adjusted in-sample seasonal-naive fit.
        if len(y) > m:
            past_promo_arr = (
                np.asarray(promo_past, dtype=float) > 0.5
                if promo_past is not None
                else np.zeros(len(y), dtype=bool)
            )
            fitted = y[:-m] * (1 + (lift - 1) * past_promo_arr[m:].astype(float))
            resid = y[m:] - fitted
            resid = resid[np.isfinite(resid)]
        else:
            resid = np.array([], dtype=float)
        if len(resid) >= 5:
            offsets = np.quantile(resid, req.quantile_levels)
        else:
            offsets = np.zeros(len(req.quantile_levels))
        quantiles = point[None, :] + offsets[:, None]
        if used:
            warnings.append(f"promo lift multiplier estimated from history: {lift:.4f}")
        return self.finalize(
            req,
            point,
            quantiles,
            started,
            device="cpu",
            n_params=0,
            warnings=warnings,
            used_covariates=used,
            metadata={"family": "covariate-baseline", "promo_lift": round(lift, 6)},
        )

