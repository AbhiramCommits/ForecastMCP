"""Probabilistic and point forecast metrics.

All functions accept array-likes and return a single float. Multi-series
aggregation is done by the caller (typically a mean over series/windows), which
keeps the metric definitions unambiguous and unit-testable.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _arrays(*values) -> list[np.ndarray]:
    return [np.asarray(v, dtype=float).ravel() for v in values]


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Mean pinball (quantile) loss at one quantile level.

    ``QL_q(y, f) = (1{y <= f} - q) * (y - f)``, averaged over observations.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    y, f = _arrays(y_true, y_pred)
    if y.shape != f.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    diff = y - f
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def weighted_quantile_loss(
    y_true,
    quantiles,
    levels,
    *,
    normalize: bool = True,
) -> float:
    """Weighted quantile loss (WQL), the scale-free probabilistic metric.

    ``WQL = 2 * sum_{q,t} QL_q(y_t, f_q,t) / sum_t |y_t|``

    Quantile rows are indexed ``[level, step]``. When ``normalize`` is False the
    denominator is omitted and the raw twice-summed pinball loss is returned.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(quantiles, dtype=float)
    levels_arr = np.asarray(levels, dtype=float)
    if q.ndim != 2:
        raise ValueError("quantiles must have shape [levels, steps]")
    if q.shape[1] != y.shape[0]:
        raise ValueError("quantile step count must match y_true")
    if q.shape[0] != levels_arr.shape[0]:
        raise ValueError("one quantile row per level is required")
    total = 0.0
    for row, level in zip(q, levels_arr, strict=False):
        total += pinball_loss(y, row, float(level))
    total *= 2.0 * y.shape[0]
    if normalize:
        denom = float(np.sum(np.abs(y)))
        total /= max(denom, _EPS)
    return float(total)


def crps_from_quantiles(y_true, quantiles, levels) -> float:
    """CRPS approximated from a quantile grid.

    For a uniform grid of quantile levels the quantile score integrates to CRPS:
    ``CRPS = 2 * integral_0^1 QL_q dq ~= 2 * mean_q QL_q``. With the 0.1..0.9
    grid used throughout this repo this omits the outer 10% tails, so it is a
    documented approximation, not the exact CRPS.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(quantiles, dtype=float)
    levels_arr = np.asarray(levels, dtype=float)
    losses = [pinball_loss(y, row, float(level)) for row, level in zip(q, levels_arr, strict=False)]
    return float(2.0 * np.mean(losses))


def mase(y_true, y_pred, context, season: int = 7) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal naive error."""
    y, f = _arrays(y_true, y_pred)
    ctx = np.asarray(context, dtype=float).ravel()
    if len(ctx) <= season:
        scale = float(np.mean(np.abs(np.diff(ctx)))) if len(ctx) > 1 else 1.0
    else:
        scale = float(np.mean(np.abs(ctx[season:] - ctx[:-season])))
    scale = max(scale, _EPS)
    return float(np.mean(np.abs(y - f)) / scale)


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error in [0, 2]."""
    y, f = _arrays(y_true, y_pred)
    denom = (np.abs(y) + np.abs(f)) / 2.0
    ratio = np.where(denom < _EPS, 0.0, np.abs(y - f) / np.maximum(denom, _EPS))
    return float(np.mean(ratio))


def interval_coverage(y_true, lower, upper) -> float:
    """Empirical coverage: fraction of actuals inside ``[lower, upper]``."""
    y, lo, hi = _arrays(y_true, lower, upper)
    return float(np.mean((y >= lo) & (y <= hi)))


def winkler_score(y_true, lower, upper, nominal: float) -> float:
    """Mean Winkler (interval) score for a central ``nominal`` interval."""
    if not 0 < nominal < 1:
        raise ValueError("nominal must be strictly between 0 and 1")
    y, lo, hi = _arrays(y_true, lower, upper)
    alpha = 1.0 - nominal
    width = hi - lo
    penalty_low = np.where(y < lo, 2.0 / alpha * (lo - y), 0.0)
    penalty_high = np.where(y > hi, 2.0 / alpha * (y - hi), 0.0)
    return float(np.mean(width + penalty_low + penalty_high))
