"""Forecaster protocol, quantile shape and availability behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from forecastmcp.forecasters.base import ForecastRequest, ModelUnavailable
from forecastmcp.forecasters.registry import availability, get_forecaster, list_forecasters

LEVELS = [0.1, 0.5, 0.9]


def _request(series_id: str = "T001", horizon: int = 14) -> ForecastRequest:
    rng = np.random.default_rng(7)
    t = np.arange(120)
    context = 100 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, 120)
    return ForecastRequest(
        series_id=series_id,
        context=context.tolist(),
        horizon=horizon,
        quantile_levels=LEVELS,
        future_covariates={"promo": [0.0] * horizon},
        past_covariates={"promo": [0.0] * 120},
        seed=3,
    )


@pytest.mark.parametrize("model", ["seasonal_naive", "autoets", "autotheta"])
def test_baselines_quantile_shape_and_monotonicity(model):
    """Classical baselines return horizon-length, monotone quantiles."""
    forecaster = get_forecaster(model)
    ok, reason = forecaster.available()
    if not ok:  # pragma: no cover - statsforecast is a core dependency
        pytest.skip(reason)
    req = _request()
    result = forecaster.predict(req)
    assert result.horizon == req.horizon
    assert result.model_name == model
    assert len(result.point) == req.horizon
    matrix = np.asarray(result.quantiles)
    assert matrix.shape == (len(req.quantile_levels), req.horizon)
    assert np.all(np.diff(matrix, axis=0) >= 0)
    assert np.all(np.isfinite(matrix))
    assert result.latency_ms >= 0


def test_covariate_baseline_uses_promo():
    """The promo-aware baseline records covariate use and lifts promo days."""
    forecaster = get_forecaster("seasonal_naive_cov")
    rng = np.random.default_rng(1)
    context = np.full(140, 100.0) + rng.normal(0, 1, 140)
    promo = np.zeros(140)
    promo[::10] = 1.0
    context[promo == 1] *= 1.5
    req = ForecastRequest(
        series_id="P1",
        context=context.tolist(),
        horizon=14,
        quantile_levels=LEVELS,
        past_covariates={"promo": promo.tolist()},
        future_covariates={"promo": [1.0] * 14},
    )
    result = forecaster.predict(req)
    assert result.used_covariates is True
    assert result.metadata["promo_lift"] > 1.1
    base = np.mean(context[promo == 0])
    assert np.mean(result.point) > base


def test_chronos_deterministic_and_monotone():
    """Chronos produces monotone quantiles and repeats exactly under a fixed seed."""
    ok, reason = availability("chronos")
    if not ok:
        assert isinstance(reason, str) and reason
        pytest.skip(f"chronos unavailable: {reason}")
    forecaster = get_forecaster("chronos")
    req = _request(horizon=7)
    first = forecaster.predict(req)
    second = forecaster.predict(req)
    assert np.asarray(first.quantiles).shape == (3, 7)
    assert np.all(np.diff(np.asarray(first.quantiles), axis=0) >= 0)
    assert first.quantiles == second.quantiles
    assert first.revision != "" and first.n_params and first.n_params > 0


def test_unavailable_model_reports_reason():
    """Optional foundation models report a verbatim reason instead of crashing."""
    rows = {row["name"]: row for row in list_forecasters(include_disabled=True)}
    assert "timesfm" in rows and "moirai" in rows
    for name in ("timesfm", "moirai"):
        ok, reason = availability(name)
        if not ok:
            assert "importable" in reason or "offline" in reason or "not permitted" in reason
    # Forcing an unavailable adapter must raise the typed error, never a bare crash.
    forecaster = get_forecaster("timesfm")
    ok, reason = forecaster.available()
    if not ok:
        with pytest.raises(ModelUnavailable):
            forecaster.predict(_request())


def test_request_validation_rejects_bad_covariate_length():
    """Pydantic rejects future covariates that do not match the horizon."""
    with pytest.raises(ValueError):
        ForecastRequest(
            series_id="x",
            context=[1.0, 2.0, 3.0],
            horizon=5,
            future_covariates={"promo": [0.0, 1.0]},
        )
