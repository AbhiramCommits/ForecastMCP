"""Closed-form checks for probabilistic metrics, plus GluonTS agreement."""

from __future__ import annotations

import numpy as np
import pytest

from forecastmcp.eval.bootstrap import bootstrap_ci, paired_bootstrap_delta
from forecastmcp.eval.metrics import (
    crps_from_quantiles,
    interval_coverage,
    mase,
    pinball_loss,
    smape,
    weighted_quantile_loss,
    winkler_score,
)


def test_pinball_hand_computed():
    """Pinball loss matches hand calculations on both sides of the forecast."""
    assert pinball_loss([10.0], [12.0], 0.9) == pytest.approx(0.2)
    assert pinball_loss([10.0], [8.0], 0.1) == pytest.approx(0.2)
    assert pinball_loss([10.0], [10.0], 0.5) == 0.0
    assert pinball_loss([10.0, 20.0], [12.0, 18.0], 0.5) == pytest.approx(1.0)


def test_pinball_matches_gluonts():
    """GluonTS quantile loss is two times the summed pinball loss."""
    from gluonts.evaluation.metrics import quantile_loss

    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 40)
    f = rng.normal(100, 10, 40)
    for q in (0.05, 0.1, 0.5, 0.9):
        ours = pinball_loss(y, f, q)
        theirs = quantile_loss(y, f, q) / (2 * len(y))
        assert ours == pytest.approx(theirs, rel=1e-10)


def test_wql_and_crps_hand_computed():
    """WQL normalizes twice-summed pinball loss by |y|; CRPS is 2*mean pinball."""
    y = [10.0]
    q = [[8.0], [12.0]]
    levels = [0.1, 0.9]
    assert weighted_quantile_loss(y, q, levels) == pytest.approx(0.08)
    assert crps_from_quantiles(y, q, levels) == pytest.approx(0.4)


def test_mase_hand_computed():
    """MASE uses the seasonal-naive in-sample error as its scale."""
    context = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # seasonal naive errors: |5-1|, |6-2|, |7-3|, |8-4| -> mean 4
    # test absolute error: |8-6| = 2 -> MASE 0.5
    assert mase([8.0], [6.0], context, season=4) == pytest.approx(0.5)
    assert mase([8.0], [8.0], context, season=4) == 0.0


def test_coverage_smape_winkler():
    """Interval coverage, sMAPE and Winkler score follow their definitions."""
    assert interval_coverage([1.0, 5.0, 9.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]) == pytest.approx(1 / 3)
    assert smape([10.0], [12.0]) == pytest.approx(2 / 11)
    # Interval [8, 12], y=14, nominal 0.8 -> alpha 0.2: width 4 + 2/0.2*(14-12) = 24
    assert winkler_score([14.0], [8.0], [12.0], 0.8) == pytest.approx(24.0)


def test_bootstrap_ci_covers_mean_and_paired_delta():
    """Bootstrap CIs are finite and a planted shift is detected as non-zero."""
    rng = np.random.default_rng(1)
    values = rng.normal(10, 2, 60)
    ci = bootstrap_ci(values, n_resamples=500, seed=2)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["n_series"] == 60

    import pandas as pd

    rows = []
    for i in range(50):
        series = f"S{i:03d}"
        base = rng.normal(5, 1)
        rows.append({"series_id": series, "model": "a", "metric": "WQL",
                     "value": base + 1.0})
        rows.append({"series_id": series, "model": "b", "metric": "WQL",
                     "value": base})
    frame = pd.DataFrame(rows)
    delta = paired_bootstrap_delta(frame, "a", "b", "WQL", n_resamples=500, seed=3)
    assert delta["delta"] == pytest.approx(1.0, abs=0.01)
    assert delta["ci_low"] > 0
    assert delta["crosses_zero"] is False
