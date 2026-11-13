"""Quantify disagreement between forecasters and reconcile their quantiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forecastmcp.forecasters.base import ForecastResult


@dataclass
class CombinedForecast:
    """Reconciled quantile forecast plus the rule that produced it."""

    point: list[float]
    quantiles: list[list[float]]
    quantile_levels: list[float]
    members: list[str]
    weights: dict[str, float]
    method: str
    rule_fired: str
    divergence: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for tool outputs and reports."""
        return {
            "point": self.point,
            "quantiles": self.quantiles,
            "quantile_levels": self.quantile_levels,
            "members": self.members,
            "weights": self.weights,
            "method": self.method,
            "rule_fired": self.rule_fired,
            "divergence": self.divergence,
            "warnings": self.warnings,
        }


def pairwise_divergence(results: list[ForecastResult]) -> dict[str, Any]:
    """Mean pairwise quantile divergence, normalized to a scale-free ratio.

    For models a and b: ``mean_{q,t}|q_a - q_b| / mean(|q_a| + |q_b|)``.
    Values near 0 mean the models agree; > 0.2 is treated as material disagreement.
    """
    pairs: dict[str, float] = {}
    values = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = np.asarray(results[i].quantiles, dtype=float)
            b = np.asarray(results[j].quantiles, dtype=float)
            if a.shape != b.shape:
                continue
            denom = float(np.mean(np.abs(a) + np.abs(b))) or 1.0
            div = float(np.mean(np.abs(a - b)) / denom)
            pairs[f"{results[i].model_name}|{results[j].model_name}"] = round(div, 6)
            values.append(div)
    mean_div = float(np.mean(values)) if values else 0.0
    if mean_div >= 0.2:
        level = "high"
    elif mean_div >= 0.08:
        level = "moderate"
    else:
        level = "low"
    return {
        "mean_pairwise": round(mean_div, 6),
        "max_pairwise": round(max(values), 6) if values else 0.0,
        "pairs": pairs,
        "level": level,
    }


def _normalised_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: round(v / total, 6) for k, v in raw.items()}


def combine(
    results: list[ForecastResult],
    method: str = "vincentization",
    weights: list[float] | None = None,
    backtest_scores: dict[str, float] | None = None,
) -> CombinedForecast:
    """Reconcile forecasts by vincentization, inverse error or best-on-backtest.

    Parameters
    ----------
    results:
        Two or more forecasts sharing a horizon and quantile grid.
    method:
        ``vincentization`` (weighted quantile average), ``inverse_error``
        (weights proportional to 1/backtest WQL) or ``best_on_backtest``.
    weights:
        Optional user weights for vincentization.
    backtest_scores:
        Model -> WQL (lower is better), required for error-based rules.
    """
    if not results:
        raise ValueError("at least one forecast is required")
    levels = list(results[0].quantile_levels)
    horizon = results[0].horizon
    for res in results:
        if res.quantile_levels != levels or res.horizon != horizon:
            raise ValueError("all forecasts must share quantile levels and horizon")
    names = [r.model_name for r in results]
    divergence = pairwise_divergence(results)
    warnings: list[str] = []

    if method == "vincentization":
        if weights is not None:
            if len(weights) != len(results):
                raise ValueError("weights must match the number of models")
            raw = {n: float(w) for n, w in zip(names, weights, strict=False)}
            rule = "user_weights"
        else:
            raw = {n: 1.0 for n in names}
            rule = "equal_weight_vincentization"
        w = _normalised_weights(raw)
        matrix = np.zeros((len(levels), horizon), dtype=float)
        for res in results:
            matrix += w[res.model_name] * np.asarray(res.quantiles, dtype=float)
    elif method in {"inverse_error", "best_on_backtest"}:
        if not backtest_scores:
            warnings.append(
                "no backtest metrics available; fell back to equal-weight vincentization"
            )
            return combine(results, method="vincentization")
        scores = {n: max(float(backtest_scores.get(n, np.inf)), 1e-9) for n in names}
        if method == "inverse_error":
            raw = {n: 1.0 / scores[n] for n in names}
            w = _normalised_weights(raw)
            rule = "inverse_backtest_wql"
        else:
            best = min(names, key=lambda n: scores[n])
            w = {n: (1.0 if n == best else 0.0) for n in names}
            rule = f"best_on_backtest:{best}"
        matrix = np.zeros((len(levels), horizon), dtype=float)
        for res in results:
            matrix += w[res.model_name] * np.asarray(res.quantiles, dtype=float)
    elif method == "quantile_average":
        w = _normalised_weights({n: 1.0 for n in names})
        matrix = np.mean([np.asarray(r.quantiles, dtype=float) for r in results], axis=0)
        rule = "equal_weight_quantile_average"
    else:
        raise ValueError(f"unknown reconciliation method {method!r}")

    # Quantile averaging can cross levels; repair monotonically.
    for step in range(horizon):
        matrix[:, step] = np.sort(matrix[:, step])
    median_idx = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
    return CombinedForecast(
        point=[float(v) for v in matrix[median_idx]],
        quantiles=[[float(v) for v in row] for row in matrix],
        quantile_levels=levels,
        members=names,
        weights=w,
        method=method,
        rule_fired=rule,
        divergence=divergence,
        warnings=warnings,
    )
