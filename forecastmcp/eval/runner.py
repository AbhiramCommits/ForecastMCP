"""Benchmark runner: frozen windows -> forecasts -> metrics -> DuckDB + CSVs."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecastmcp.agent.reconcile import combine
from forecastmcp.config import CONFIGS_DIR, RESULTS_DIR, is_offline, read_yaml
from forecastmcp.data.loaders import dataset_summary, load_dataset
from forecastmcp.data.requests import build_window_request
from forecastmcp.data.splits import eligible_series, holdout_windows
from forecastmcp.data.store import get_store
from forecastmcp.eval.bootstrap import bootstrap_ci, paired_bootstrap_delta, series_means
from forecastmcp.eval.metrics import (
    crps_from_quantiles,
    interval_coverage,
    mase,
    smape,
    weighted_quantile_loss,
    winkler_score,
)
from forecastmcp.forecasters.base import ForecastResult
from forecastmcp.forecasters.registry import (
    availability,
    estimate_cost,
    get_forecaster,
)

EVAL_GRID = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
METRICS = ["WQL", "CRPS", "MASE", "coverage_50", "coverage_80", "coverage_90",
           "sMAPE", "winkler_80"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _season(freq: str) -> int:
    return 24 if freq.lower().startswith("h") else 7


def _metric_row(y: np.ndarray, res: ForecastResult, context: list[float], freq: str) -> dict[str, float]:
    q = np.asarray(res.quantiles, dtype=float)
    levels = res.quantile_levels
    lo50, hi50 = res.lower_upper(0.5)
    lo80, hi80 = res.lower_upper(0.8)
    lo90, hi90 = res.lower_upper(0.9)
    return {
        "WQL": weighted_quantile_loss(y, q, levels),
        "CRPS": crps_from_quantiles(y, q, levels),
        "MASE": mase(y, res.point, context, _season(freq)),
        "coverage_50": interval_coverage(y, lo50, hi50),
        "coverage_80": interval_coverage(y, lo80, hi80),
        "coverage_90": interval_coverage(y, lo90, hi90),
        "sMAPE": smape(y, res.point),
        "winkler_80": winkler_score(y, lo80, hi80, 0.8),
    }


def _combined_metric_row(
    y: np.ndarray, combined, context: list[float], freq: str
) -> dict[str, float]:
    levels = combined.quantile_levels
    q = np.asarray(combined.quantiles, dtype=float)

    def q_at(level: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(np.asarray(levels) - level)))
        return q[idx]

    return {
        "WQL": weighted_quantile_loss(y, q, levels),
        "CRPS": crps_from_quantiles(y, q, levels),
        "MASE": mase(y, combined.point, context, _season(freq)),
        "coverage_50": interval_coverage(y, q_at(0.25), q_at(0.75)),
        "coverage_80": interval_coverage(y, q_at(0.1), q_at(0.9)),
        "coverage_90": interval_coverage(y, q_at(0.05), q_at(0.95)),
        "sMAPE": smape(y, combined.point),
        "winkler_80": winkler_score(y, q_at(0.1), q_at(0.9), 0.8),
    }


def _forecast_frame(
    run_id: str,
    dataset: str,
    model: str,
    series_ids: list[str],
    requests: list,
    results: list[ForecastResult],
) -> pd.DataFrame:
    rows = []
    for sid, wr, res in zip(series_ids, requests, results, strict=True):
        dates = pd.to_datetime(wr.dates)
        for step in range(res.horizon):
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "model": model,
                    "series_id": sid,
                    "window_id": wr.window.window_id,
                    "step": step + 1,
                    "ts": dates[step],
                    "point": res.point[step],
                    "actual": wr.actual[step],
                    "quantiles": [row[step] for row in res.quantiles],
                    "quantile_levels": res.quantile_levels,
                }
            )
    return pd.DataFrame(rows)


def run_benchmark(
    config_path: Path | str = CONFIGS_DIR / "benchmark.yaml",
    run_id: str | None = None,
    fast: bool = False,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    max_series: int | None = None,
    n_windows: int | None = None,
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Evaluate every available model on the frozen holdout and persist results."""
    cfg = read_yaml(config_path, default=None) or {}
    run_id = run_id or f"benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    ds_cfg = cfg.get("datasets", {"synthetic_supply": {"max_series": 50}})
    dataset_names = datasets or (["synthetic_supply"] if fast else list(ds_cfg))
    if is_offline():
        dataset_names = [d for d in dataset_names if d == "synthetic_supply"] or ["synthetic_supply"]
    model_override = list(models) if models else None
    if is_offline() and model_override:
        model_override = [m for m in model_override if availability(m)[0]]
    n_windows = int(n_windows or (2 if fast else cfg.get("n_windows", 6)))
    if fast:
        n_resamples = min(n_resamples, 200)
    seed = int(cfg.get("seed", 42))
    default_ensembles = [
        "ensemble_vincentization", "ensemble_best_on_backtest", "ensemble_inverse_error"
    ]

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    store = get_store()
    store.delete_run(run_id)
    store.create_run(run_id, "benchmark", config=cfg, git_sha=_git_sha())

    metric_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    latency: dict[tuple[str, str], list[float]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    from forecastmcp.forecasters.registry import list_forecasters

    availability_table = {
        row["name"]: {
            "available": row["available"],
            "reason": row["reason"],
            "adapter": row["adapter"],
            "enabled": row["enabled"],
        }
        for row in list_forecasters(include_disabled=True)
    }
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "n_resamples": n_resamples,
        "eval_quantile_grid": EVAL_GRID,
        "datasets": {},
        "unavailable_models": unavailable,
        "model_availability": availability_table,
        "fast": fast,
    }
    dataset_frames: dict[str, pd.DataFrame] = {}

    for dataset in dataset_names:
        df = load_dataset(dataset)
        summary = dataset_summary(df)
        all_windows = holdout_windows(dataset)
        windows = all_windows[:n_windows]
        eligible = eligible_series(df, windows)
        limit = max_series if max_series is not None else int(ds_cfg.get(dataset, {}).get("max_series", 50))
        if limit:
            eligible = eligible[:limit]
        if not eligible:
            manifest["datasets"][dataset] = {"skipped": "no eligible series"}
            continue
        configured_models = ds_cfg.get(dataset, {}).get(
            "models", cfg.get("models", ["seasonal_naive", "autoets", "chronos"])
        )
        ds_models = [
            m for m in configured_models if not model_override or m in model_override
        ]
        configured_ensembles = ds_cfg.get(dataset, {}).get(
            "ensembles", cfg.get("ensembles", default_ensembles)
        )
        ds_ensembles = ["ensemble_vincentization"] if (is_offline() or fast) else list(configured_ensembles)
        manifest["datasets"][dataset] = {
            "n_series": len(eligible),
            "freq": summary["freq"],
            "horizon": windows[0].horizon,
            "n_windows": len(windows),
            "window_ids": [w.window_id for w in windows],
            "models_requested": ds_models,
            "covariates": summary["covariates"],
        }
        unavailable.setdefault(dataset, {})
        # model -> window_id -> (requests, results)
        per_model: dict[str, dict[int, tuple[list, list[ForecastResult]]]] = {}
        model_scores: dict[str, float] = {}

        for model in ds_models:
            ok, reason = availability(model)
            if not ok:
                unavailable[dataset][model] = reason
                continue
            forecaster = get_forecaster(model)
            per_model[model] = {}
            for window in windows:
                requests = [
                    build_window_request(
                        df, dataset, sid, horizon=window.horizon,
                        window_id=window.window_id, covariates=True, quantiles=EVAL_GRID,
                    )
                    for sid in eligible
                ]
                try:
                    results = forecaster.predict_many([w.request for w in requests])
                except Exception as exc:
                    unavailable[dataset][model] = f"{type(exc).__name__}: {exc}"
                    break
                per_model[model][window.window_id] = (requests, results)
                frame = _forecast_frame(run_id, dataset, model, eligible, requests, results)
                store.write_forecasts(frame)
                for sid, wr, res in zip(eligible, requests, results, strict=True):
                    y = np.asarray(wr.actual, dtype=float)
                    row = _metric_row(y, res, wr.request.context, wr.window.freq)
                    for metric, value in row.items():
                        metric_rows.append(
                            {
                                "run_id": run_id, "dataset": dataset, "model": model,
                                "series_id": sid, "window_id": window.window_id,
                                "metric": metric, "value": value,
                            }
                        )
                    q = np.asarray(res.quantiles)
                    levels = np.asarray(res.quantile_levels)
                    for step in range(res.horizon):
                        step_loss = np.mean([
                            np.where(
                                (y[step] - q[j, step]) >= 0,
                                levels[j] * (y[step] - q[j, step]),
                                (levels[j] - 1) * (y[step] - q[j, step]),
                            )
                            for j in range(len(levels))
                        ])
                        horizon_rows.append(
                            {
                                "run_id": run_id, "dataset": dataset, "model": model,
                                "step": step + 1, "quantile_loss": 2.0 * step_loss,
                                "abs_actual": abs(float(y[step])),
                            }
                        )
                    latency.setdefault((dataset, model), []).append(float(res.latency_ms))
            if model in per_model and per_model[model]:
                subset = [r for r in metric_rows if r["dataset"] == dataset
                          and r["model"] == model and r["metric"] == "WQL"]
                model_scores[model] = float(np.mean([r["value"] for r in subset])) if subset else np.inf

        # -- ensembles over the models that actually ran --------------------
        available_models = [m for m in ds_models if m in per_model and per_model[m]]
        for ensemble_name in ds_ensembles:
            if len(available_models) < 2:
                continue
            if ensemble_name == "ensemble_vincentization":
                method, scores = "vincentization", None
            elif ensemble_name == "ensemble_inverse_error":
                method, scores = "inverse_error", model_scores
            elif ensemble_name == "ensemble_best_on_backtest":
                method, scores = "best_on_backtest", model_scores
            else:
                continue
            for window in windows:
                requests = None
                member_results = []
                for model in available_models:
                    if window.window_id not in per_model[model]:
                        member_results = []
                        break
                    reqs, res = per_model[model][window.window_id]
                    requests = requests or reqs
                    member_results.append(res)
                if not member_results or requests is None:
                    continue
                for i, wr in enumerate(requests):
                    combo = combine(
                        [res[i] for res in member_results],
                        method=method,
                        backtest_scores=scores,
                    )
                    y = np.asarray(wr.actual, dtype=float)
                    row = _combined_metric_row(y, combo, wr.request.context, wr.window.freq)
                    for metric, value in row.items():
                        metric_rows.append(
                            {
                                "run_id": run_id, "dataset": dataset, "model": ensemble_name,
                                "series_id": eligible[i], "window_id": window.window_id,
                                "metric": metric, "value": value,
                            }
                        )
                    for step in range(len(combo.point)):
                        q = np.asarray(combo.quantiles)
                        levels = np.asarray(combo.quantile_levels)
                        step_loss = np.mean([
                            np.where(
                                (y[step] - q[j, step]) >= 0,
                                levels[j] * (y[step] - q[j, step]),
                                (levels[j] - 1) * (y[step] - q[j, step]),
                            )
                            for j in range(len(levels))
                        ])
                        horizon_rows.append(
                            {
                                "run_id": run_id, "dataset": dataset, "model": ensemble_name,
                                "step": step + 1, "quantile_loss": 2.0 * step_loss,
                                "abs_actual": abs(float(y[step])),
                            }
                        )
            member_lat = [
                float(np.median(latency.get((dataset, m), [np.nan]))) for m in available_models
            ]
            latency.setdefault((dataset, ensemble_name), []).append(float(np.nansum(member_lat)))
        dataset_frames[dataset] = pd.DataFrame(
            [r for r in metric_rows if r["dataset"] == dataset]
        )

    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        store.write_metrics(
            metric_df.assign(ci_low=None, ci_high=None)
        )
        metric_df.to_csv(out_dir / "per_series_metrics.csv", index=False)

    # -- bootstrap summaries and paired deltas ------------------------------
    summary: dict[str, Any] = {"run_id": run_id, "manifest": manifest,
                               "datasets": {}, "paired_deltas": {}, "ablations": {}}
    summary_rows = []
    for dataset, frame in dataset_frames.items():
        if frame.empty:
            continue
        models = sorted(frame["model"].unique())
        summary["datasets"][dataset] = {}
        for model in models:
            entry: dict[str, Any] = {}
            for metric in METRICS:
                values = series_means(frame, metric, model=model)
                ci = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
                entry[metric] = ci
                summary_rows.append(
                    {
                        "dataset": dataset, "model": model, "metric": metric,
                        "mean": ci["mean"], "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"], "n_series": ci["n_series"],
                    }
                )
            lat = latency.get((dataset, model), [])
            median_latency = float(np.median(lat)) if lat else float("nan")
            if model.startswith("ensemble_"):
                member_costs = []
                for member in models:
                    if member.startswith("ensemble_"):
                        continue
                    member_lat = latency.get((dataset, member), [])
                    if member_lat:
                        member_costs.append(
                            estimate_cost(member, float(np.median(member_lat)))
                        )
                cost = float(np.sum(member_costs)) if member_costs else float("nan")
            else:
                cost = estimate_cost(model, median_latency) if lat else float("nan")
            entry["median_latency_ms"] = median_latency
            entry["estimated_cost_per_call_usd"] = cost
            summary["datasets"][dataset][model] = entry

        singles = [m for m in models if not m.startswith("ensemble_")]
        best_single = min(singles, key=lambda m: summary["datasets"][dataset][m]["WQL"]["mean"]) \
            if singles else None
        summary["datasets"][dataset]["_best_single_model"] = best_single
        pairs = []
        if "chronos-lora-v1" in models and "chronos" in models:
            pairs.append(("chronos-lora-v1", "chronos", "zero_shot_vs_lora"))
        if best_single:
            for ens in models:
                if ens.startswith("ensemble_"):
                    pairs.append((ens, best_single, "ensemble_vs_best_single"))
        if "seasonal_naive_cov" in models and "seasonal_naive" in models:
            pairs.append(("seasonal_naive_cov", "seasonal_naive", "with_vs_without_covariates"))
        summary["paired_deltas"][dataset] = {}
        for a, b, label in pairs:
            for metric in ("WQL", "CRPS", "MASE", "coverage_80"):
                delta = paired_bootstrap_delta(
                    frame, a, b, metric, n_resamples=n_resamples, seed=seed
                )
                summary["paired_deltas"][dataset][f"{label}:{a}_minus_{b}:{metric}"] = delta

    summary["ablations"] = {
        "zero_shot_vs_finetuned": "reported in paired_deltas under zero_shot_vs_lora",
        "covariates": (
            "seasonal_naive_cov vs seasonal_naive in paired_deltas; Chronos T5 does not "
            "consume covariates so no foundation-model covariate arm exists"
        ),
        "univariate_vs_multivariate": {
            "status": "unavailable",
            "reason": (
                "TimesFM and Moirai (the covariate/multivariate-capable adapters) are not "
                "installed; see unavailable_models"
            ),
        },
        "ensemble_vs_best_single": "reported in paired_deltas under ensemble_vs_best_single",
    }
    (out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(summary_rows).to_csv(out_dir / "model_summary.csv", index=False)

    horizon_df = pd.DataFrame(horizon_rows)
    if not horizon_df.empty:
        grouped = (
            horizon_df.groupby(["dataset", "model", "step"], as_index=False)
            .agg(quantile_loss=("quantile_loss", "mean"), abs_actual=("abs_actual", "mean"))
        )
        grouped["wql_by_step"] = grouped["quantile_loss"] / grouped["abs_actual"].clip(lower=1e-9)
        grouped.to_csv(out_dir / "wql_by_horizon.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
