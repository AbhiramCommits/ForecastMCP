"""Evaluate the agent harness itself: validity, regret, citations, budgets, recovery."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecastmcp.agent.loop import ForecastAgent
from forecastmcp.agent.report import unsupported_number_rate
from forecastmcp.config import CONFIGS_DIR, RESULTS_DIR, is_offline, read_yaml
from forecastmcp.data.store import get_store
from forecastmcp.eval.bootstrap import run_to_run_variance
from forecastmcp.eval.metrics import weighted_quantile_loss
from forecastmcp.mcp.tools import ToolRuntime
from forecastmcp.obs.tracing import read_trace


@dataclass
class TrialResult:
    """One task x one repetition."""

    task_id: str
    trial: int
    success: bool
    chosen_model: str
    unsupported_number_rate: float
    tool_call_validity_rate: float
    budget_adherent: bool
    recovery: bool | None
    selection_regret: float | None
    ensemble_regret: float | None
    latency_seconds: float
    cost_usd: float
    calls: int
    error: str = ""


def _benchmark_run_id(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    try:
        return str(json.loads(summary_path.read_text())["run_id"])
    except Exception:
        return None


def _oracle_wql(
    run_id: str | None,
    dataset: str,
    series_id: str,
    window_id: int,
    levels: list[float],
    actual: np.ndarray,
) -> dict[str, float]:
    """Per-model WQL on the agent's own quantile grid for one series/window.

    Ranking on a fixed benchmark grid is sufficient for selection regret, but
    comparing the agent's ensemble WQL with a different grid would not be
    meaningful, so both sides are recomputed at the requested levels.
    """
    if not run_id:
        return {}
    try:
        frame = get_store().query(
            "SELECT model, step, quantiles, quantile_levels FROM forecasts "
            "WHERE run_id = ? AND dataset = ? AND series_id = ? AND window_id = ? "
            "AND model NOT LIKE 'ensemble_%' ORDER BY model, step",
            [run_id, dataset, str(series_id), int(window_id)],
        )
    except Exception:
        return {}
    horizon = len(actual)
    scores: dict[str, float] = {}
    for model, sub in frame.groupby("model"):
        sub = sub.sort_values("step")
        if len(sub) < horizon:
            continue
        stored_levels = json.loads(sub["quantile_levels"].iloc[0])
        idx = [int(np.argmin(np.abs(np.asarray(stored_levels) - level))) for level in levels]
        parsed = [json.loads(q) for q in sub["quantiles"].tolist()]
        if parsed and isinstance(parsed[0][0], list):  # legacy full-matrix rows
            matrix = np.asarray([parsed[step][:, step] for step in range(horizon)], dtype=float).T
        else:
            matrix = np.asarray(parsed, dtype=float).T
        selected = matrix[np.ix_(idx, np.arange(horizon))]
        scores[str(model)] = weighted_quantile_loss(actual, selected, levels)
    return scores


def _run_trial(task: dict[str, Any], trial: int, prefix: str,
               oracle_run: str | None) -> TrialResult:
    runtime = ToolRuntime(
        run_id=f"{prefix}-{task['id']}-t{trial}", role="analyst"
    )
    if task.get("inject_failure"):
        runtime.inject_failure("forecast", "injected fault for recovery evaluation")
    agent = ForecastAgent(runtime=runtime, prefer_llm=False)
    success = False
    result = None
    error = ""
    try:
        result = asyncio.run(agent.run(task["question"]))
        success = True
    except Exception as exc:  # a harness failure is recorded, not hidden
        error = f"{type(exc).__name__}: {exc}"
    finally:
        runtime.close()
    validity = (
        (runtime.attempts - runtime.validation_failures) / runtime.attempts
        if runtime.attempts
        else 1.0
    )
    if result is None:
        return TrialResult(
            task_id=task["id"], trial=trial, success=False, chosen_model="none",
            unsupported_number_rate=1.0, tool_call_validity_rate=validity,
            budget_adherent=False, recovery=False if task.get("inject_failure") else None,
            selection_regret=None, ensemble_regret=None, latency_seconds=0.0,
            cost_usd=0.0, calls=runtime.attempts, error=error,
        )
    records = [{"tool": c.tool, "call_id": c.call_id, "payload": c.payload} for c in result.calls]
    unsupported = unsupported_number_rate(result.report, records)
    snap = result.guard
    budget_ok = (
        snap["calls"] <= snap["max_calls"]
        and snap["cost_usd"] <= snap["max_cost_usd"] + 1e-12
        and snap["elapsed_seconds"] <= snap["max_wall_clock_seconds"] + 1e-9
    )
    spans = read_trace(runtime.trace.run_dir)
    recovered_spans = any(s["name"] == "agent.recovery" for s in spans)
    recovery = None
    if task.get("inject_failure"):
        recovery = bool(success and recovered_spans)

    ensemble = next((c for c in result.calls if c.tool == "ensemble_forecast"), None)
    forecast = next(
        (
            c
            for c in result.calls
            if c.tool == "forecast"
            and ensemble is not None
            and c.payload.get("window_id") == ensemble.payload.get("window_id")
        ),
        None,
    )
    selection_regret = ensemble_regret = None
    if ensemble is not None and forecast is not None:
        actual = np.asarray(forecast.payload["actual"], dtype=float)
        q = np.asarray(ensemble.payload["quantiles"], dtype=float)
        levels = ensemble.payload["quantile_levels"]
        horizon = min(len(actual), q.shape[1], len(ensemble.payload["point"]))
        agent_wql = weighted_quantile_loss(actual[:horizon], q[:, :horizon], levels)
        oracle = _oracle_wql(
            oracle_run,
            task["dataset"],
            task["series_id"],
            int(ensemble.payload["window_id"]),
            levels,
            actual[:horizon],
        )
        if oracle:
            best = min(oracle.values())
            ensemble_regret = float(agent_wql - best)
            weights = ensemble.payload["weights"]
            selected = max(weights.items(), key=lambda kv: kv[1])[0] if weights else "none"
            if selected in oracle:
                selection_regret = float(oracle[selected] - best)
    return TrialResult(
        task_id=task["id"], trial=trial, success=success,
        chosen_model=result.chosen_model, unsupported_number_rate=unsupported,
        tool_call_validity_rate=validity, budget_adherent=budget_ok,
        recovery=recovery, selection_regret=selection_regret,
        ensemble_regret=ensemble_regret,
        latency_seconds=float(snap["elapsed_seconds"]), cost_usd=float(snap["cost_usd"]),
        calls=int(snap["calls"]), error=error,
    )


def run_agent_eval(
    tasks_path: Path | str = CONFIGS_DIR / "agent_tasks.yaml",
    trials: int = 3,
    limit: int | None = None,
    prefix: str = "agenteval",
    benchmark_run_id: str | None = None,
) -> dict[str, Any]:
    """Run every task ``trials`` times and score the harness."""
    payload = read_yaml(tasks_path, default=None) or {}
    tasks = list(payload.get("tasks", []))
    if limit:
        tasks = tasks[: int(limit)]
    if not tasks:
        raise FileNotFoundError(f"no tasks in {tasks_path}")
    run_id = benchmark_run_id or _benchmark_run_id(RESULTS_DIR / "summary.json")
    results = [
        _run_trial(task, trial, prefix, run_id)
        for task in tasks
        for trial in range(trials)
    ]
    frame = pd.DataFrame([r.__dict__ for r in results])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "agent_eval.csv", index=False)

    def _mean(column: str) -> float | None:
        values = frame[column].dropna()
        return float(values.mean()) if len(values) else None

    per_task = []
    for task_id, sub in frame.groupby("task_id"):
        per_task.append(
            {
                "task_id": task_id,
                "success_rate": float(sub["success"].mean()),
                "mean_selection_regret": float(sub["selection_regret"].dropna().mean())
                if sub["selection_regret"].notna().any() else None,
                "std_selection_regret": float(sub["selection_regret"].dropna().std(ddof=1))
                if sub["selection_regret"].notna().sum() > 1 else 0.0,
                "mean_latency_seconds": float(sub["latency_seconds"].mean()),
                "std_latency_seconds": float(sub["latency_seconds"].std(ddof=1))
                if len(sub) > 1 else 0.0,
                "mean_cost_usd": float(sub["cost_usd"].mean()),
                "std_cost_usd": float(sub["cost_usd"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "unsupported_number_rate": float(sub["unsupported_number_rate"].mean()),
            }
        )
    lat_var = run_to_run_variance(frame["latency_seconds"].tolist())
    cost_var = run_to_run_variance(frame["cost_usd"].tolist())
    regret_values = frame["selection_regret"].dropna().tolist()

    injected = frame[frame["recovery"].notna()]
    summary = {
        "run_id": prefix,
        "benchmark_run_id": run_id,
        "n_tasks": len(tasks),
        "trials_per_task": trials,
        "n_runs": len(results),
        "success_rate": float(frame["success"].mean()),
        "tool_call_validity_rate": float(frame["tool_call_validity_rate"].mean()),
        "unsupported_number_rate": float(frame["unsupported_number_rate"].mean()),
        "budget_adherence_rate": float(frame["budget_adherent"].mean()),
        "recovery_rate": float(injected["recovery"].mean()) if len(injected) else None,
        "n_failure_injection_runs": int(len(injected)),
        "selection_regret": float(np.mean(regret_values)) if regret_values else None,
        "selection_regret_run_to_run_std": float(
            np.mean([p["std_selection_regret"] for p in per_task
                     if p["std_selection_regret"] is not None])
        ) if any(p["std_selection_regret"] is not None for p in per_task) else None,
        "ensemble_regret": _mean("ensemble_regret"),
        "mean_latency_seconds": lat_var["mean"],
        "latency_run_to_run_std": lat_var["std"],
        "mean_cost_usd": cost_var["mean"],
        "cost_run_to_run_std": cost_var["std"],
        "mean_calls_per_task": float(frame["calls"].mean()),
        "n_oracle_scored_runs": int(frame["selection_regret"].notna().sum()),
        "per_task": per_task,
        "errors": frame[frame["error"] != ""]["error"].tolist()[:10],
        "offline": is_offline(),
    }
    (RESULTS_DIR / "agent_eval.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
