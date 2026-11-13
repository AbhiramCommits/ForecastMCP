"""Tool implementations and the traced, budgeted, permissioned runtime around them.

The MCP server and the in-process agent both call :class:`ToolRuntime.execute`,
so validation, budgets, permissions, tracing and replay cover both entry points.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecastmcp.agent.reconcile import combine
from forecastmcp.config import CONFIGS_DIR, read_yaml
from forecastmcp.data.loaders import load_dataset
from forecastmcp.data.requests import build_window_request
from forecastmcp.data.splits import holdout_windows
from forecastmcp.data.store import get_store
from forecastmcp.eval.metrics import (
    crps_from_quantiles,
    interval_coverage,
    mase,
    smape,
    weighted_quantile_loss,
    winkler_score,
)
from forecastmcp.forecasters.base import ModelUnavailable
from forecastmcp.forecasters.registry import (
    configured_cost,
    estimate_cost,
    get_forecaster,
    model_card,
)
from forecastmcp.mcp.budgets import BudgetExceeded, BudgetGuard
from forecastmcp.mcp.permissions import PermissionDenied, check_permission
from forecastmcp.mcp.schemas import (
    INPUT_MODELS,
    OUTPUT_MODELS,
    SeriesFilter,
    validate_input,
)
from forecastmcp.obs.tracing import RunTrace, canonical_json, sha256_text

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|"
    r"pragma|install|load|call|set|truncate|merge|vacuum|replace)\b",
    re.IGNORECASE,
)


class ToolFailure(RuntimeError):
    """A typed, user-facing tool failure (bad series, unavailable model, ...)."""


def strip_sql_comments(query: str) -> str:
    """Remove SQL line and block comments."""
    query = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", query)


def ensure_read_only(query: str) -> str:
    """Validate that a SQL string is a single read-only SELECT/WITH query.

    Raises ``ValueError`` for anything else. This is a defence-in-depth text
    guard on top of the tool only ever executing against the local metrics DB.
    """
    cleaned = strip_sql_comments(query).strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("empty SQL query")
    if ";" in cleaned:
        raise ValueError("multiple SQL statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise ValueError("only SELECT (or WITH ... SELECT) queries are allowed")
    unquoted = re.sub(r"'(\\.|[^'])*'", "''", cleaned)
    match = FORBIDDEN_SQL.search(unquoted)
    if match:
        raise ValueError(f"forbidden SQL keyword: {match.group(0).upper()}")
    return cleaned


def season_length(freq: str) -> int:
    """Season length used for MASE scaling: 24 hourly, 7 daily."""
    return 24 if freq.lower().startswith("h") else 7


class ForecastTools:
    """Pure tool implementations over the local data store."""

    def __init__(self) -> None:
        self._data: dict[str, pd.DataFrame] = {}

    def data(self, dataset: str) -> pd.DataFrame:
        """Load a dataset once per process."""
        if dataset not in self._data:
            self._data[dataset] = load_dataset(dataset)
        return self._data[dataset]

    # -- series inspection -------------------------------------------------
    def list_series(self, dataset: str, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """List series metadata with optional prefix/category filters."""
        spec = SeriesFilter.model_validate(filter or {})
        df = self.data(dataset).sort_values(["series_id", "timestamp"])
        g = df.groupby("series_id", observed=True)
        meta = g.agg(
            n_obs=("target", "size"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            mean=("target", "mean"),
            std=("target", "std"),
            last_value=("target", "last"),
        ).reset_index()
        if "category" in df.columns:
            meta["category"] = g["category"].first().values
        else:
            meta["category"] = None
        if spec.prefix:
            meta = meta[meta["series_id"].astype(str).str.startswith(spec.prefix)]
        if spec.category:
            meta = meta[meta["category"] == spec.category]
        total = len(meta)
        page = meta.iloc[spec.offset : spec.offset + spec.limit]
        rows = []
        for row in page.itertuples():
            rows.append(
                {
                    "series_id": str(row.series_id),
                    "n_obs": int(row.n_obs),
                    "start": str(pd.Timestamp(row.start).isoformat()),
                    "end": str(pd.Timestamp(row.end).isoformat()),
                    "mean": float(row.mean),
                    "std": float(row.std) if pd.notna(row.std) else 0.0,
                    "last_value": float(row.last_value),
                    "category": None if row.category is None or pd.isna(row.category) else str(row.category),
                }
            )
        return {"dataset": dataset, "count": len(rows), "total_series": total, "series": rows}

    def _series_frame(self, dataset: str, series_id: str) -> pd.DataFrame:
        df = self.data(dataset)
        series = df[df["series_id"].astype(str) == str(series_id)].sort_values("timestamp")
        if series.empty:
            raise ToolFailure(f"series_id {series_id!r} not found in dataset {dataset!r}")
        return series

    @staticmethod
    def _summary(series: pd.DataFrame, series_id: str) -> dict[str, Any]:
        category = None
        if "category" in series.columns and series["category"].notna().any():
            category = str(series["category"].iloc[0])
        return {
            "series_id": str(series_id),
            "n_obs": int(len(series)),
            "start": str(pd.Timestamp(series["timestamp"].min()).isoformat()),
            "end": str(pd.Timestamp(series["timestamp"].max()).isoformat()),
            "mean": float(series["target"].mean()),
            "std": float(series["target"].std() or 0.0),
            "last_value": float(series["target"].iloc[-1]),
            "category": category,
        }

    def describe_series(self, series_id: str, dataset: str = "synthetic_supply") -> dict[str, Any]:
        """Describe one series: stats, covariates, recent values and holdout windows."""
        series = self._series_frame(dataset, series_id)
        covariates = [
            c
            for c in series.columns
            if c not in ("dataset", "series_id", "timestamp", "target")
        ]
        recent = []
        for row in series.tail(5).to_dict("records"):
            recent.append(
                {
                    "timestamp": str(pd.Timestamp(row["timestamp"]).date()),
                    "target": float(row["target"]),
                    "promo": float(row.get("promo", 0.0)),
                }
            )
        windows = [
            {
                "window_id": w.window_id,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
            }
            for w in holdout_windows(dataset)
        ]
        return {
            "dataset": dataset,
            "series": self._summary(series, series_id),
            "covariates": covariates,
            "recent": recent,
            "holdout_windows": windows,
        }

    # -- forecasting -------------------------------------------------------
    def _request(self, dataset: str, series_id: str, horizon: int, window_id: int | None,
                 quantiles: list[float] | None, covariates: bool):
        return build_window_request(
            self.data(dataset),
            dataset,
            series_id,
            horizon=horizon,
            window_id=window_id,
            covariates=covariates,
            quantiles=quantiles,
        )

    def forecast(
        self,
        model: str,
        series_id: str,
        horizon: int = 28,
        quantiles: list[float] | None = None,
        covariates: bool = True,
        dataset: str = "synthetic_supply",
        window_id: int | None = None,
    ) -> dict[str, Any]:
        """Run one forecaster at a frozen origin and attach ground truth."""
        wr = self._request(dataset, series_id, horizon, window_id, quantiles, covariates)
        try:
            forecaster = get_forecaster(model)
        except (KeyError, RuntimeError) as exc:
            raise ToolFailure(str(exc)) from exc
        ok, reason = forecaster.available()
        if not ok:
            raise ToolFailure(f"model {model!r} unavailable: {reason}")
        try:
            result = forecaster.predict(wr.request)
        except ModelUnavailable as exc:
            raise ToolFailure(f"model {model!r} failed at inference: {exc}") from exc
        matrix = np.asarray(result.quantiles, dtype=float)
        lo80, hi80 = result.lower_upper(0.8)
        return {
            "dataset": dataset,
            "series_id": str(series_id),
            "window_id": wr.window.window_id,
            "forecast": result.model_dump(mode="json"),
            "actual": wr.actual,
            "dates": wr.dates,
            "cost_usd": configured_cost(model),
            "summary": {
                "mean_point": float(np.mean(result.point)),
                "last_point": float(result.point[-1]),
                "p80_low_mean": float(np.mean(lo80)),
                "p80_high_mean": float(np.mean(hi80)),
                "median_mean": float(np.mean(matrix[result.quantile_levels.index(0.5)]))
                if 0.5 in result.quantile_levels
                else float(np.mean(result.point)),
            },
        }

    def ensemble_forecast(
        self,
        models: list[str],
        series_id: str,
        horizon: int = 28,
        quantiles: list[float] | None = None,
        method: str = "vincentization",
        weights: list[float] | None = None,
        dataset: str = "synthetic_supply",
        window_id: int | None = None,
        covariates: bool = True,
    ) -> dict[str, Any]:
        """Forecast with several models and reconcile their quantiles."""
        started = time.perf_counter()
        wr = self._request(dataset, series_id, horizon, window_id, quantiles, covariates)
        results = []
        for model in models:
            try:
                forecaster = get_forecaster(model)
            except (KeyError, RuntimeError) as exc:
                raise ToolFailure(str(exc)) from exc
            ok, reason = forecaster.available()
            if not ok:
                raise ToolFailure(f"model {model!r} unavailable: {reason}")
            try:
                results.append(forecaster.predict(wr.request))
            except ModelUnavailable as exc:
                raise ToolFailure(f"model {model!r} failed: {exc}") from exc
        scores = None
        if method in {"inverse_error", "best_on_backtest"}:
            scores = self._backtest_scores(dataset, series_id, models)
        combined = combine(results, method=method, weights=weights, backtest_scores=scores)
        latency = (time.perf_counter() - started) * 1000.0
        clevels = np.asarray(combined.quantile_levels)
        cq = np.asarray(combined.quantiles)
        lo_idx = int(np.argmin(np.abs(clevels - 0.1)))
        hi_idx = int(np.argmin(np.abs(clevels - 0.9)))
        return {
            "dataset": dataset,
            "series_id": str(series_id),
            "window_id": wr.window.window_id,
            "model_name": f"ensemble:{combined.rule_fired}",
            "method": method,
            "rule_fired": combined.rule_fired,
            "horizon": wr.horizon,
            "point": combined.point,
            "quantile_levels": combined.quantile_levels,
            "quantiles": combined.quantiles,
            "members": combined.members,
            "weights": combined.weights,
            "divergence": combined.divergence,
            "used_covariates": any(r.used_covariates for r in results),
            "warnings": combined.warnings,
            "latency_ms": round(latency, 3),
            "cost_usd": round(sum(configured_cost(m) for m in models), 8),
            "dates": wr.dates,
            "summary": {
                "mean_point": float(np.mean(combined.point)),
                "last_point": float(combined.point[-1]),
                "p80_low_mean": float(np.mean(cq[lo_idx])),
                "p80_high_mean": float(np.mean(cq[hi_idx])),
            },
        }

    def _backtest_scores(
        self, dataset: str, series_id: str, models: list[str]
    ) -> dict[str, float] | None:
        try:
            frame = get_store().query(
                "SELECT model, AVG(value) AS wql FROM metrics "
                "WHERE dataset = ? AND series_id = ? AND metric = 'WQL' "
                "AND run_id = (SELECT run_id FROM runs WHERE kind = 'benchmark' "
                "ORDER BY created_at DESC LIMIT 1) "
                "GROUP BY model",
                [dataset, series_id],
            )
        except Exception:
            return None
        if frame.empty:
            return None
        return {str(r.model): float(r.wql) for r in frame.itertuples() if r.model in models}

    def backtest(
        self,
        model: str,
        series_id: str,
        n_windows: int = 6,
        dataset: str = "synthetic_supply",
    ) -> dict[str, Any]:
        """Backtest one model on the frozen rolling-origin windows for one series."""
        windows = holdout_windows(dataset)[-int(n_windows) :]
        try:
            forecaster = get_forecaster(model)
        except (KeyError, RuntimeError) as exc:
            raise ToolFailure(str(exc)) from exc
        ok, reason = forecaster.available()
        if not ok:
            raise ToolFailure(f"model {model!r} unavailable: {reason}")
        per_window = []
        collected: dict[str, list[float]] = {
            "WQL": [],
            "CRPS": [],
            "MASE": [],
            "coverage_80": [],
            "coverage_90": [],
            "sMAPE": [],
            "winkler_80": [],
        }
        eval_quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for window in windows:
            wr = self._request(
                dataset, series_id, window.horizon, window.window_id, eval_quantiles, True
            )
            try:
                res = forecaster.predict(wr.request)
            except ModelUnavailable as exc:
                raise ToolFailure(f"model {model!r} failed: {exc}") from exc
            q = np.asarray(res.quantiles)
            y = np.asarray(wr.actual)
            m = season_length(wr.window.freq)
            levels = res.quantile_levels
            lo80, hi80 = res.lower_upper(0.8)
            lo90, hi90 = res.lower_upper(0.9)
            row = {
                "window_id": window.window_id,
                "WQL": weighted_quantile_loss(y, q, levels),
                "CRPS": crps_from_quantiles(y, q, levels),
                "MASE": mase(y, res.point, wr.request.context, m),
                "coverage_80": interval_coverage(y, lo80, hi80),
                "coverage_90": interval_coverage(y, lo90, hi90),
                "sMAPE": smape(y, res.point),
                "winkler_80": winkler_score(y, lo80, hi80, 0.8),
            }
            per_window.append(row)
            for key in collected:
                collected[key].append(row[key])
        metrics = {
            key: {"value": float(np.mean(vals)), "ci_low": None, "ci_high": None}
            for key, vals in collected.items()
        }
        return {
            "dataset": dataset,
            "model": model,
            "series_id": str(series_id),
            "n_windows": len(windows),
            "metrics": metrics,
            "per_window": per_window,
            "cost_usd": round(configured_cost(model) * len(windows), 8),
        }

    def model_card(self, model: str) -> dict[str, Any]:
        """Return the registry model card."""
        try:
            return {"card": model_card(model)}
        except KeyError as exc:
            raise ToolFailure(str(exc)) from exc

    def sql(self, query: str) -> dict[str, Any]:
        """Execute a guarded read-only SQL query with a hard row cap."""
        cleaned = ensure_read_only(query)
        frame = get_store().query(f"SELECT * FROM ({cleaned}) AS _sub LIMIT 500")
        frame = frame.where(pd.notna(frame), None)
        rows = [
            [str(v) if isinstance(v, (pd.Timestamp, datetime)) else v for v in row]
            for row in frame.itertuples(index=False)
        ]
        return {"columns": [str(c) for c in frame.columns], "rows": rows, "row_count": len(rows)}


def stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace wall-clock-dependent fields so replay can compare bytes exactly."""
    volatile = {"latency_ms": 0.0}

    def scrub(value: Any) -> Any:
        """Recursively replace wall-clock fields with deterministic zeros."""
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in volatile or key.endswith("_seconds") or key == "batch_seconds":
                    out[key] = 0.0
                else:
                    out[key] = scrub(item)
            return out
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return scrub(payload)


def _model_for_cost(tool: str, arguments: dict[str, Any]) -> str | None:
    if tool in {"forecast", "backtest"}:
        return str(arguments.get("model"))
    if tool == "ensemble_forecast":
        return None
    return None


class ToolRuntime:
    """Validation + permissions + budgets + OTel trace + output persistence."""

    def __init__(
        self,
        run_id: str | None = None,
        role: str = "analyst",
        trace: RunTrace | None = None,
        guard: BudgetGuard | None = None,
        tools: ForecastTools | None = None,
        base_dir: Path | str | None = None,
    ):
        self.run_id = run_id or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"
        self.role = role
        self.trace = trace or RunTrace(self.run_id, base_dir=base_dir)
        budget_cfg = read_yaml(CONFIGS_DIR / "costs.yaml", default={}) or {}
        if not budget_cfg:
            budget_cfg = {"budgets": {}}
        cfg = budget_cfg.get("budgets", {})
        self.guard = guard or BudgetGuard(
            max_calls=int(cfg.get("max_calls_per_run", 40)),
            max_wall_clock_seconds=float(cfg.get("max_wall_clock_seconds", 900)),
            max_cost_usd=float(cfg.get("max_cost_usd", 0.5)),
            per_call_timeout_seconds=float(cfg.get("per_call_timeout_seconds", 120)),
        )
        self.tools = tools or ForecastTools()
        self._counter = 0
        self.faults: dict[str, list[str]] = {}
        self.compactions = 0
        self.attempts = 0
        self.validation_failures = 0

    # -- fault injection for recovery evaluation ---------------------------
    def inject_failure(self, tool: str, message: str = "injected tool failure") -> None:
        """Make the next call to ``tool`` fail once (used by agent evaluation)."""
        self.faults.setdefault(tool, []).append(message)

    # -- main entry --------------------------------------------------------
    def execute(
        self,
        tool: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
        on_span: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        """Validate, authorize, budget, execute and trace one tool call."""
        if tool not in INPUT_MODELS:
            raise KeyError(f"unknown tool {tool!r}")
        self.attempts += 1
        try:
            validated = validate_input(tool, arguments)
        except Exception:
            self.validation_failures += 1
            raise
        args = validated.model_dump(mode="json")
        if call_id is None:
            call_id = f"c{self._counter:04d}"
        self._counter += 1
        input_json = canonical_json(args)
        input_hash = sha256_text(input_json)
        span_attrs = {
            "tool.name": tool,
            "call.id": call_id,
            "call.input_hash": input_hash,
            "call.input_json": input_json,
            "tool.role": self.role,
            "run.id": self.run_id,
            "dataset": args.get("dataset"),
            "model.name": args.get("model"),
        }
        started = time.perf_counter()
        with self.trace.span(f"tool.{tool}", span_attrs) as span:
            try:
                check_permission(self.role, tool)
            except PermissionDenied as exc:
                self.trace.event("permission.denied", {"role": self.role, "tool": tool})
                span.set_attribute("call.success", False)
                span.set_attribute("call.error", str(exc))
                raise
            model = _model_for_cost(tool, args)
            pre_cost = configured_cost(model) if model else 0.0
            try:
                self.guard.check(pre_cost)
            except BudgetExceeded as exc:
                self.trace.event(
                    "budget.exceeded",
                    {"reason": exc.reason, "tool": tool, "call.id": call_id},
                )
                span.set_attribute("call.success", False)
                span.set_attribute("call.error", str(exc))
                raise
            faults = self.faults.get(tool)
            if faults:
                message = faults.pop(0)
                span.set_attribute("call.success", False)
                span.set_attribute("call.error", message)
                span.set_attribute("call.injected_failure", True)
                raise RuntimeError(f"injected failure in {tool}: {message}")

            try:
                def _invoke() -> dict[str, Any]:
                    return getattr(self.tools, tool)(**args)

                payload = self.guard.run(_invoke)
            except Exception as exc:
                span.set_attribute("call.success", False)
                span.set_attribute("call.error", f"{type(exc).__name__}: {exc}")
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latency_ms = float(payload.get("latency_ms") or elapsed_ms)
            device = str((payload.get("forecast") or {}).get("device", "cpu"))
            cost = estimate_cost(model, latency_ms, device) if model else 0.0
            if tool == "ensemble_forecast":
                cost = float(payload.get("cost_usd", 0.0))
            self.guard.record(cost)
            payload = dict(payload)
            payload["call_id"] = call_id
            stable = stable_payload(payload)
            path, digest = self.trace.write_call_output(call_id, stable)
            span.set_attribute("call.latency_ms", round(elapsed_ms, 3))
            span.set_attribute("call.cost_usd", round(cost, 8))
            span.set_attribute("call.success", True)
            span.set_attribute("call.output_path", str(path))
            span.set_attribute("call.output_sha256", digest)
            revision = None
            if isinstance(payload.get("forecast"), dict):
                revision = payload["forecast"].get("revision")
            elif isinstance(payload.get("card"), dict):
                revision = payload["card"].get("revision")
            if revision:
                span.set_attribute("model.revision", str(revision))
            if on_span is not None:
                on_span(span)
            return payload

    def close(self) -> None:
        """Flush traces."""
        self.trace.close()


def output_model(tool: str, payload: dict[str, Any]):
    """Validate a runtime payload against its declared Pydantic output schema."""
    if tool not in OUTPUT_MODELS:
        raise KeyError(f"unknown tool {tool!r}")
    return OUTPUT_MODELS[tool].model_validate(payload)


