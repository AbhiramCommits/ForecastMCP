"""Thin DuckDB persistence layer for series metadata, runs, forecasts and metrics."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from forecastmcp.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    dataset VARCHAR,
    series_id VARCHAR,
    freq VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    n_obs BIGINT,
    target_mean DOUBLE,
    target_std DOUBLE,
    PRIMARY KEY (dataset, series_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    kind VARCHAR,
    dataset VARCHAR,
    model VARCHAR,
    config_json VARCHAR,
    git_sha VARCHAR
);
CREATE TABLE IF NOT EXISTS forecasts (
    run_id VARCHAR,
    dataset VARCHAR,
    model VARCHAR,
    series_id VARCHAR,
    window_id INTEGER,
    step INTEGER,
    ts TIMESTAMP,
    point DOUBLE,
    actual DOUBLE,
    quantiles VARCHAR,
    quantile_levels VARCHAR
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id VARCHAR,
    dataset VARCHAR,
    model VARCHAR,
    series_id VARCHAR,
    window_id INTEGER,
    metric VARCHAR,
    value DOUBLE,
    ci_low DOUBLE,
    ci_high DOUBLE
);
"""


class Store:
    """Small DuckDB wrapper used by the CLI, benchmark and MCP tools."""

    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.con = duckdb.connect(str(self.path))
        self.con.execute(SCHEMA)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def register_series(self, df: pd.DataFrame, freq: str) -> int:
        """Upsert per-series metadata derived from a long frame."""
        ts = pd.to_datetime(df["timestamp"])
        frame = df.assign(timestamp=ts)
        meta = (
            frame.groupby(frame["series_id"].astype(str))
            .agg(
                start_ts=("timestamp", "min"),
                end_ts=("timestamp", "max"),
                n_obs=("target", "size"),
                target_mean=("target", "mean"),
                target_std=("target", "std"),
            )
            .reset_index()
            .rename(columns={"series_id": "series_id"})
        )
        dataset = str(df["dataset"].iloc[0])
        rows = [
            (
                dataset,
                str(r.series_id),
                freq,
                r.start_ts.to_pydatetime(),
                r.end_ts.to_pydatetime(),
                int(r.n_obs),
                float(r.target_mean),
                float(r.target_std) if pd.notna(r.target_std) else 0.0,
            )
            for r in meta.itertuples()
        ]
        with self._lock:
            self.con.executemany(
                "INSERT OR REPLACE INTO series VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def create_run(
        self,
        run_id: str,
        kind: str,
        dataset: str | None = None,
        model: str | None = None,
        config: dict[str, Any] | None = None,
        git_sha: str | None = None,
    ) -> None:
        """Insert (or replace) a run record."""
        with self._lock:
            self.con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    pd.Timestamp.utcnow().to_pydatetime(),
                    kind,
                    dataset,
                    model,
                    json.dumps(config or {}, sort_keys=True, default=str),
                    git_sha,
                ],
            )

    def write_forecasts(self, df: pd.DataFrame) -> int:
        """Append forecast rows to the ``forecasts`` table."""
        if df.empty:
            return 0
        frame = df.copy()
        if "quantiles" in frame.columns and not isinstance(frame["quantiles"].iloc[0], str):
            frame["quantiles"] = frame["quantiles"].map(lambda v: json.dumps(list(v)))
        if "quantile_levels" in frame.columns and not isinstance(
            frame["quantile_levels"].iloc[0], str
        ):
            frame["quantile_levels"] = frame["quantile_levels"].map(
                lambda v: json.dumps(list(v))
            )
        frame["ts"] = pd.to_datetime(frame["ts"])
        cols = [
            "run_id",
            "dataset",
            "model",
            "series_id",
            "window_id",
            "step",
            "ts",
            "point",
            "actual",
            "quantiles",
            "quantile_levels",
        ]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"forecast frame missing columns: {missing}")
        frame = frame[cols]
        with self._lock:
            self.con.register("_forecast_batch", frame)
            self.con.execute("INSERT INTO forecasts SELECT * FROM _forecast_batch")
            self.con.unregister("_forecast_batch")
        return len(frame)

    def write_metrics(self, df: pd.DataFrame) -> int:
        """Append metric rows (optionally with bootstrap CIs)."""
        if df.empty:
            return 0
        frame = df.copy()
        for col in ("ci_low", "ci_high"):
            if col not in frame.columns:
                frame[col] = None
        cols = [
            "run_id",
            "dataset",
            "model",
            "series_id",
            "window_id",
            "metric",
            "value",
            "ci_low",
            "ci_high",
        ]
        frame = frame[cols]
        with self._lock:
            self.con.register("_metric_batch", frame)
            self.con.execute("INSERT INTO metrics SELECT * FROM _metric_batch")
            self.con.unregister("_metric_batch")
        return len(frame)

    def delete_run(self, run_id: str) -> None:
        """Remove all rows belonging to a run (used before re-running benchmarks)."""
        with self._lock:
            for table in ("forecasts", "metrics"):
                self.con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
            self.con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])

    # -- reads -------------------------------------------------------------
    def query(self, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
        """Execute SQL and return a DataFrame."""
        with self._lock:
            return self.con.execute(sql, list(params) if params else None).fetch_df()

    def series_frame(self, dataset: str | None = None) -> pd.DataFrame:
        """Return registered series metadata, optionally filtered by dataset."""
        if dataset:
            return self.query("SELECT * FROM series WHERE dataset = ? ORDER BY series_id", [dataset])
        return self.query("SELECT * FROM series ORDER BY dataset, series_id")

    def latest_metrics(self, dataset: str | None = None, run_id: str | None = None) -> pd.DataFrame:
        """Return metric rows, preferring the latest benchmark run when unspecified."""
        clauses, params = [], []
        if dataset:
            clauses.append("dataset = ?")
            params.append(dataset)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM metrics {where}", params)


_store: Store | None = None


def get_store() -> Store:
    """Process-wide store singleton (DuckDB is single-writer)."""
    global _store
    if _store is None:
        _store = Store(DB_PATH)
    return _store


def reset_store() -> None:
    """Drop the singleton (used by tests that point at a temporary database)."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
