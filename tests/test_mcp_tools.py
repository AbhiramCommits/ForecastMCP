"""MCP tool schemas, read-only SQL guard and permission denial."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from forecastmcp.mcp.budgets import BudgetGuard
from forecastmcp.mcp.permissions import PermissionDenied, check_permission
from forecastmcp.mcp.schemas import BacktestInput, ForecastInput, ForecastOutput
from forecastmcp.mcp.tools import ToolRuntime, ensure_read_only


@pytest.fixture()
def runtime(tmp_path):
    rt = ToolRuntime(run_id="test-mcp", role="analyst", base_dir=tmp_path,
                     guard=BudgetGuard(max_calls=20))
    yield rt
    rt.close()


def test_list_series_and_describe(runtime):
    """Read tools return schema-valid payloads."""
    listing = runtime.execute("list_series", {"dataset": "synthetic_supply",
                                              "filter": {"limit": 3}})
    assert listing["count"] == 3
    assert listing["total_series"] == 200
    described = runtime.execute(
        "describe_series", {"series_id": listing["series"][0]["series_id"]}
    )
    assert described["series"]["n_obs"] > 100
    assert "promo" in described["covariates"]


def test_forecast_output_schema(runtime):
    """Forecast payload validates against the declared Pydantic output model."""
    payload = runtime.execute(
        "forecast",
        {"model": "seasonal_naive", "series_id": "S001", "horizon": 7,
         "window_id": 6},
    )
    model = ForecastOutput.model_validate(payload)
    assert model.forecast.horizon == 7
    assert len(model.forecast.quantiles) == len(model.forecast.quantile_levels)
    assert model.call_id == payload["call_id"]


def test_malformed_input_is_rejected():
    """Pydantic rejects invalid horizons and mismatched weights."""
    with pytest.raises(ValidationError):
        ForecastInput(model="seasonal_naive", series_id="S001", horizon=0)
    from forecastmcp.mcp.schemas import EnsembleInput

    with pytest.raises(ValidationError):
        EnsembleInput(models=["a", "b"], series_id="S1", weights=[1.0])


def test_sql_guard_accepts_select_and_rejects_writes():
    """Only a single SELECT/WITH statement may run."""
    assert ensure_read_only("SELECT 1").startswith("SELECT")
    assert ensure_read_only("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")
    for bad in (
        "DROP TABLE series",
        "DELETE FROM series",
        "SELECT 1; DROP TABLE series",
        "PRAGMA database_list",
        "ATTACH 'x.duckdb' AS x",
        "COPY series TO 'x.csv'",
    ):
        with pytest.raises(ValueError):
            ensure_read_only(bad)


def test_sql_tool_runs_read_only_query(runtime):
    """The SQL tool returns rows and caps output at 500."""
    out = runtime.execute("sql", {"query": "SELECT count(*) AS n FROM series"})
    assert out["row_count"] == 1
    assert out["columns"] == ["n"]
    with pytest.raises(ValueError):
        runtime.execute("sql", {"query": "DELETE FROM series"})


def test_permission_denial_is_logged(tmp_path):
    """A role outside the allow-list is denied and the denial is traced."""
    rt = ToolRuntime(run_id="test-perm", role="intern", base_dir=tmp_path)
    try:
        with pytest.raises(PermissionDenied):
            rt.execute("list_series", {"dataset": "synthetic_supply"})
    finally:
        rt.close()
    lines = (tmp_path / "test-perm" / "trace.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert any(r["name"] == "permission.denied" for r in records)
    assert any(r["attributes"].get("call.success") is False for r in records)
    with pytest.raises(PermissionDenied):
        check_permission("analyst", "finetune")
    check_permission("admin", "finetune")


def test_ensemble_returns_members_and_divergence(runtime):
    """Ensemble reconciliation reports weights, rule and pairwise divergence."""
    out = runtime.execute(
        "ensemble_forecast",
        {
            "models": ["seasonal_naive", "autotheta"],
            "series_id": "S001",
            "horizon": 7,
            "window_id": 6,
            "method": "vincentization",
        },
    )
    assert set(out["members"]) == {"seasonal_naive", "autotheta"}
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-6
    assert "mean_pairwise" in out["divergence"]
    assert out["rule_fired"] == "equal_weight_vincentization"


def test_backtest_returns_metrics(runtime):
    """Backtest uses the frozen windows and returns probabilistic metrics."""
    out = runtime.execute(
        "backtest",
        {"model": "seasonal_naive", "series_id": "S001", "n_windows": 2},
    )
    assert out["n_windows"] == 2
    assert set(out["metrics"]) >= {"WQL", "CRPS", "MASE", "coverage_80"}
    assert len(out["per_window"]) == 2
    BacktestInput(model="seasonal_naive", series_id="S001", n_windows=2)
