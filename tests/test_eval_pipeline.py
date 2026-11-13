"""End-to-end evaluation pipeline: benchmark, plots, summary and agent scoring."""

from __future__ import annotations

import json

import pytest
import yaml

from forecastmcp.agent import loop as loop_module
from forecastmcp.agent.planner import Plan
from forecastmcp.data.store import Store
from forecastmcp.eval import agent_eval, plots, runner
from forecastmcp.eval import summary as summary_module
from forecastmcp.mcp import tools as tools_module


@pytest.fixture()
def temp_eval(tmp_path, monkeypatch):
    store = Store(tmp_path / "eval.duckdb")
    monkeypatch.setattr(runner, "get_store", lambda: store)
    monkeypatch.setattr(tools_module, "get_store", lambda: store)
    monkeypatch.setattr(agent_eval, "get_store", lambda: store)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(plots, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(summary_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(summary_module, "FINETUNE_DIR", tmp_path / "finetune")
    monkeypatch.setattr(agent_eval, "RESULTS_DIR", tmp_path)
    yield tmp_path, store
    store.close()


def test_full_eval_pipeline(temp_eval, monkeypatch):
    """A tiny benchmark feeds plots, summary and agent scoring end-to-end."""
    tmp_path, store = temp_eval
    bench = runner.run_benchmark(
        run_id="test-bench",
        datasets=["synthetic_supply"],
        models=["seasonal_naive", "autotheta"],
        max_series=3,
        n_windows=1,
        n_resamples=80,
    )
    dataset = bench["datasets"]["synthetic_supply"]
    assert "seasonal_naive" in dataset and "autotheta" in dataset
    assert "ensemble_vincentization" in dataset
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "per_series_metrics.csv").exists()
    assert (tmp_path / "wql_by_horizon.csv").exists()
    assert (tmp_path / "model_summary.csv").exists()

    figure_paths = plots.make_all_plots(out_dir=tmp_path / "figures")
    assert figure_paths
    assert all(path.exists() for path in figure_paths)
    assert all(path.suffix == ".png" for path in figure_paths)

    plan = Plan(
        question="q",
        dataset="synthetic_supply",
        series_id="S001",
        horizon=7,
        quantile_levels=[0.1, 0.5, 0.9],
        models=["seasonal_naive", "autotheta"],
        method="inverse_error",
        window_id=1,
        source="rules",
    )
    monkeypatch.setattr(loop_module, "make_plan", lambda *a, **k: plan)
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "id": "t1",
                        "question": "Forecast S001 for one week with an 80% interval at split 1.",
                        "dataset": "synthetic_supply",
                        "series_id": "S001",
                        "horizon": 7,
                        "quantiles": [0.1, 0.5, 0.9],
                        "window_id": 1,
                        "inject_failure": True,
                    }
                ]
            }
        )
    )
    agent_summary = agent_eval.run_agent_eval(
        tasks_path=tasks_file,
        trials=1,
        prefix="test-agent",
        benchmark_run_id="test-bench",
    )
    assert agent_summary["success_rate"] == 1.0
    assert agent_summary["tool_call_validity_rate"] == 1.0
    assert agent_summary["unsupported_number_rate"] == 0.0
    assert agent_summary["budget_adherence_rate"] == 1.0
    assert agent_summary["recovery_rate"] == 1.0
    assert agent_summary["n_oracle_scored_runs"] == 1
    assert agent_summary["selection_regret"] is not None

    summary_path, negative_path = summary_module.write_outputs()
    text = summary_path.read_text()
    negative = negative_path.read_text()
    assert "Benchmark run id" in text
    assert "Agent harness metrics" in text
    assert "tool-call validity rate" in text
    assert (tmp_path / "agent_eval.json").exists()
    assert (tmp_path / "agent_eval.csv").exists()
    assert "Negative results" in negative
    assert "Unavailable models" in negative
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["run_id"] == "test-bench"
