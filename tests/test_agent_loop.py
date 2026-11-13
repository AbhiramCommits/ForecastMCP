"""Agent loop, fallback planner, context compaction and citation checking."""

from __future__ import annotations

import asyncio
import json

from forecastmcp.agent.loop import ContextWindow, ForecastAgent
from forecastmcp.agent.planner import (
    detect_horizon,
    detect_quantiles,
    make_plan,
    rule_based_plan,
)
from forecastmcp.agent.reconcile import combine
from forecastmcp.agent.report import (
    citation_coverage,
    unsupported_number_rate,
    unsupported_numbers,
)
from forecastmcp.mcp.budgets import BudgetExceeded, BudgetGuard
from forecastmcp.mcp.tools import ToolRuntime
from forecastmcp.obs.replay import replay


def test_rule_planner_parses_question():
    """The fallback planner extracts series, horizon and 80% quantiles."""
    plan = rule_based_plan(
        "What will capacity demand for S001 look like over the next 4 weeks, "
        "with an 80% interval?",
        "synthetic_supply",
        ["S001", "S002"],
    )
    assert plan.series_id == "S001"
    assert plan.horizon == 28
    assert plan.quantile_levels == [0.1, 0.5, 0.9]
    assert plan.source == "rules"
    assert len(plan.models) >= 2


def test_planner_horizon_and_quantile_parsing():
    """Weeks/days/months and 90% intervals are parsed deterministically."""
    assert detect_horizon("next 10 days") == 10
    assert detect_horizon("next 2 months") == 56
    assert detect_horizon("next fortnight") == 14
    assert detect_quantiles("95% interval") == [0.025, 0.5, 0.975]


def test_make_plan_falls_back_without_api_key(monkeypatch):
    """With no key, make_plan returns the deterministic rule-based plan."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = make_plan(
        "Forecast S002 for the next 2 weeks with an 80% interval",
        "synthetic_supply",
        ["S001", "S002"],
        ["seasonal_naive", "autotheta"],
    )
    assert plan.source == "rules"
    assert plan.series_id == "S002"
    assert len(plan.models) >= 2
    assert all(m in {"seasonal_naive", "autotheta"} for m in plan.models)


def test_reconcile_falls_back_without_scores():
    """Error-weighted reconciliation without metrics falls back and warns."""
    from forecastmcp.forecasters.base import ForecastResult

    def result(name, shift):
        return ForecastResult(
            series_id="S1", model_name=name, model_version=name, revision="1",
            point=[10 + shift] * 3, quantile_levels=[0.1, 0.5, 0.9],
            quantiles=[[8 + shift] * 3, [10 + shift] * 3, [12 + shift] * 3],
            horizon=3, latency_ms=1.0,
        )

    combined = combine([result("a", 0), result("b", 2)], method="inverse_error")
    assert combined.method == "vincentization"
    assert any("fell back" in w for w in combined.warnings)


def test_context_compaction_preserves_tail():
    """Compaction shrinks context while keeping the first and last calls."""
    ctx = ContextWindow(max_tokens=50)
    for i in range(6):
        ctx.add("tool", {"call_id": f"c{i}", "blob": "x" * 400})
    assert ctx.compactions >= 1
    assert ctx.entries[0]["call_id"] == "c0"
    assert ctx.entries[-1]["call_id"] == "c5"
    assert any(e.get("compacted") for e in ctx.entries)


class _FakeClient:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = []

    async def call(self, tool, arguments):
        self.calls.append((tool, dict(arguments)))
        if len(self.calls) == 1:
            raise self.exc
        return {"call_id": f"c{len(self.calls)}", "model": arguments.get("model")}


def test_agent_downgrades_on_budget_error(tmp_path):
    """A BudgetExceeded triggers a cheaper-model retry and a downgrade span."""
    runtime = ToolRuntime(run_id="downgrade", base_dir=tmp_path, guard=BudgetGuard())
    agent = ForecastAgent(runtime=runtime)
    client = _FakeClient(BudgetExceeded("max_cost", "too expensive"))
    payload = asyncio.run(
        agent._call(client, "forecast", {"model": "chronos"}, ContextWindow(), [])
    )
    assert payload["model"] == "seasonal_naive"
    assert client.calls[-1][1]["model"] == "seasonal_naive"
    assert agent._downgrades == 1
    runtime.close()
    names = [json.loads(line)["name"] for line in
             (tmp_path / "downgrade" / "trace.jsonl").read_text().splitlines()]
    assert "agent.downgrade" in names


def test_agent_retries_after_tool_failure(tmp_path):
    """One injected tool failure is retried and counted as a recovery."""
    from forecastmcp.agent.loop import ToolCallFailure

    runtime = ToolRuntime(run_id="retry", base_dir=tmp_path, guard=BudgetGuard())
    agent = ForecastAgent(runtime=runtime)
    client = _FakeClient(ToolCallFailure("injected"))
    payload = asyncio.run(
        agent._call(client, "backtest", {"model": "seasonal_naive"}, ContextWindow(), [])
    )
    assert payload["call_id"] == "c2"
    assert agent._recoveries == 1
    runtime.close()


def test_citation_checker_catches_planted_number(tmp_path):
    """A fabricated number is flagged by the unsupported-number check."""
    runtime = ToolRuntime(run_id="cite", base_dir=tmp_path, guard=BudgetGuard(max_calls=5))
    payload = runtime.execute("list_series", {"dataset": "synthetic_supply",
                                              "filter": {"limit": 1}})
    runtime.close()
    records = [{"tool": "list_series", "call_id": payload["call_id"], "payload": payload}]
    honest = f"The dataset has {payload['total_series']} series [call:{payload['call_id']}]."
    assert unsupported_numbers(honest, records) == []
    planted = honest + " Forecast error is 12345.678 [call:c0000]."
    flagged = unsupported_numbers(planted, records)
    assert 12345.678 in flagged
    assert unsupported_number_rate(planted, records) > 0


def test_full_agent_run_writes_cited_report(tmp_path, monkeypatch):
    """The full loop runs end-to-end with baselines and produces a replayable run."""
    from forecastmcp.agent import loop as loop_module
    from forecastmcp.agent.planner import Plan

    plan = Plan(
        question="q",
        dataset="synthetic_supply",
        series_id="S001",
        horizon=7,
        quantile_levels=[0.1, 0.5, 0.9],
        models=["seasonal_naive", "autotheta"],
        method="vincentization",
        window_id=6,
        source="rules",
    )
    monkeypatch.setattr(loop_module, "make_plan", lambda *a, **k: plan)
    runtime = ToolRuntime(
        run_id="agent-e2e",
        base_dir=tmp_path,
        guard=BudgetGuard(max_calls=20, max_wall_clock_seconds=300),
    )
    agent = ForecastAgent(runtime=runtime, prefer_llm=False)
    result = agent.run_sync("Forecast S001 for the next week")
    runtime.close()

    assert result.report_path.exists()
    assert result.report_path.name == "report.md"
    assert result.chosen_model.startswith("ensemble:")
    assert len(result.calls) == 7
    records = [{"tool": c.tool, "call_id": c.call_id, "payload": c.payload} for c in result.calls]
    assert unsupported_number_rate(result.report, records) == 0.0
    assert "[call:" in result.report
    assert citation_coverage(result.report, records) == 1.0

    report = replay("agent-e2e", base_dir=tmp_path)
    assert report.ok, report.details
    assert report.matched >= 7
