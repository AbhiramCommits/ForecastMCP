"""MCP-client agent loop with context compaction and error recovery."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from forecastmcp.agent.planner import Plan, detect_dataset, make_plan
from forecastmcp.agent.report import build_report, write_report
from forecastmcp.forecasters.registry import availability, available_names
from forecastmcp.mcp.budgets import BudgetExceeded
from forecastmcp.mcp.permissions import PermissionDenied
from forecastmcp.mcp.server import build_server
from forecastmcp.mcp.tools import ToolRuntime


class ToolCallFailure(RuntimeError):
    """An MCP tool returned ``is_error``; recovered or surfaced by the agent."""


def error_from_text(tool: str, text: str) -> Exception:
    """Reconstruct a typed exception from the MCP error text."""
    if "BudgetExceeded[" in text:
        start = text.index("BudgetExceeded[") + len("BudgetExceeded[")
        end = text.index("]", start)
        return BudgetExceeded(text[start:end], text)
    if "PermissionDenied" in text:
        return PermissionDenied("unknown", tool)
    return ToolCallFailure(f"{tool} failed: {text}")


class McpToolSession:
    """Async MCP client bound to an in-memory server with the given runtime."""

    def __init__(self, runtime: ToolRuntime):
        self.runtime = runtime
        self.server = None
        self.session: ClientSession | None = None
        self._cm = None
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> McpToolSession:
        self.server, self.runtime = build_server(runtime=self.runtime)
        self._cm = create_client_server_memory_streams()
        client_streams, server_streams = await self._cm.__aenter__()
        low = self.server._lowlevel_server
        self._task = asyncio.create_task(
            low.run(server_streams[0], server_streams[1], low.create_initialization_options())
        )
        self.session = ClientSession(client_streams[0], client_streams[1])
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.session is not None:
            await self.session.__aexit__(*exc)  # type: ignore[arg-type]
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        if self._cm is not None:
            await self._cm.__aexit__(*exc)  # type: ignore[arg-type]

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool and return its structured payload."""
        if self.session is None:
            raise RuntimeError("session not started")
        result = await self.session.call_tool(tool, arguments)
        if result.is_error:
            text = " ".join(
                getattr(item, "text", "") for item in result.content if getattr(item, "text", None)
            )
            raise error_from_text(tool, text or f"{tool} returned an error")
        if result.structured_content is not None:
            return dict(result.structured_content)
        text = "".join(
            getattr(item, "text", "") for item in result.content if getattr(item, "text", None)
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - structured output is configured
            raise ToolCallFailure(f"{tool} returned non-JSON content: {text[:200]}") from exc


class ContextWindow:
    """Approximate-token context with compaction of older tool outputs."""

    def __init__(self, max_tokens: int = 2500):
        self.max_tokens = int(max_tokens)
        self.entries: list[dict[str, Any]] = []
        self.compactions = 0

    def tokens(self) -> int:
        """Rough token estimate (4 characters per token)."""
        return sum(len(json.dumps(e, default=str)) // 4 for e in self.entries)

    def add(self, tool: str, payload: dict[str, Any]) -> tuple[int, int, bool]:
        """Add a tool output and compact if over budget. Returns (before, after, compacted)."""
        self.entries.append({"tool": tool, "call_id": payload.get("call_id"), "payload": payload})
        before = self.tokens()
        compacted = False
        if before > self.max_tokens and len(self.entries) > 3:
            self.compact()
            compacted = True
            self.compactions += 1
        return before, self.tokens(), compacted

    def compact(self) -> None:
        """Replace the middle of the context with a compact summary."""
        first = self.entries[0]
        tail = self.entries[-2:]
        middle = self.entries[1:-2]
        summary = {
            "tool": "compacted_context",
            "compacted": True,
            "n_entries": len(middle),
            "tools": [e["tool"] for e in middle],
            "call_ids": [e.get("call_id") for e in middle],
        }
        self.entries = [first, summary, *tail]


@dataclass
class AgentCall:
    """One tool call the agent made, with its cached payload."""

    tool: str
    call_id: str
    payload: dict[str, Any]


@dataclass
class AgentResult:
    """Result of one agent run."""

    run_id: str
    question: str
    plan: Plan
    report_path: Path
    report: str
    calls: list[AgentCall] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    compactions: int = 0
    downgrades: int = 0
    recoveries: int = 0

    @property
    def chosen_model(self) -> str:
        """The reconciled model name selected by the agent."""
        for call in self.calls:
            if call.tool == "ensemble_forecast":
                return str(call.payload.get("model_name", "unknown"))
        return self.plan.models[0] if self.plan.models else "unknown"


class ForecastAgent:
    """Inspect -> forecast candidates -> compare -> reconcile -> report."""

    def __init__(
        self,
        run_id: str | None = None,
        role: str = "analyst",
        runtime: ToolRuntime | None = None,
        max_context_tokens: int = 2500,
        prefer_llm: bool = True,
    ):
        self.runtime = runtime or ToolRuntime(
            run_id=run_id or f"agent-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}",
            role=role,
        )
        self.run_id = self.runtime.run_id
        self.max_context_tokens = max_context_tokens
        self.prefer_llm = prefer_llm
        self._downgrades = 0
        self._recoveries = 0

    async def _call(
        self,
        client: McpToolSession,
        tool: str,
        arguments: dict[str, Any],
        context: ContextWindow,
        calls: list[AgentCall],
    ) -> dict[str, Any]:
        try:
            payload = await client.call(tool, arguments)
        except BudgetExceeded as exc:
            if "model" not in arguments:
                raise
            downgraded = dict(arguments)
            downgraded["model"] = "seasonal_naive"
            self.runtime.trace.event(
                "agent.downgrade",
                {"tool": tool, "from_model": arguments.get("model"), "to_model": "seasonal_naive",
                 "reason": exc.reason},
            )
            self._downgrades += 1
            payload = await client.call(tool, downgraded)
        except ToolCallFailure as exc:
            self.runtime.trace.event(
                "agent.recovery", {"tool": tool, "error": str(exc)[:300], "retry": True}
            )
            self._recoveries += 1
            payload = await client.call(tool, arguments)
        before, after, compacted = context.add(tool, payload)
        if compacted:
            self.runtime.trace.event(
                "agent.compaction", {"tokens_before": before, "tokens_after": after,
                                     "tool": tool}
            )
        calls.append(AgentCall(tool=tool, call_id=str(payload["call_id"]), payload=payload))
        return payload

    async def run(self, question: str) -> AgentResult:
        """Execute the full agent loop for one natural-language question."""
        self._downgrades = 0
        self._recoveries = 0
        context = ContextWindow(self.max_context_tokens)
        calls: list[AgentCall] = []
        dataset = detect_dataset(question)
        async with McpToolSession(self.runtime) as client:
            listing = await self._call(
                client,
                "list_series",
                {"dataset": dataset, "filter": {"limit": 50}},
                context,
                calls,
            )
            series_ids = [s["series_id"] for s in listing["series"]]
            models_available = available_names()
            plan = make_plan(
                question,
                dataset,
                series_ids,
                models_available,
                prefer_llm=self.prefer_llm,
            )
            plan.run_id = self.run_id
            if plan.dataset != dataset:
                dataset = plan.dataset
            described = await self._call(
                client,
                "describe_series",
                {"series_id": plan.series_id, "dataset": dataset},
                context,
                calls,
            )
            if plan.window_id is None:
                plan.window_id = int(described["holdout_windows"][-1]["window_id"])

            for model in plan.models:
                await self._call(
                    client,
                    "forecast",
                    {
                        "model": model,
                        "series_id": plan.series_id,
                        "horizon": plan.horizon,
                        "quantiles": plan.quantile_levels,
                        "covariates": True,
                        "dataset": dataset,
                        "window_id": plan.window_id,
                    },
                    context,
                    calls,
                )
            for model in plan.models:
                await self._call(
                    client,
                    "backtest",
                    {
                        "model": model,
                        "series_id": plan.series_id,
                        "n_windows": 6,
                        "dataset": dataset,
                    },
                    context,
                    calls,
                )
            await self._call(
                client,
                "ensemble_forecast",
                {
                    "models": plan.models,
                    "series_id": plan.series_id,
                    "horizon": plan.horizon,
                    "quantiles": plan.quantile_levels,
                    "method": plan.method,
                    "dataset": dataset,
                    "window_id": plan.window_id,
                    "covariates": True,
                },
                context,
                calls,
            )

        unavailable = {}
        for name in ("timesfm", "moirai", "chronos-small"):
            ok, reason = availability(name)
            if not ok:
                unavailable[name] = reason
        records = [{"tool": c.tool, "call_id": c.call_id, "payload": c.payload} for c in calls]
        plan.window_id = int(
            next(
                (c.payload.get("window_id") for c in calls if c.tool == "forecast"),
                plan.window_id,
            )
        )
        report_text = build_report(question, plan, records, unavailable)
        report_path = write_report(self.runtime.trace.run_dir, report_text)
        snapshot = self.runtime.guard.snapshot()
        self.runtime.trace.event(
            "agent.summary",
            {
                "run.id": self.run_id,
                "calls": len(calls),
                "cost_usd": snapshot["cost_usd"],
                "elapsed_seconds": snapshot["elapsed_seconds"],
                "compactions": context.compactions,
                "downgrades": self._downgrades,
                "recoveries": self._recoveries,
                "report_path": str(report_path),
            },
        )
        return AgentResult(
            run_id=self.run_id,
            question=question,
            plan=plan,
            report_path=report_path,
            report=report_text,
            calls=calls,
            guard=snapshot,
            unavailable=unavailable,
            compactions=context.compactions,
            downgrades=self._downgrades,
            recoveries=self._recoveries,
        )

    def run_sync(self, question: str) -> AgentResult:
        """Synchronous wrapper used by the CLI and evaluation harness."""
        return asyncio.run(self.run(question))
