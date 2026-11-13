"""Official-SDK MCP server exposing the typed forecasting tools."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from forecastmcp.mcp.budgets import BudgetExceeded
from forecastmcp.mcp.permissions import PermissionDenied
from forecastmcp.mcp.schemas import (
    BacktestOutput,
    DescribeSeriesOutput,
    EnsembleOutput,
    ForecastOutput,
    ListSeriesOutput,
    ModelCardOutput,
    SqlOutput,
)
from forecastmcp.mcp.tools import ToolFailure, ToolRuntime, output_model

INSTRUCTIONS = (
    "ForecastMCP exposes pretrained time-series forecasters (Chronos, TimesFM, Moirai) "
    "and classical baselines as typed tools. Forecasts are probabilistic: pass quantile "
    "levels. Every call is budgeted, permission-checked and traced to "
    "artifacts/runs/<run_id>/trace.jsonl. SQL is read-only."
)


def _dispatch(runtime: ToolRuntime, tool: str, arguments: dict[str, Any]):
    """Execute through the runtime and adapt failures to MCP tool errors."""
    try:
        payload = runtime.execute(tool, arguments)
    except (BudgetExceeded, PermissionDenied, ToolFailure, ValueError, KeyError, RuntimeError) as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    return output_model(tool, payload)


def build_server(
    runtime: ToolRuntime | None = None,
    role: str = "analyst",
    name: str = "forecastmcp",
) -> tuple[MCPServer, ToolRuntime]:
    """Create the MCP server and its runtime (returned so callers can close it)."""
    rt = runtime or ToolRuntime(role=role)
    server = MCPServer(name=name, version="0.1.0", instructions=INSTRUCTIONS)

    @server.tool()
    def list_series(
        dataset: str = "synthetic_supply",
        filter: dict[str, Any] | None = None,
    ) -> ListSeriesOutput:
        """List series in a dataset with optional {limit, offset, prefix, category}."""
        return _dispatch(rt, "list_series", {"dataset": dataset, "filter": filter})

    @server.tool()
    def describe_series(
        series_id: str,
        dataset: str = "synthetic_supply",
    ) -> DescribeSeriesOutput:
        """Describe one series: stats, covariates, recent values and holdout windows."""
        return _dispatch(rt, "describe_series", {"series_id": series_id, "dataset": dataset})

    @server.tool()
    def forecast(
        model: str,
        series_id: str,
        horizon: int = 28,
        quantiles: list[float] | None = None,
        covariates: bool = True,
        dataset: str = "synthetic_supply",
        window_id: int | None = None,
    ) -> ForecastOutput:
        """Quantile forecast for one series at a frozen holdout origin."""
        return _dispatch(
            rt,
            "forecast",
            {
                "model": model,
                "series_id": series_id,
                "horizon": horizon,
                "quantiles": quantiles,
                "covariates": covariates,
                "dataset": dataset,
                "window_id": window_id,
            },
        )

    @server.tool()
    def ensemble_forecast(
        models: list[str],
        series_id: str,
        horizon: int = 28,
        quantiles: list[float] | None = None,
        method: str = "vincentization",
        weights: list[float] | None = None,
        dataset: str = "synthetic_supply",
        window_id: int | None = None,
        covariates: bool = True,
    ) -> EnsembleOutput:
        """Combine several forecasters by vincentization, inverse error or best-on-backtest."""
        return _dispatch(
            rt,
            "ensemble_forecast",
            {
                "models": models,
                "series_id": series_id,
                "horizon": horizon,
                "quantiles": quantiles,
                "method": method,
                "weights": weights,
                "dataset": dataset,
                "window_id": window_id,
                "covariates": covariates,
            },
        )

    @server.tool()
    def backtest(
        model: str,
        series_id: str,
        n_windows: int = 6,
        dataset: str = "synthetic_supply",
    ) -> BacktestOutput:
        """Backtest one model on the frozen rolling-origin windows for one series."""
        return _dispatch(
            rt,
            "backtest",
            {"model": model, "series_id": series_id, "n_windows": n_windows, "dataset": dataset},
        )

    @server.tool()
    def model_card(model: str) -> ModelCardOutput:
        """Availability, parameter count, context length and training domain for a model."""
        return _dispatch(rt, "model_card", {"model": model})

    @server.tool()
    def sql(query: str) -> SqlOutput:
        """Run a read-only SELECT query against the metrics/forecasts DuckDB (max 500 rows)."""
        return _dispatch(rt, "sql", {"query": query})

    return server, rt


def serve(
    transport: str = "stdio",
    role: str = "analyst",
    run_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the MCP server (``stdio`` or ``streamable-http``)."""
    server, runtime = build_server(role=role, runtime=ToolRuntime(run_id=run_id, role=role))
    try:
        if transport == "stdio":
            server.run("stdio")
        elif transport == "streamable-http":
            server.run("streamable-http", host=host, port=port)
        else:
            raise ValueError(f"unknown transport {transport!r}")
    finally:
        runtime.close()


def serve_stdio(role: str = "analyst", run_id: str | None = None) -> None:
    """Backward-compatible stdio entry point."""
    serve("stdio", role=role, run_id=run_id)
