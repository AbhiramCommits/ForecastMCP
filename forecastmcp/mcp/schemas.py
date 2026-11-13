"""Pydantic input/output schemas for every MCP tool."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forecastmcp.forecasters.base import DEFAULT_QUANTILES, ForecastResult


class ToolInput(BaseModel):
    """Base for strict tool inputs (unknown keys are rejected)."""

    model_config = ConfigDict(extra="forbid")


class ListSeriesInput(ToolInput):
    """Input for ``list_series``."""

    dataset: str = "synthetic_supply"
    filter: dict[str, Any] | None = None


class SeriesFilter(ToolInput):
    """Optional filter/pagination for ``list_series``."""

    limit: int = Field(default=20, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    prefix: str | None = None
    category: str | None = None


class DescribeSeriesInput(ToolInput):
    """Input for ``describe_series``."""

    series_id: str = Field(min_length=1)
    dataset: str = "synthetic_supply"


class ForecastInput(ToolInput):
    """Input for ``forecast``."""

    model: str = Field(min_length=1)
    series_id: str = Field(min_length=1)
    horizon: int = Field(default=28, ge=1, le=1024)
    quantiles: list[float] | None = None
    covariates: bool = True
    dataset: str = "synthetic_supply"
    window_id: int | None = Field(default=None, ge=1)


class EnsembleInput(ToolInput):
    """Input for ``ensemble_forecast``."""

    models: list[str] = Field(min_length=2)
    series_id: str = Field(min_length=1)
    horizon: int = Field(default=28, ge=1, le=1024)
    quantiles: list[float] | None = None
    method: Literal["vincentization", "inverse_error", "best_on_backtest"] = "vincentization"
    weights: list[float] | None = None
    dataset: str = "synthetic_supply"
    window_id: int | None = Field(default=None, ge=1)
    covariates: bool = True

    @model_validator(mode="after")
    def _weights(self) -> EnsembleInput:
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError("weights must have one entry per model")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("weights must be non-negative and sum to a positive value")
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must be unique")
        return self


class BacktestInput(ToolInput):
    """Input for ``backtest``."""

    model: str = Field(min_length=1)
    series_id: str = Field(min_length=1)
    n_windows: int = Field(default=6, ge=1, le=24)
    dataset: str = "synthetic_supply"


class ModelCardInput(ToolInput):
    """Input for ``model_card``."""

    model: str = Field(min_length=1)


class SqlInput(ToolInput):
    """Input for the read-only ``sql`` tool."""

    query: str = Field(min_length=6, max_length=8000)


class SeriesSummary(BaseModel):
    """Compact per-series metadata."""

    series_id: str
    n_obs: int
    start: str
    end: str
    mean: float
    std: float
    last_value: float
    category: str | None = None


class ListSeriesOutput(BaseModel):
    """Output of ``list_series``."""

    call_id: str
    dataset: str
    count: int
    total_series: int
    series: list[SeriesSummary]


class DescribeSeriesOutput(BaseModel):
    """Output of ``describe_series``."""

    call_id: str
    dataset: str
    series: SeriesSummary
    covariates: list[str]
    recent: list[dict[str, Any]]
    holdout_windows: list[dict[str, Any]]


class ForecastOutput(BaseModel):
    """Output of ``forecast``."""

    call_id: str
    dataset: str
    series_id: str
    window_id: int
    forecast: ForecastResult
    actual: list[float] | None = None
    dates: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    summary: dict[str, float] = Field(default_factory=dict)


class EnsembleOutput(BaseModel):
    """Output of ``ensemble_forecast``."""

    call_id: str
    dataset: str
    series_id: str
    window_id: int
    model_name: str
    method: str
    rule_fired: str
    horizon: int
    point: list[float]
    quantile_levels: list[float]
    quantiles: list[list[float]]
    members: list[str]
    weights: dict[str, float]
    divergence: dict[str, Any]
    used_covariates: bool = False
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float
    cost_usd: float
    dates: list[str] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)


class MetricValue(BaseModel):
    """One aggregated metric with an optional bootstrap CI."""

    value: float
    ci_low: float | None = None
    ci_high: float | None = None


class BacktestOutput(BaseModel):
    """Output of ``backtest``."""

    call_id: str
    dataset: str
    model: str
    series_id: str
    n_windows: int
    metrics: dict[str, MetricValue]
    per_window: list[dict[str, Any]]
    cost_usd: float = 0.0


class ModelCardOutput(BaseModel):
    """Output of ``model_card``."""

    call_id: str
    card: dict[str, Any]


class SqlOutput(BaseModel):
    """Output of the read-only ``sql`` tool."""

    call_id: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int


INPUT_MODELS = {
    "list_series": ListSeriesInput,
    "describe_series": DescribeSeriesInput,
    "forecast": ForecastInput,
    "ensemble_forecast": EnsembleInput,
    "backtest": BacktestInput,
    "model_card": ModelCardInput,
    "sql": SqlInput,
}

OUTPUT_MODELS = {
    "list_series": ListSeriesOutput,
    "describe_series": DescribeSeriesOutput,
    "forecast": ForecastOutput,
    "ensemble_forecast": EnsembleOutput,
    "backtest": BacktestOutput,
    "model_card": ModelCardOutput,
    "sql": SqlOutput,
}


def validate_input(tool: str, arguments: dict[str, Any]) -> BaseModel:
    """Validate raw MCP arguments with the tool's Pydantic model."""
    if tool not in INPUT_MODELS:
        raise KeyError(f"unknown tool {tool!r}")
    return INPUT_MODELS[tool].model_validate(arguments or {})


def default_quantiles() -> list[float]:
    """The default quantile grid exposed by tools."""
    return list(DEFAULT_QUANTILES)
