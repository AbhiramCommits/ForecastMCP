"""Command line interface for ForecastMCP."""

from __future__ import annotations

from typing import Any

import typer

from forecastmcp import __version__
from forecastmcp.config import (
    CONFIGS_DIR,
    dataset_spec,
    ensure_dirs,
    is_offline,
)
from forecastmcp.data.loaders import (
    DatasetUnavailable,
    dataset_summary,
    load_dataset,
)
from forecastmcp.data.splits import freeze_holdout, load_holdout
from forecastmcp.data.store import get_store
from forecastmcp.forecasters.base import ForecastRequest, ModelUnavailable
from forecastmcp.forecasters.registry import (
    estimate_cost,
    get_forecaster,
    list_forecasters,
    model_card,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ForecastMCP: pretrained time-series forecasting as typed MCP tools.",
)
data_app = typer.Typer(no_args_is_help=True, help="Prepare and inspect datasets.")
models_app = typer.Typer(no_args_is_help=True, help="Inspect the forecaster registry.")
finetune_app = typer.Typer(no_args_is_help=True, help="LoRA fine-tuning utilities.")
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(finetune_app, name="finetune")


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Print a plain-text aligned table (no optional rich dependency)."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    typer.echo(fmt.format(*headers))
    typer.echo(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        typer.echo(fmt.format(*[str(c) for c in row]))


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    """Global options."""
    if version:
        typer.echo(f"forecastmcp {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@data_app.command("prepare")
def data_prepare(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name."),
    force: bool = typer.Option(False, "--force", help="Re-download and rebuild the cache."),
) -> None:
    """Load a dataset, register series metadata and freeze the holdout split."""
    ensure_dirs()
    spec = dataset_spec(dataset)
    try:
        df = load_dataset(dataset, use_cache=not force)
    except DatasetUnavailable as exc:
        typer.secho(f"dataset unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    store = get_store()
    n = store.register_series(df, freq=spec.freq)
    entry = freeze_holdout(df, horizon=spec.horizon, n_windows=spec.n_windows)
    summary = dataset_summary(df)
    typer.echo(f"prepared {summary['dataset']}: {n} series registered")
    typer.echo(
        f"  observations={summary['n_observations']} freq={summary['freq']} "
        f"span={summary['start']}..{summary['end']}"
    )
    typer.echo(
        f"  frozen holdout: horizon={entry['horizon']} windows={entry['n_windows']} "
        f"({CONFIGS_DIR / 'holdout.yaml'})"
    )
    if is_offline():
        typer.echo("  FORECASTMCP_OFFLINE=1 active: synthetic data and CPU-cheap models only")


@data_app.command("describe")
def data_describe(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Limit to one dataset."),
) -> None:
    """Print series counts, frequency, horizon and frozen split boundaries."""
    store = get_store()
    holdout = load_holdout()
    meta = store.series_frame(dataset)
    names = sorted(set(meta["dataset"].tolist()) | set(holdout["datasets"]))
    if dataset:
        names = [dataset]
    for name in names:
        entry = holdout["datasets"].get(name)
        rows = meta[meta["dataset"] == name] if not meta.empty else meta
        n_series = int(rows["series_id"].nunique()) if not rows.empty else (
            entry["n_series"] if entry else 0
        )
        typer.secho(f"\n{name}", bold=True)
        typer.echo(f"  series: {n_series}")
        if entry:
            typer.echo(
                f"  freq: {entry['freq']}  horizon: {entry['horizon']}  "
                f"windows: {entry['n_windows']}"
            )
            typer.echo(
                f"  history: {entry['history_start']} .. {entry['history_end']}"
            )
            print_table(
                ["window", "train_start", "train_end", "test_start", "test_end"],
                [
                    [
                        w["window_id"],
                        w["train_start"],
                        w["train_end"],
                        w["test_start"],
                        w["test_end"],
                    ]
                    for w in entry["windows"]
                ],
            )
        else:
            typer.echo("  no frozen holdout yet (run `forecastmcp data prepare`)")


@data_app.command("list")
def data_list() -> None:
    """List datasets supported by the loaders."""
    from forecastmcp.config import DATASET_SPECS

    print_table(
        ["dataset", "kind", "freq", "horizon", "windows", "max_series"],
        [
            [s.name, s.kind, s.freq, s.horizon, s.n_windows, s.max_series]
            for s in DATASET_SPECS.values()
        ],
    )


# ---------------------------------------------------------------------------
# models / forecast
# ---------------------------------------------------------------------------
@models_app.command("list")
def models_list(
    all_models: bool = typer.Option(False, "--all", help="Include disabled entries."),
) -> None:
    """Print the live availability table for every registered forecaster."""
    rows = []
    for row in list_forecasters(include_disabled=all_models):
        rows.append(
            [
                row["name"],
                "yes" if row["available"] else "no",
                row["device"],
                row["n_params"] if row["n_params"] is not None else "?",
                row["context_length"],
                row["cost_per_call_usd"],
                (row["reason"] or "")[:70],
            ]
        )
    print_table(
        ["model", "available", "device", "params", "ctx", "usd/call", "reason"],
        rows,
    )
    if is_offline():
        typer.echo("\nFORECASTMCP_OFFLINE=1: only CPU-cheap models are permitted.")


@models_app.command("card")
def models_card(model: str = typer.Option(..., "--model", "-m", help="Model name.")) -> None:
    """Print a single model card as JSON."""
    import json

    typer.echo(json.dumps(model_card(model), indent=2, default=str))


def _build_request(dataset: str, series_id: str, horizon: int, window_id: int) -> ForecastRequest:
    from forecastmcp.data.requests import build_window_request

    df = load_dataset(dataset)
    return build_window_request(
        df, dataset, series_id, horizon=horizon, window_id=window_id, covariates=True
    ).request


@app.command("forecast")
def forecast_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Registered model name."),
    dataset: str = typer.Option("synthetic_supply", "--dataset", "-d", help="Dataset name."),
    series: str = typer.Option(..., "--series", "-s", help="Series id."),
    horizon: int = typer.Option(28, "--horizon", "-h", min=1, help="Forecast horizon."),
    window: int = typer.Option(1, "--window", "-w", min=1, help="Frozen holdout window id."),
    covariates: bool = typer.Option(True, "--covariates/--no-covariates", help="Pass covariates."),
) -> None:
    """Run one forecaster and print its quantiles."""
    req = _build_request(dataset, series, horizon, window)
    if not covariates:
        req = req.model_copy(update={"past_covariates": None, "future_covariates": None})
    try:
        forecaster = get_forecaster(model)
    except (KeyError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    ok, reason = forecaster.available()
    if not ok:
        typer.secho(f"{model} unavailable: {reason}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=3)
    try:
        result = forecaster.predict(req)
    except ModelUnavailable as exc:
        typer.secho(f"{model} failed at inference: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=3) from exc

    levels = result.quantile_levels
    lo_i = levels.index(min(levels, key=lambda v: abs(v - 0.1)))
    mid_i = levels.index(min(levels, key=lambda v: abs(v - 0.5)))
    hi_i = levels.index(min(levels, key=lambda v: abs(v - 0.9)))
    rows = [
        [
            step + 1,
            f"{result.point[step]:.3f}",
            f"{result.quantiles[lo_i][step]:.3f}",
            f"{result.quantiles[mid_i][step]:.3f}",
            f"{result.quantiles[hi_i][step]:.3f}",
        ]
        for step in range(result.horizon)
    ]
    typer.echo(
        f"{result.model_name} | {result.model_version} | revision={result.revision} | "
        f"device={result.device} | params={result.n_params} | "
        f"latency={result.latency_ms:.1f}ms | "
        f"est_cost=${estimate_cost(result.model_name, result.latency_ms, result.device):.6f}"
    )
    print_table(["step", "point", "q10", "q50", "q90"], rows)
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")



@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(f"forecastmcp {__version__}")


@app.command("serve")
def serve_cmd(
    role: str = typer.Option("analyst", "--role", help="Permission role (analyst|admin)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit trace run id."),
) -> None:
    """Run the MCP server over stdio."""
    from forecastmcp.mcp.server import serve_stdio

    serve_stdio(role=role, run_id=run_id)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Natural-language forecasting question."),
    role: str = typer.Option("analyst", "--role", help="Permission role (analyst|admin)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit trace run id."),
    print_report: bool = typer.Option(True, "--print/--no-print", help="Print the report."),
) -> None:
    """Run the agent loop and write a citation-checked report."""
    from forecastmcp.agent.loop import ForecastAgent

    agent = ForecastAgent(run_id=run_id, role=role)
    result = agent.run_sync(question)
    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"plan: {result.plan.source} planner, models={result.plan.models}, "
               f"series={result.plan.series_id}, horizon={result.plan.horizon}")
    typer.echo(f"chosen: {result.chosen_model} | calls: {len(result.calls)} | "
               f"cost: ${result.guard['cost_usd']:.6f} | "
               f"elapsed: {result.guard['elapsed_seconds']:.1f}s")
    typer.echo(f"report: {result.report_path}")
    if print_report:
        typer.echo("\n" + result.report)


@app.command("replay")
def replay_cmd(
    run_id: str = typer.Argument(..., help="Run id under artifacts/runs/."),
) -> None:
    """Re-execute a traced run and assert byte-identical outputs."""
    import json as _json

    from forecastmcp.obs.replay import replay

    report = replay(run_id)
    typer.echo(_json.dumps(report.to_dict(), indent=2))
    if not report.ok:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
