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
eval_app = typer.Typer(no_args_is_help=True, help="Benchmark, agent evaluation and plots.")
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(finetune_app, name="finetune")
app.add_typer(eval_app, name="eval")


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



@finetune_app.callback(invoke_without_command=True)
def finetune_root(
    ctx: typer.Context,
    dataset: str = typer.Option("synthetic_supply", "--dataset", "-d", help="Training dataset."),
    tag: str = typer.Option("v1", "--tag", help="Adapter tag / output folder."),
    config: str = typer.Option("configs/finetune.yaml", "--config", help="Fine-tune YAML."),
) -> None:
    """Train a LoRA adapter (runs when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        return
    from pathlib import Path

    from forecastmcp.config import ROOT
    from forecastmcp.finetune.train import TrainConfig, finetune

    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = TrainConfig.load(config_path, tag=tag, dataset=dataset)
    typer.echo(
        f"fine-tuning {cfg.base_checkpoint} with LoRA r={cfg.lora_r} on {cfg.dataset} "
        f"(device={cfg.device}, windows={cfg.max_windows})"
    )
    card = finetune(cfg, config_path=config_path)
    typer.echo(
        f"saved adapter+card to artifacts/finetune/{card['tag']} | "
        f"trainable={card['trainable_params']} / total={card['total_params']} | "
        f"best val quantile loss={card['best_val_quantile_loss']:.5f} | "
        f"wall={card['wall_clock_seconds']:.1f}s"
    )


@finetune_app.command("compare")
def finetune_compare(
    tag: str = typer.Option("v1", "--tag", help="Adapter tag."),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Override dataset."),
    series: int = typer.Option(20, "--series", min=1, help="Number of series to evaluate."),
    windows: int = typer.Option(6, "--windows", min=1, help="Frozen windows to evaluate."),
) -> None:
    """Compare zero-shot Chronos against the fine-tuned adapter on the holdout."""
    from forecastmcp.finetune.train import compare_finetuned

    payload = compare_finetuned(tag, dataset=dataset, n_series=series, n_windows=windows)
    names = list(payload["models"])
    rows = []
    for metric in ("WQL", "CRPS", "MASE", "coverage_80", "coverage_90"):
        rows.append(
            [metric]
            + [f"{payload['models'][name][metric]:.4f}" for name in names]
            + [f"{payload['delta'][metric]:+.4f}"]
        )
    print_table(["metric", *names, "delta"], rows)
    typer.echo(
        f"\n{payload['n_series']} series x {payload['n_windows']} frozen windows; "
        f"results written to artifacts/finetune/{tag}/compare.json"
    )


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


@app.command("bench")
def bench_cmd(
    config: str = typer.Option("configs/benchmark.yaml", "--config", help="Benchmark YAML."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit benchmark run id."),
    datasets: str | None = typer.Option(None, "--datasets", help="Comma-separated datasets."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model names."),
    series: int | None = typer.Option(None, "--series", min=1, help="Max series per dataset."),
    windows: int | None = typer.Option(None, "--windows", min=1, help="Frozen windows to use."),
    resamples: int = typer.Option(1000, "--resamples", min=50, help="Bootstrap resamples."),
    fast: bool = typer.Option(False, "--fast", help="Small CI-friendly run."),
) -> None:
    """Run the full benchmark on the frozen holdout."""
    from pathlib import Path

    from forecastmcp.config import ROOT
    from forecastmcp.eval.runner import run_benchmark

    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    summary = run_benchmark(
        config_path=config_path,
        run_id=run_id,
        fast=fast,
        datasets=_csv(datasets),
        models=_csv(models),
        max_series=series,
        n_windows=windows,
        n_resamples=resamples,
    )
    for dataset, models_summary in summary["datasets"].items():
        typer.secho(f"\n{dataset}", bold=True)
        rows = []
        for model, entry in models_summary.items():
            if model.startswith("_"):
                continue
            wql = entry["WQL"]
            rows.append(
                [model, f"{wql['mean']:.4f}", f"{wql['ci_low']:.4f}",
                 f"{wql['ci_high']:.4f}", f"{entry['median_latency_ms']:.1f}",
                 f"{entry['estimated_cost_per_call_usd']:.6f}"]
            )
        print_table(["model", "WQL", "ci_low", "ci_high", "median_ms", "usd/call"], rows)
    typer.echo(f"\nrun_id: {summary['run_id']} | results/summary.json written")


@eval_app.command("agents")
def eval_agents_cmd(
    tasks: str = typer.Option("configs/agent_tasks.yaml", "--tasks", help="Task YAML."),
    trials: int = typer.Option(3, "--trials", min=1, help="Repetitions per task."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Limit number of tasks."),
    prefix: str = typer.Option("agenteval", "--prefix", help="Run id prefix."),
    benchmark_run: str | None = typer.Option(None, "--benchmark-run", help="Oracle run id."),
) -> None:
    """Score the agent harness on verifiable holdout tasks."""
    from pathlib import Path

    from forecastmcp.config import ROOT
    from forecastmcp.eval.agent_eval import run_agent_eval

    tasks_path = Path(tasks)
    if not tasks_path.is_absolute():
        tasks_path = ROOT / tasks_path
    summary = run_agent_eval(
        tasks_path=tasks_path,
        trials=trials,
        limit=limit,
        prefix=prefix,
        benchmark_run_id=benchmark_run,
    )
    rows = [
        ["success_rate", summary["success_rate"]],
        ["tool_call_validity_rate", summary["tool_call_validity_rate"]],
        ["unsupported_number_rate", summary["unsupported_number_rate"]],
        ["budget_adherence_rate", summary["budget_adherence_rate"]],
        ["recovery_rate", summary["recovery_rate"]],
        ["selection_regret", summary["selection_regret"]],
        ["mean_latency_seconds", summary["mean_latency_seconds"]],
        ["mean_cost_usd", summary["mean_cost_usd"]],
    ]
    print_table(["metric", "value"], rows)
    typer.echo(f"runs={summary['n_runs']} -> results/agent_eval.json")


@eval_app.command("plots")
def eval_plots_cmd() -> None:
    """Regenerate calibration, horizon and cost/accuracy figures."""
    from forecastmcp.eval.plots import make_all_plots

    paths = make_all_plots()
    for path in paths:
        typer.echo(str(path))


@eval_app.command("summary")
def eval_summary_cmd() -> None:
    """Write results/summary.md and results/NEGATIVE_RESULTS.md from generated data."""
    from forecastmcp.eval.summary import write_outputs

    summary_path, negative_path = write_outputs()
    typer.echo(str(summary_path))
    typer.echo(str(negative_path))


@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(f"forecastmcp {__version__}")


@app.command("serve")
def serve_cmd(
    transport: str = typer.Option("stdio", "--transport", help="stdio or streamable-http."),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind host."),
    port: int = typer.Option(8000, "--port", help="HTTP bind port."),
    role: str = typer.Option("analyst", "--role", help="Permission role (analyst|admin)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit trace run id."),
) -> None:
    """Run the MCP server over stdio or Streamable HTTP."""
    from forecastmcp.mcp.server import serve

    serve(transport=transport, role=role, run_id=run_id, host=host, port=port)


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
