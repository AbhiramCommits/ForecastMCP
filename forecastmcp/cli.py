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
    fmt = "  ".join("{:<%d}" % w for w in widths)
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


@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(f"forecastmcp {__version__}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
