"""Calibration, horizon-error and cost/accuracy figures from benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from forecastmcp.config import RESULTS_DIR  # noqa: E402

FIGURES_DIR = RESULTS_DIR / "figures"
NOMINALS = {"coverage_50": 0.5, "coverage_80": 0.8, "coverage_90": 0.9}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_calibration(summary: dict, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Empirical vs nominal coverage for every model, one panel per dataset."""
    _style()
    paths = []
    for dataset, models in summary["datasets"].items():
        model_names = [m for m in models if not m.startswith("_")]
        if not model_names:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1,
                label="perfect calibration")
        for model in model_names:
            xs = [NOMINALS[k] for k in NOMINALS]
            ys = [models[model][k]["mean"] for k in NOMINALS]
            ax.plot(xs, ys, marker="o", linewidth=1.2, label=model)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"Calibration - {dataset}")
        ax.legend(fontsize=6, loc="lower right", frameon=False)
        fig.tight_layout()
        path = out_dir / f"calibration_{dataset}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_wql_by_horizon(out_dir: Path = FIGURES_DIR) -> list[Path]:
    """WQL by forecast step for each model and dataset."""
    _style()
    path = RESULTS_DIR / "wql_by_horizon.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    paths = []
    for dataset, sub in frame.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for model, model_sub in sub.groupby("model"):
            ax.plot(model_sub["step"], model_sub["wql_by_step"], marker=".",
                    linewidth=1.1, label=model)
        ax.set_xlabel("forecast step")
        ax.set_ylabel("WQL by step")
        ax.set_title(f"Error by horizon - {dataset}")
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        out = out_dir / f"wql_by_horizon_{dataset}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_cost_vs_accuracy(summary: dict, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """WQL against estimated cost per call; annotations include median latency."""
    _style()
    paths = []
    for dataset, models in summary["datasets"].items():
        names = [m for m in models if not m.startswith("_")]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for name in names:
            entry = models[name]
            cost = max(entry["estimated_cost_per_call_usd"], 1e-6)
            wql = entry["WQL"]["mean"]
            ax.scatter(cost, wql, s=28)
            ax.annotate(
                f"{name} ({entry['median_latency_ms']:.0f}ms)",
                (cost, wql),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=5.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("estimated cost per call (USD, log scale)")
        ax.set_ylabel("mean WQL (lower is better)")
        ax.set_title(f"Accuracy vs cost - {dataset}")
        fig.tight_layout()
        out = out_dir / f"cost_vs_accuracy_{dataset}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths


def make_all_plots(summary_path: Path | None = None, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate every figure and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = summary_path or (RESULTS_DIR / "summary.json")
    summary = json.loads(path.read_text())
    paths = []
    paths.extend(plot_calibration(summary, out_dir))
    paths.extend(plot_wql_by_horizon(out_dir))
    paths.extend(plot_cost_vs_accuracy(summary, out_dir))
    return paths
