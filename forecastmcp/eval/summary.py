"""Generate summary.md and NEGATIVE_RESULTS.md strictly from produced result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecastmcp.config import FINETUNE_DIR, RESULTS_DIR

METRIC_LABELS = {
    "WQL": "WQL",
    "CRPS": "CRPS",
    "MASE": "MASE",
    "coverage_50": "cov@50",
    "coverage_80": "cov@80",
    "coverage_90": "cov@90",
}


def _ci(entry: dict[str, Any], digits: int = 4) -> str:
    if entry is None or entry.get("mean") is None:
        return "n/a"
    return (
        f"{entry['mean']:.{digits}f} "
        f"[{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]"
    )


def _delta(entry: dict[str, Any], digits: int = 4) -> str:
    if entry is None or entry.get("delta") is None:
        return "n/a"
    return (
        f"{entry['delta']:+.{digits}f} "
        f"[{entry['ci_low']:+.{digits}f}, {entry['ci_high']:+.{digits}f}]"
    )


def _load(name: str, root: Path = RESULTS_DIR) -> dict[str, Any] | None:
    path = root / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _finetune_compare() -> dict[str, Any] | None:
    path = FINETUNE_DIR / "v1" / "compare.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_summary_markdown(
    summary: dict[str, Any],
    agent: dict[str, Any] | None,
    finetune: dict[str, Any] | None,
) -> str:
    """Render a benchmark + agent-eval summary with 95% bootstrap CIs."""
    manifest = summary.get("manifest", {})
    lines = ["# ForecastMCP benchmark summary", ""]
    lines.append(f"Benchmark run id: `{summary.get('run_id', 'unknown')}`")
    lines.append(
        f"Bootstrap resamples: {manifest.get('n_resamples', 'n/a')} "
        f"(seed {manifest.get('seed', 'n/a')}). All metrics are means over eligible series; "
        "brackets are 95% percentile bootstrap CIs from series-level resampling."
    )
    lines.append("")

    lines.append("## Datasets")
    lines.append("")
    lines.append("| dataset | series | freq | horizon | windows |")
    lines.append("|---|---:|---|---:|---:|")
    for dataset, meta in manifest.get("datasets", {}).items():
        if "n_series" not in meta:
            continue
        lines.append(
            f"| {dataset} | {meta['n_series']} | {meta['freq']} | {meta['horizon']} | "
            f"{meta['n_windows']} |"
        )
    lines.append("")

    for dataset, models in summary.get("datasets", {}).items():
        public = {k: v for k, v in models.items() if not k.startswith("_")}
        if not public:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(
            "| model | WQL | CRPS | MASE | cov@50 | cov@80 | cov@90 | median latency (ms) "
            "| est. cost/call (USD) |"
        )
        lines.append("|---|---|---|---|---|---|---|---:|---:|")
        for model, entry in public.items():
            lines.append(
                "| {m} | {wql} | {crps} | {mase} | {c50} | {c80} | {c90} | {lat:.1f} | "
                "{cost:.6f} |".format(
                    m=model,
                    wql=_ci(entry.get("WQL")),
                    crps=_ci(entry.get("CRPS")),
                    mase=_ci(entry.get("MASE")),
                    c50=_ci(entry.get("coverage_50"), 3),
                    c80=_ci(entry.get("coverage_80"), 3),
                    c90=_ci(entry.get("coverage_90"), 3),
                    lat=entry.get("median_latency_ms", float("nan")),
                    cost=entry.get("estimated_cost_per_call_usd", float("nan")),
                )
            )
        best = models.get("_best_single_model")
        if best:
            lines.append("")
            lines.append(f"Best single model by mean WQL: `{best}`.")
        lines.append("")

    if finetune:
        lines.append("## Zero-shot vs LoRA fine-tuned (independent compare run)")
        lines.append("")
        lines.append(
            f"Evaluated {finetune['n_series']} series over {finetune['n_windows']} frozen "
            "windows. Delta is fine-tuned minus zero-shot; negative is better for error metrics. "
            "Paired bootstrap significance for the benchmark run is in the deltas table below."
        )
        lines.append("")
        lines.append("| metric | zero-shot | fine-tuned | delta |")
        lines.append("|---|---|---|---|")
        names = list(finetune["models"])
        for metric in ("WQL", "CRPS", "MASE", "coverage_80", "coverage_90"):
            values = " | ".join(f"{finetune['models'][n][metric]:.4f}" for n in names)
            lines.append(f"| {metric} | {values} | {finetune['delta'][metric]:+.4f} |")
        lines.append("")

    lines.append("## Paired bootstrap deltas")
    lines.append("")
    lines.append(
        "| dataset | comparison | metric | delta (A-B) | 95% CI | crosses zero |"
    )
    lines.append("|---|---|---|---:|---|---|")
    for dataset, deltas in summary.get("paired_deltas", {}).items():
        for key, entry in deltas.items():
            label, pair, metric = key.split(":", 2)
            lines.append(
                f"| {dataset} | {label} ({pair.replace('_minus_', ' - ')}) | {metric} | "
                f"{entry['delta']:+.4f} | [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] | "
                f"{entry['crosses_zero']} |"
            )
    lines.append("")

    lines.append("## Ablations")
    lines.append("")
    ablations = summary.get("ablations", {})
    for name, value in ablations.items():
        text = value if isinstance(value, str) else value.get("reason", str(value))
        lines.append(f"- **{name}**: {text}")
    lines.append("")

    if agent:
        lines.append("## Agent harness metrics")
        lines.append("")
        lines.append(
            f"{agent['n_tasks']} tasks x {agent['trials_per_task']} trials = "
            f"{agent['n_runs']} runs; benchmark oracle run `{agent.get('benchmark_run_id')}`."
        )
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        rows = [
            ("success rate", agent["success_rate"]),
            ("tool-call validity rate", agent["tool_call_validity_rate"]),
            ("selection regret (WQL vs oracle)", agent["selection_regret"]),
            ("ensemble regret (WQL vs oracle)", agent["ensemble_regret"]),
            ("unsupported-number rate", agent["unsupported_number_rate"]),
            ("budget adherence rate", agent["budget_adherence_rate"]),
            ("recovery rate after injected failure", agent["recovery_rate"]),
            (
                "mean latency per task (s)",
                f"{agent['mean_latency_seconds']:.4f} "
                f"(run-to-run std {agent['latency_run_to_run_std']:.4f})",
            ),
            (
                "mean cost per task (USD)",
                f"{agent['mean_cost_usd']:.6f} "
                f"(run-to-run std {agent['cost_run_to_run_std']:.6f})",
            ),
        ]
        for label, value in rows:
            if isinstance(value, str):
                lines.append(f"| {label} | {value} |")
            else:
                lines.append(f"| {label} | {'n/a' if value is None else f'{value:.4f}'} |")
        if agent.get("selection_regret_run_to_run_std") is not None:
            lines.append(
                f"| selection regret run-to-run std | "
                f"{agent['selection_regret_run_to_run_std']:.4f} |"
            )
        lines.append("")
        lines.append("| task | success | selection regret | latency (s) | cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|")
        for task in agent.get("per_task", []):
            regret = task["mean_selection_regret"]
            lines.append(
                f"| {task['task_id']} | {task['success_rate']:.2f} | "
                f"{'n/a' if regret is None else f'{regret:+.4f}'} | "
                f"{task['mean_latency_seconds']:.2f} | {task['mean_cost_usd']:.6f} |"
            )
        lines.append("")

    availability = manifest.get("model_availability", {})
    if availability:
        lines.append("## Model availability")
        lines.append("")
        lines.append("| model | available | ran on | reason |")
        lines.append("|---|---|---|---|")
        ran: dict[str, list[str]] = {}
        for dataset, models in summary.get("datasets", {}).items():
            for model in models:
                if model.startswith("_"):
                    continue
                ran.setdefault(model, []).append(dataset)
        for model, entry in availability.items():
            lines.append(
                f"| {model} | {entry['available']} | {', '.join(ran.get(model, [])) or '-'} | "
                f"{entry['reason']} |"
            )
        lines.append("")

    figures = sorted((RESULTS_DIR / "figures").glob("*.png"))
    if figures:
        lines.append("## Figures")
        lines.append("")
        for fig in figures:
            lines.append(f"![{fig.stem}](figures/{fig.name})")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_negative_results(
    summary: dict[str, Any],
    agent: dict[str, Any] | None,
    finetune: dict[str, Any] | None,
) -> str:
    """Generate an honest negative-results note from the actual metric files."""
    lines = ["# Negative results and limitations", ""]
    lines.append(
        "This file is generated from results/summary.json, results/agent_eval.json and the "
        "fine-tune compare artifact. It records where the methods did not help."
    )
    lines.append("")

    lines.append("## Fine-tuning vs zero-shot")
    if finetune:
        for metric in ("WQL", "CRPS", "MASE"):
            delta = finetune["delta"][metric]
            lines.append(
                f"- {metric}: fine-tuned minus zero-shot = {delta:+.4f} "
                f"on {finetune['n_series']} series."
            )
        wql_delta = finetune["delta"]["WQL"]
        ci = None
        for dataset_deltas in summary.get("paired_deltas", {}).values():
            for key, entry in dataset_deltas.items():
                if "zero_shot_vs_lora" in key and key.endswith(":WQL"):
                    ci = entry
        if ci is not None:
            if ci["crosses_zero"]:
                lines.append(
                    f"- Paired 95% CI for the WQL delta is "
                    f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}], which includes zero: "
                    "there is no statistically detectable WQL gain from fine-tuning."
                )
            elif ci["delta"] < 0:
                lines.append(
                    f"- The LoRA adapter improved WQL with a 95% CI "
                    f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] excluding zero."
                )
            else:
                lines.append(
                    f"- The LoRA adapter made WQL worse with a 95% CI "
                    f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] excluding zero."
                )
        elif wql_delta > 0:
            lines.append("- The LoRA adapter did NOT improve WQL on the holdout.")
        elif wql_delta < 0:
            lines.append(
                "- The LoRA adapter's point estimate improved WQL, but no paired CI was available."
            )
        else:
            lines.append("- WQL was unchanged by fine-tuning.")
    else:
        lines.append("- No fine-tune compare artifact was available, so no claim is made.")
    lines.append("")

    lines.append("## Ensembles vs the best single model")
    any_ensemble_win = False
    for dataset, deltas in summary.get("paired_deltas", {}).items():
        best = summary.get("datasets", {}).get(dataset, {}).get("_best_single_model")
        for key, entry in deltas.items():
            label, pair, metric = key.split(":", 2)
            if label != "ensemble_vs_best_single" or metric != "WQL":
                continue
            improved = entry["delta"] < 0 and not entry["crosses_zero"]
            any_ensemble_win = any_ensemble_win or improved
            if abs(entry["delta"]) < 1e-12 and abs(entry["ci_low"]) < 1e-12:
                lines.append(
                    f"- {dataset}: {pair.replace('_minus_', ' vs ')} is identical to {best} "
                    "by construction (the selection rule picks that same model)."
                )
                continue
            verdict = "significantly better" if improved else "not significantly better"
            lines.append(
                f"- {dataset}: {pair.replace('_minus_', ' vs ')} WQL delta "
                f"{entry['delta']:+.4f} [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}], "
                f"{verdict} than {best}."
            )
    if not any_ensemble_win:
        lines.append(
            "- No ensemble/reconciliation arm beat the best single model with a 95% CI "
            "excluding zero."
        )
    lines.append("")

    lines.append("## Covariates")
    cov_found = False
    for dataset, deltas in summary.get("paired_deltas", {}).items():
        for key, entry in deltas.items():
            label, pair, metric = key.split(":", 2)
            if label != "with_vs_without_covariates" or metric != "WQL":
                continue
            cov_found = True
            helped = entry["delta"] < 0 and not entry["crosses_zero"]
            lines.append(
                f"- {dataset}: promo-aware seasonal naive minus plain seasonal naive WQL "
                f"{entry['delta']:+.4f} [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]; "
                f"{'helped' if helped else 'no significant help'}."
            )
    if not cov_found:
        lines.append("- No covariate ablation was produced.")
    lines.append(
        "- Chronos T5 does not consume exogenous covariates in this implementation, so there "
        "is no foundation-model covariate arm."
    )
    lines.append("")

    lines.append("## Multivariate / covariate-capable foundation models")
    ablations = summary.get("ablations", {}).get("univariate_vs_multivariate", {})
    lines.append(f"- Status: {ablations.get('status', 'unknown')}. {ablations.get('reason', '')}")
    lines.append("")

    lines.append("## Agent routing cost")
    if agent:
        regret = agent.get("selection_regret")
        regret_text = "n/a" if regret is None else f"{regret:+.4f} WQL"
        recovery = agent.get("recovery_rate")
        recovery_text = "n/a" if recovery is None else f"{recovery:.2f}"
        lines.append(
            f"- Mean selection regret vs oracle: {regret_text}; mean cost per task "
            f"{agent['mean_cost_usd']:.6f} USD; mean latency "
            f"{agent['mean_latency_seconds']:.2f}s; recovery rate {recovery_text}."
        )
        if regret is not None and regret > 0:
            lines.append(
                "- Multi-model routing added cost and latency without beating the oracle model "
                "on WQL."
            )
    else:
        lines.append("- Agent evaluation was not run.")
    lines.append("")

    lines.append("## Unavailable models")
    manifest = summary.get("manifest", {})
    availability = manifest.get("model_availability", {})
    unavailable = manifest.get("unavailable_models", {})
    wrote = False
    for model, entry in availability.items():
        if not entry.get("available"):
            ran = ", ".join(
                dataset for dataset, models in summary.get("datasets", {}).items()
                if model in models
            )
            lines.append(
                f"- {model}: {entry.get('reason')} "
                f"({'not in the benchmark model list' if not ran else 'partially run on ' + ran})"
            )
            wrote = True
    for dataset, models in unavailable.items():
        for model, reason in models.items():
            lines.append(f"- {model} on {dataset}: {reason}")
            wrote = True
    if not wrote:
        lines.append("- None recorded.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs() -> tuple[Path, Path]:
    """Write summary.md and NEGATIVE_RESULTS.md, returning both paths."""
    summary = _load("summary.json")
    if summary is None:
        raise FileNotFoundError("results/summary.json is missing; run `forecastmcp bench` first")
    agent = _load("agent_eval.json")
    finetune = _finetune_compare()
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(build_summary_markdown(summary, agent, finetune), encoding="utf-8")
    negative_path = RESULTS_DIR / "NEGATIVE_RESULTS.md"
    negative_path.write_text(build_negative_results(summary, agent, finetune), encoding="utf-8")
    return summary_path, negative_path
