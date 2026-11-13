"""LLM-backed planner plus a deterministic rule-based fallback."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forecastmcp.config import CONFIGS_DIR, read_yaml
from forecastmcp.forecasters.registry import availability

DEFAULT_MODELS = ["seasonal_naive", "autoets", "chronos"]


class LLMPlan(BaseModel):
    """Strict schema for a plan returned by the LLM planner."""

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(min_length=1)
    dataset: str = "synthetic_supply"
    horizon: int = Field(ge=1, le=365)
    quantile_levels: list[float] = Field(min_length=3)
    models: list[str] = Field(min_length=2)
    method: str = "inverse_error"
    window_id: int | None = None


@dataclass
class Plan:
    """A resolved plan for one agent question."""

    question: str
    dataset: str
    series_id: str
    horizon: int
    quantile_levels: list[float]
    models: list[str]
    method: str
    window_id: int | None = None
    source: str = "rules"
    run_id: str = ""
    notes: list[str] = field(default_factory=list)


def detect_dataset(question: str) -> str:
    """Infer the dataset name from the question text."""
    q = question.lower()
    if "m4" in q:
        return "m4_daily"
    if "electric" in q:
        return "electricity"
    if "traffic" in q:
        return "traffic"
    return "synthetic_supply"


def detect_horizon(question: str) -> int:
    """Parse a horizon in weeks/days/months from natural language."""
    q = question.lower()
    match = re.search(r"(\d+)\s*(week|day|month)s?", q)
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        return n * 7 if unit == "week" else n * 28 if unit == "month" else n
    if "fortnight" in q:
        return 14
    return 28


def detect_window(question: str) -> int | None:
    """Parse an explicit frozen split id from the question ("split 4")."""
    match = re.search(r"(?:split|window)\s*(\d+)", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def detect_quantiles(question: str) -> list[float]:
    """Map a requested interval to a quantile grid."""
    match = re.search(r"(\d{2})\s*%", question)
    if match:
        coverage = int(match.group(1)) / 100.0
        alpha = round((1 - coverage) / 2, 3)
        return [alpha, 0.5, round(1 - alpha, 3)]
    return [0.1, 0.5, 0.9]


def pick_models(question: str, dataset: str) -> list[str]:
    """Choose two or more currently available candidate forecasters."""
    candidates = list(DEFAULT_MODELS)
    if re.search(r"\b(compare|all|every)\b", question, re.IGNORECASE):
        candidates.insert(1, "autoets")
    available = []
    for name in candidates:
        ok, _ = availability(name)
        if ok:
            available.append(name)
    for fallback in ("autoets", "autotheta", "seasonal_naive_cov"):
        if len(available) >= 2:
            break
        ok, _ = availability(fallback)
        if ok and fallback not in available:
            available.append(fallback)
    if not available:
        available = ["seasonal_naive"]
    return available


def rule_based_plan(
    question: str,
    dataset: str,
    series_ids: list[str],
    default_series: str | None = None,
) -> Plan:
    """Deterministic planner that always works without an API key."""
    series_id = default_series
    match = re.search(r"\bS\d{3}\b", question, re.IGNORECASE)
    if match:
        candidate = match.group(0).upper()
        if not series_ids or candidate in series_ids:
            series_id = candidate
    if series_id is None and series_ids:
        series_id = series_ids[0]
    if series_id is None:
        raise ValueError("no series available in the dataset")
    return Plan(
        question=question,
        dataset=dataset,
        series_id=series_id,
        horizon=detect_horizon(question),
        quantile_levels=detect_quantiles(question),
        models=pick_models(question, dataset),
        method="inverse_error",
        window_id=detect_window(question),
        source="rules",
    )


def _llm_config() -> dict[str, Any]:
    return (read_yaml(CONFIGS_DIR / "costs.yaml", default={}) or {}).get("llm", {})


def llm_plan(
    question: str,
    dataset: str,
    series_ids: list[str],
    available_models: list[str],
    timeout: float = 30.0,
) -> Plan | None:
    """Ask the Anthropic Messages API for a strict JSON plan.

    Returns ``None`` on any failure (no key, network error, invalid JSON, schema
    violation) so the caller can fall back to the deterministic planner.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    cfg = _llm_config()
    prompt = (
        "You are the planner for a forecasting agent. Return ONLY JSON with keys "
        "series_id, dataset, horizon, quantile_levels, models, method. "
        f"Question: {question}\nDataset: {dataset}\n"
        f"Series ids (first 50): {series_ids[:50]}\n"
        f"Available models: {available_models}\n"
        "models must contain at least two names from the available list. method must be "
        "vincentization, inverse_error or best_on_backtest. Include window_id when the "
        "question names a split."
    )
    body = json.dumps(
        {
            "model": cfg.get("model", "claude-sonnet-4-5"),
            "max_tokens": 400,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
        text = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        parsed = LLMPlan.model_validate(json.loads(text[start : end + 1]))
        if parsed.series_id not in series_ids or any(m not in available_models for m in parsed.models):
            return None
        return Plan(
            question=question,
            dataset=parsed.dataset,
            series_id=parsed.series_id,
            horizon=parsed.horizon,
            quantile_levels=parsed.quantile_levels,
            models=parsed.models,
            method=parsed.method,
            window_id=parsed.window_id,
            source="llm",
        )
    except Exception:
        return None


def make_plan(
    question: str,
    dataset: str,
    series_ids: list[str],
    available_models: list[str],
    prefer_llm: bool = True,
) -> Plan:
    """Try the LLM planner first, always falling back to deterministic rules."""
    if prefer_llm:
        plan = llm_plan(question, dataset, series_ids, available_models)
        if plan is not None:
            return plan
    plan = rule_based_plan(question, dataset, series_ids)
    plan.models = [m for m in plan.models if m in available_models][:3]
    if len(plan.models) < 2:
        for fallback in available_models:
            if fallback not in plan.models and fallback != "chronos-lora-v1":
                plan.models.append(fallback)
            if len(plan.models) >= 2:
                break
    return plan
