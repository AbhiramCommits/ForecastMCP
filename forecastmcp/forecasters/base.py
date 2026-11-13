"""Typed request/result models and the forecaster protocol."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ModelUnavailable(RuntimeError):
    """Raised when a foundation-model checkpoint cannot be loaded.

    The message is the verbatim exception text so traces and model cards record
    the real reason (missing package, OOM, network failure, ...).
    """



class ForecastRequest(BaseModel):
    """Everything a forecaster needs for one series and one horizon.

    Covariates are split into past values (aligned with ``context``) and future
    values (length ``horizon``). Adapters that cannot consume them say so via
    ``used_covariates=False`` and a warning rather than failing.
    """

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(min_length=1)
    context: list[float] = Field(min_length=2)
    horizon: int = Field(gt=0, le=1024)
    quantile_levels: list[float] = Field(default_factory=lambda: list(DEFAULT_QUANTILES))
    past_covariates: dict[str, list[float]] | None = None
    future_covariates: dict[str, list[float]] | None = None
    timestamps: list[datetime] | None = None
    freq: str = "D"
    seed: int = 42

    @field_validator("context")
    @classmethod
    def _finite_context(cls, value: list[float]) -> list[float]:
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("context must contain only finite values")
        return value

    @field_validator("quantile_levels")
    @classmethod
    def _valid_quantiles(cls, value: list[float]) -> list[float]:
        levels = sorted({float(v) for v in value})
        if not levels or any(v <= 0 or v >= 1 for v in levels):
            raise ValueError("quantile_levels must be strictly between 0 and 1")
        return levels

    @model_validator(mode="after")
    def _covariate_shapes(self) -> ForecastRequest:
        if self.past_covariates:
            for name, values in self.past_covariates.items():
                if len(values) != len(self.context):
                    raise ValueError(
                        f"past covariate {name!r} has {len(values)} values, "
                        f"context has {len(self.context)}"
                    )
        if self.future_covariates:
            for name, values in self.future_covariates.items():
                if len(values) != self.horizon:
                    raise ValueError(
                        f"future covariate {name!r} has {len(values)} values, "
                        f"horizon is {self.horizon}"
                    )
        if self.timestamps is not None and len(self.timestamps) != len(self.context):
            raise ValueError("timestamps must align with the context window")
        return self


class ForecastResult(BaseModel):
    """A quantile forecast plus provenance needed for tracing and replay."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    model_name: str
    model_version: str
    revision: str = "unknown"
    point: list[float]
    quantile_levels: list[float]
    quantiles: list[list[float]]
    horizon: int
    latency_ms: float
    n_params: int | None = None
    device: str = "cpu"
    used_covariates: bool = False
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> ForecastResult:
        if len(self.point) != self.horizon:
            raise ValueError("point forecast length must equal horizon")
        if len(self.quantiles) != len(self.quantile_levels):
            raise ValueError("one quantile row per quantile level is required")
        if any(len(row) != self.horizon for row in self.quantiles):
            raise ValueError("quantile row length must equal horizon")
        matrix = np.asarray(self.quantiles, dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("quantiles must be finite")
        if not np.all(np.diff(matrix, axis=0) >= 0):
            raise ValueError("quantiles must be non-decreasing across levels at each step")
        return self

    def quantile(self, level: float) -> list[float]:
        """Return the forecast at one requested quantile level."""
        idx = int(np.argmin(np.abs(np.asarray(self.quantile_levels) - float(level))))
        return list(self.quantiles[idx])

    def lower_upper(self, nominal: float) -> tuple[list[float], list[float]]:
        """Return the central interval bounds for a nominal coverage (e.g. 0.8)."""
        alpha = (1 - nominal) / 2
        return self.quantile(alpha), self.quantile(1 - alpha)


@runtime_checkable
class Forecaster(Protocol):
    """Structural interface every adapter implements."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Return ``(available, reason)`` without raising on missing capabilities."""

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Produce a quantile forecast for one request."""


class BaseForecaster(ABC):
    """Shared helpers for concrete adapters."""

    name: str = "base"
    supports_covariates: bool = False

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = dict(config or {})
        self.checkpoint = self.config.get("checkpoint", "unknown")
        self.revision = str(self.config.get("revision", "unknown"))
        self.context_length = int(self.config.get("context_length", 512))
        self.requested_device = str(self.config.get("device", "cpu"))

    def available(self) -> tuple[bool, str]:
        """Default: the adapter is importable and configured."""
        return True, "ok"

    @abstractmethod
    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Forecast one series."""

    def predict_many(self, reqs: list[ForecastRequest]) -> list[ForecastResult]:
        """Forecast several series; adapters may override for true batching."""
        return [self.predict(req) for req in reqs]

    # -- utilities ---------------------------------------------------------
    def device(self, torch_module=torch) -> str:
        """Resolve the configured device ('auto' prefers mps, then cuda, then cpu)."""
        requested = self.requested_device
        if requested == "auto":
            if torch_module.backends.mps.is_available():
                return "mps"
            if torch_module.cuda.is_available():
                return "cuda"
            return "cpu"
        return requested

    def revision_for(self, checkpoint: str | None = None) -> str:
        """Resolve a Hugging Face commit hash, with an offline-safe fallback."""
        return resolve_revision(checkpoint or self.checkpoint, fallback=self.revision)

    def finalize(
        self,
        req: ForecastRequest,
        point: np.ndarray,
        quantiles: np.ndarray,
        started: float,
        *,
        device: str = "cpu",
        n_params: int | None = None,
        warnings: list[str] | None = None,
        used_covariates: bool = False,
        revision: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ForecastResult:
        """Clamp exotic outputs into a schema-valid ``ForecastResult``."""
        warns = list(warnings or [])
        levels = np.asarray(req.quantile_levels, dtype=float)
        q = np.asarray(quantiles, dtype=float).reshape(len(levels), req.horizon)
        crossings = 0
        fixed = q.copy()
        for step in range(req.horizon):
            column = fixed[:, step]
            finite = np.isfinite(column)
            if not np.all(finite):
                idx = np.flatnonzero(finite)
                fixed[:, step] = (
                    np.interp(np.arange(len(column)), idx, column[idx]) if len(idx) else 0.0
                )
                warns.append(f"non-finite quantiles at step {step} were interpolated")
                column = fixed[:, step]
            if np.any(np.diff(column) < 0):
                crossings += 1
            fixed[:, step] = np.sort(column)
        if crossings:
            warns.append(f"quantile crossing at {crossings}/{req.horizon} steps; sorted")
        point_arr = np.asarray(point, dtype=float).reshape(req.horizon)
        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        point_arr = np.where(np.isfinite(point_arr), point_arr, fixed[median_idx])
        latency = (time.perf_counter() - started) * 1000.0
        return ForecastResult(
            series_id=req.series_id,
            model_name=self.name,
            model_version=str(self.config.get("checkpoint", self.name)),
            revision=revision if revision is not None else self.revision,
            point=[float(v) for v in point_arr],
            quantile_levels=[float(v) for v in levels],
            quantiles=[[float(v) for v in row] for row in fixed],
            horizon=req.horizon,
            latency_ms=round(latency, 3),
            n_params=n_params,
            device=device,
            used_covariates=used_covariates,
            warnings=warns,
            metadata=metadata or {},
        )


def resolve_revision(checkpoint: str, fallback: str = "unknown") -> str:
    """Look up the Hugging Face commit SHA for a checkpoint.

    Returns the configured fallback for local paths, missing packages or offline
    machines; never raises.
    """
    from pathlib import Path

    if "/" not in checkpoint or Path(checkpoint).exists():
        return fallback
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(checkpoint, timeout=5)
        return str(info.sha)
    except Exception:
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == checkpoint and repo.refs:
                    ref = next(iter(repo.refs.values()))
                    return str(ref.commit_hash)
        except Exception:
            pass
        return fallback


def check_import(module: str) -> tuple[bool, str]:
    """Try importing an optional module and return its failure reason verbatim."""
    try:
        __import__(module)
        return True, "ok"
    except Exception as exc:  # ImportError, OSError from native libs, ...
        return False, f"{type(exc).__name__}: {exc}"


def count_parameters(model: Any) -> int:
    """Count trainable + frozen parameters of a torch module."""
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return 0
