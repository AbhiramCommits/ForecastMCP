"""Chronos T5 / Bolt adapters (sampling-based quantiles, zero-shot)."""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from forecastmcp.config import is_offline
from forecastmcp.forecasters.base import (
    BaseForecaster,
    ForecastRequest,
    ForecastResult,
    ModelUnavailable,
    check_import,
    count_parameters,
    resolve_revision,
)

logger = logging.getLogger(__name__)

_PIPELINES: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _chronos_import() -> tuple[bool, str]:
    ok, reason = check_import("chronos")
    if not ok:
        return False, f"chronos-forecasting not importable ({reason}); install the [fm] extra"
    return True, "ok"


def _checkpoint_cached(checkpoint: str) -> bool:
    """Return True when a HF snapshot config is present in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(checkpoint, "config.json")
        return isinstance(hit, str)
    except Exception:
        return False


class ChronosForecaster(BaseForecaster):
    """ChronosPipeline wrapper. T5 checkpoints sample paths then take quantiles."""

    supports_covariates = False

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self.num_samples = int(self.config.get("num_samples", 64))
        self._load_error: str | None = None
        self._pipeline = None
        self._n_params: int | None = None

    def available(self) -> tuple[bool, str]:
        """Check the package, the offline cache and any previous load failure."""
        ok, reason = _chronos_import()
        if not ok:
            return False, reason
        if self._load_error:
            return False, self._load_error
        if is_offline() and not _checkpoint_cached(self.checkpoint):
            return False, (
                f"offline mode: checkpoint {self.checkpoint!r} is not in the local Hugging Face "
                "cache; run once online to populate it"
            )
        if is_offline() and self.checkpoint != "amazon/chronos-t5-tiny":
            return False, "offline mode: only chronos-t5-tiny is permitted (CPU-cheap)"
        suffix = " (cached)" if _checkpoint_cached(self.checkpoint) else ""
        return True, f"chronos importable{suffix}"

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        ok, reason = self.available()
        if not ok:
            raise ModelUnavailable(reason)
        device = self.device()
        key = (self.checkpoint, device)
        with _LOCK:
            if key not in _PIPELINES:
                from chronos import ChronosBoltPipeline, ChronosPipeline

                started = time.perf_counter()
                try:
                    if "bolt" in self.checkpoint:
                        pipeline = ChronosBoltPipeline.from_pretrained(
                            self.checkpoint, device_map=device, dtype=torch.float32
                        )
                    else:
                        pipeline = ChronosPipeline.from_pretrained(
                            self.checkpoint, device_map=device, dtype=torch.float32
                        )
                except Exception as exc:
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    raise ModelUnavailable(self._load_error) from exc
                logger.info("loaded %s on %s in %.2fs", self.checkpoint, device, time.perf_counter() - started)
                _PIPELINES[key] = pipeline
            self._pipeline = _PIPELINES[key]
        self._n_params = count_parameters(self._pipeline.model)
        return self._pipeline

    def predict_many(self, reqs: list[ForecastRequest]) -> list[ForecastResult]:
        """Batch several series into one Chronos generation call."""
        if not reqs:
            return []
        started = time.perf_counter()
        pipeline = self._ensure_pipeline()
        device = self.device()
        torch.manual_seed(int(reqs[0].seed))
        np.random.seed(int(reqs[0].seed))
        contexts = [
            torch.tensor(np.asarray(r.context, dtype=np.float32)[-self.context_length :])
            for r in reqs
        ]
        try:
            q_tensor, mean_tensor = pipeline.predict_quantiles(
                contexts,
                prediction_length=reqs[0].horizon,
                quantile_levels=list(reqs[0].quantile_levels),
                num_samples=self.num_samples,
                temperature=1.0,
            )
        except Exception as exc:  # OOM at inference is reported, not hidden
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started
        quantiles = q_tensor.detach().cpu().numpy()
        means = mean_tensor.detach().cpu().numpy()
        revision = resolve_revision(self.checkpoint, self.revision)
        results = []
        for i, req in enumerate(reqs):
            warnings = []
            if req.past_covariates or req.future_covariates:
                warnings.append(
                    "chronos-t5 does not support exogenous covariates; inputs ignored "
                    "(used_covariates=false)"
                )
            if len(req.context) > self.context_length:
                warnings.append(f"context truncated to last {self.context_length} points")
            res = self.finalize(
                req,
                means[i],
                quantiles[i].T,
                time.perf_counter(),
                device=device,
                n_params=self._n_params,
                warnings=warnings,
                used_covariates=False,
                revision=revision,
                metadata={
                    "family": "chronos",
                    "checkpoint": self.checkpoint,
                    "num_samples": self.num_samples,
                    "amortized_over_batch": len(reqs),
                    "batch_seconds": round(elapsed, 4),
                    "sampling": "multinomial",
                },
            )
            res.latency_ms = round(elapsed * 1000.0 / max(len(reqs), 1), 3)
            results.append(res)
        return results

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Forecast one series."""
        return self.predict_many([req])[0]
