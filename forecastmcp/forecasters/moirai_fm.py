"""Moirai adapter via ``uni2ts`` (optional ``[fm]`` extra).

Moirai is covariate-capable (past/future dynamic real features). The adapter is
import-guarded so a missing package produces ``available: false`` with the exact
ImportError rather than a crash.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np

from forecastmcp.forecasters.base import (
    BaseForecaster,
    ForecastRequest,
    ForecastResult,
    ModelUnavailable,
    check_import,
)

_LOCK = threading.Lock()
_MODULES: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _uni2ts_import() -> tuple[bool, str]:
    ok, reason = check_import("uni2ts")
    if not ok:
        return False, f"uni2ts not importable ({reason}); install the [fm] extra"
    return True, "ok"


class MoiraiForecaster(BaseForecaster):
    """Salesforce Moirai small through the uni2ts GluonTS predictor."""

    supports_covariates = True

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._load_error: str | None = None
        self._module = None

    def available(self) -> tuple[bool, str]:
        """Check the optional package and prior load failures."""
        ok, reason = _uni2ts_import()
        if not ok:
            return False, reason
        if self._load_error:
            return False, self._load_error
        return True, "uni2ts importable (untested checkpoint load)"

    def _ensure_module(self):
        if self._module is not None:
            return self._module
        ok, reason = self.available()
        if not ok:
            raise ModelUnavailable(reason)
        with _LOCK:
            if self.checkpoint not in _MODULES:
                try:
                    from uni2ts.model.moirai import MoiraiModule

                    _MODULES[self.checkpoint] = MoiraiModule.from_pretrained(self.checkpoint)
                except Exception as exc:
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    raise ModelUnavailable(self._load_error) from exc
            self._module = _MODULES[self.checkpoint]
        return self._module

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Forecast one series, passing covariates when supplied."""
        started = time.perf_counter()
        module = self._ensure_module()
        warnings: list[str] = []
        used_covariates = False
        try:
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai import MoiraiForecast

            target = np.asarray(req.context, dtype=np.float32)
            feat_dynamic_real = None
            past_feat_dynamic_real = None
            if req.future_covariates:
                names = sorted(req.future_covariates)
                future = np.vstack(
                    [np.asarray(req.future_covariates[n], dtype=np.float32) for n in names]
                )
                past = np.vstack(
                    [
                        np.asarray(
                            (req.past_covariates or {}).get(n, [0.0] * len(req.context)),
                            dtype=np.float32,
                        )
                        for n in names
                    ]
                )
                feat_dynamic_real = np.concatenate([past, future], axis=1)
                used_covariates = True
            elif req.past_covariates:
                names = sorted(req.past_covariates)
                past_feat_dynamic_real = np.vstack(
                    [np.asarray(req.past_covariates[n], dtype=np.float32) for n in names]
                )
                used_covariates = True

            model = MoiraiForecast(
                module=module,
                prediction_length=req.horizon,
                context_length=min(self.context_length, len(target)),
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=(
                    0 if feat_dynamic_real is None else feat_dynamic_real.shape[0]
                ),
                past_feat_dynamic_real_dim=(
                    0 if past_feat_dynamic_real is None else past_feat_dynamic_real.shape[0]
                ),
            )
            item = {"target": target, "start": "2024-01-01 00:00:00"}
            if feat_dynamic_real is not None:
                item["feat_dynamic_real"] = feat_dynamic_real
            if past_feat_dynamic_real is not None:
                item["past_feat_dynamic_real"] = past_feat_dynamic_real
            dataset = ListDataset([item], freq=req.freq)
            predictor = model.create_predictor(batch_size=1)
            forecasts = list(predictor.predict(dataset))
            fc = forecasts[0]
            point = np.asarray(fc.mean, dtype=float)[: req.horizon]
            q = np.vstack(
                [
                    np.asarray(fc.quantile(level), dtype=float)[: req.horizon]
                    for level in req.quantile_levels
                ]
            )
        except Exception as exc:
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return self.finalize(
            req,
            point,
            q,
            started,
            device=self.device(),
            n_params=None,
            warnings=warnings,
            used_covariates=used_covariates,
            revision=self.revision_for(),
            metadata={"family": "moirai", "checkpoint": self.checkpoint},
        )
