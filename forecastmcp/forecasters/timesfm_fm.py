"""TimesFM adapter (optional ``[fm]`` extra).

The adapter is import-guarded: when ``timesfm`` is not installed, ``available()``
returns the verbatim ImportError and the registry reports it in the model table.
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
_MODELS: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _timesfm_import() -> tuple[bool, str]:
    ok, reason = check_import("timesfm")
    if not ok:
        return False, f"timesfm not importable ({reason}); install the [fm] extra"
    return True, "ok"


class TimesFMForecaster(BaseForecaster):
    """Google TimesFM zero-shot forecaster."""

    supports_covariates = False

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._load_error: str | None = None
        self._model = None

    def available(self) -> tuple[bool, str]:
        """Check the optional package and prior load failures."""
        ok, reason = _timesfm_import()
        if not ok:
            return False, reason
        if self._load_error:
            return False, self._load_error
        return True, "timesfm importable (untested checkpoint load)"

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        ok, reason = self.available()
        if not ok:
            raise ModelUnavailable(reason)
        with _LOCK:
            if self.checkpoint not in _MODELS:
                try:
                    import timesfm

                    hparams = timesfm.TimesFmHparams(
                        backend="cpu",
                        per_core_batch_size=32,
                        horizon_len=max(int(self.config.get("max_horizon", 128)), 128),
                        num_layers=50,
                        num_heads=16,
                        model_dims=1280,
                    )
                    ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.checkpoint)
                    _MODELS[self.checkpoint] = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
                except Exception as exc:
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    raise ModelUnavailable(self._load_error) from exc
            self._model = _MODELS[self.checkpoint]
        return self._model

    def predict(self, req: ForecastRequest) -> ForecastResult:
        """Forecast one series with TimesFM's native quantile head."""
        started = time.perf_counter()
        model = self._ensure_model()
        warnings = []
        if req.past_covariates or req.future_covariates:
            warnings.append(
                "timesfm 1.x has no exogenous covariate input in this adapter; "
                "inputs ignored (used_covariates=false)"
            )
        context = np.asarray(req.context, dtype=np.float32)[-self.context_length :]
        try:
            point, quant = model.forecast(inputs=[context], freq=[0])
        except Exception as exc:
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc
        point = np.asarray(point)[0][: req.horizon]
        quant = np.asarray(quant)[0][: req.horizon, :]
        # TimesFM returns deciles 0.1..0.9; interpolate to the requested levels.
        deciles = np.linspace(0.1, 0.9, quant.shape[-1])
        q = np.vstack(
            [np.interp(req.quantile_levels, deciles, quant[i, :]) for i in range(req.horizon)]
        ).T
        return self.finalize(
            req,
            point,
            q,
            started,
            device="cpu",
            n_params=None,
            warnings=warnings,
            used_covariates=False,
            revision=self.revision_for(),
            metadata={"family": "timesfm", "checkpoint": self.checkpoint},
        )
