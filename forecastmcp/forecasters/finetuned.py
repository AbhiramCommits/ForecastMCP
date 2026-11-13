"""Registry adapter that serves a LoRA-adapted Chronos checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from forecastmcp.config import ROOT
from forecastmcp.forecasters.base import (
    ForecastRequest,
    ModelUnavailable,
    count_parameters,
)
from forecastmcp.forecasters.chronos_fm import ChronosForecaster, _checkpoint_cached

_ADAPTERS: dict[str, Any] = {}


class FinetunedChronosForecaster(ChronosForecaster):
    """Chronos T5 base plus a committed PEFT LoRA adapter directory."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self.adapter_dir = Path(config.get("checkpoint", ""))
        if not self.adapter_dir.is_absolute():
            self.adapter_dir = ROOT / self.adapter_dir
        self.base_checkpoint = str(config.get("base_checkpoint", "amazon/chronos-t5-tiny"))

    def adapter_ready(self) -> tuple[bool, str]:
        """Adapter directory must contain a PEFT config and weights."""
        if not self.adapter_dir.exists():
            return False, f"adapter directory {self.adapter_dir} does not exist"
        if not (self.adapter_dir / "adapter_config.json").exists():
            return False, f"adapter_config.json missing in {self.adapter_dir}"
        weights = list(self.adapter_dir.glob("adapter_model.*"))
        if not weights:
            return False, f"adapter weights missing in {self.adapter_dir}"
        return True, "adapter present"

    def available(self) -> tuple[bool, str]:
        """Check package, adapter files and base-checkpoint cache in offline mode."""
        from forecastmcp.config import is_offline
        from forecastmcp.forecasters.chronos_fm import _chronos_import

        ok, reason = _chronos_import()
        if not ok:
            return False, reason
        if self._load_error:
            return False, self._load_error
        ok, reason = self.adapter_ready()
        if not ok:
            return False, reason
        if is_offline():
            if self.base_checkpoint != "amazon/chronos-t5-tiny":
                return False, "offline mode: only the chronos-t5-tiny base is permitted"
            if not _checkpoint_cached(self.base_checkpoint):
                return False, f"offline: base checkpoint {self.base_checkpoint!r} not cached"
        return True, "base + adapter available"

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        ok, reason = self.available()
        if not ok:
            raise ModelUnavailable(reason)
        key = (self.checkpoint, self.device())
        if key in _ADAPTERS:
            self._pipeline = _ADAPTERS[key]
            return self._pipeline
        from chronos import ChronosPipeline
        from peft import PeftModel

        device = self.device()
        try:
            pipeline = ChronosPipeline.from_pretrained(
                self.base_checkpoint, device_map=device, dtype=torch.float32
            )
            base_t5 = pipeline.model.model
            adapted = PeftModel.from_pretrained(base_t5, str(self.adapter_dir))
            adapted.eval()
            pipeline.model.model = adapted
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise ModelUnavailable(self._load_error) from exc
        _ADAPTERS[key] = pipeline
        self._pipeline = pipeline
        self._n_params = count_parameters(adapted)
        return pipeline

    def predict_many(self, reqs: list[ForecastRequest]):
        """Serve predictions through the same Chronos sampling path as the base model."""
        results = super().predict_many(reqs)
        card_path = self.adapter_dir / "model_card.json"
        card: dict[str, Any] = {}
        if card_path.exists():
            card = json.loads(card_path.read_text())
        for res in results:
            res.model_name = self.name
            res.model_version = self.base_checkpoint
            res.revision = str(card.get("revision", self.revision))
            res.metadata.update(
                {
                    "family": "chronos-lora",
                    "adapter_dir": str(self.adapter_dir),
                    "trainable_params": card.get("trainable_params"),
                    "lora_r": (card.get("lora_config") or {}).get("r"),
                }
            )
        return results
