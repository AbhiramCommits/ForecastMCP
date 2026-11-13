"""LoRA adapter artifacts, registry entry and tokenization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forecastmcp.config import FINETUNE_DIR
from forecastmcp.forecasters.base import ForecastRequest
from forecastmcp.forecasters.registry import availability, get_forecaster


def _card() -> dict:
    path = FINETUNE_DIR / "v1" / "model_card.json"
    if not path.exists():
        pytest.skip("no fine-tuned adapter in this checkout")
    return json.loads(path.read_text())


def test_model_card_records_provenance():
    """The committed model card carries base revision, LoRA config and loss curve."""
    card = _card()
    assert card["base_checkpoint"] == "amazon/chronos-t5-tiny"
    assert card["revision"] and card["revision"] != "main"
    assert card["trainable_params"] > 0
    assert card["total_params"] > card["trainable_params"]
    assert card["lora_config"]["target_modules"] == ["q", "k", "v", "o"]
    assert card["history"] and "val_quantile_loss" in card["history"][0]
    assert card["wall_clock_seconds"] > 0
    assert card["train_end"]


def test_adapter_is_registered_and_serves_predictions():
    """chronos-lora-v1 loads through the registry and returns valid quantiles."""
    _card()
    ok, reason = availability("chronos-lora-v1")
    if not ok:
        pytest.skip(reason)
    forecaster = get_forecaster("chronos-lora-v1")
    rng = np.random.default_rng(3)
    context = (100 + 10 * np.sin(np.arange(120) / 7) + rng.normal(0, 2, 120)).tolist()
    result = forecaster.predict(
        ForecastRequest(series_id="S001", context=context, horizon=7,
                        quantile_levels=[0.1, 0.5, 0.9], seed=1)
    )
    matrix = np.asarray(result.quantiles)
    assert matrix.shape == (3, 7)
    assert np.all(np.diff(matrix, axis=0) >= 0)
    assert result.model_name == "chronos-lora-v1"
    assert result.metadata["family"] == "chronos-lora"
    assert result.metadata["trainable_params"] > 0


def test_tokenization_shapes_and_leakage_safe_windows():
    """Chronos tokenization matches the architecture's expected decoder length."""
    _card()
    from chronos import ChronosPipeline

    from forecastmcp.data.loaders import load_dataset
    from forecastmcp.finetune.dataset import build_training_windows, tokenize_windows

    ok, _ = availability("chronos")
    if not ok:
        pytest.skip("chronos unavailable")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny", device_map="cpu", dtype=torch.float32
    )
    windows = build_training_windows(
        load_dataset("synthetic_supply"),
        context_length=512,
        horizon=64,
        max_windows=4,
        series_limit=2,
    )
    tokenized = tokenize_windows(windows.val or windows.train, pipeline.tokenizer)
    assert tokenized
    sample = tokenized[0]
    assert sample["input_ids"].shape[-1] >= 512
    assert sample["labels"].shape[-1] == 65  # 64 future bins + EOS
    assert int((sample["labels"] != -100).sum()) > 0
    assert Path(FINETUNE_DIR / "v1" / "adapter_model.safetensors").exists()
