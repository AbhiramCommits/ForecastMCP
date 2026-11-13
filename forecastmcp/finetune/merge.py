"""Optional merge of a trained LoRA adapter into the base Chronos weights."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from forecastmcp.config import FINETUNE_DIR


def merge_adapter(tag: str, out_dir: Path | str | None = None) -> Path:
    """Merge ``artifacts/finetune/<tag>`` into a standalone HF checkpoint.

    The merged checkpoint is large and is intentionally git-ignored; only the
    adapter and model card are committed. Returns the output directory.
    """
    from chronos import ChronosPipeline
    from peft import PeftModel

    adapter_dir = FINETUNE_DIR / tag
    card_path = adapter_dir / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"no model_card.json for tag {tag!r}")
    card = json.loads(card_path.read_text())
    base = str(card["base_checkpoint"])
    target = Path(out_dir) if out_dir else adapter_dir / "merged"
    target.mkdir(parents=True, exist_ok=True)

    pipeline = ChronosPipeline.from_pretrained(base, device_map="cpu", dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(pipeline.model.model, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(target)
    return target
