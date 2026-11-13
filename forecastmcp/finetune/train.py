"""LoRA fine-tuning of Chronos T5 with HF Trainer and quantile-loss early stopping."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from forecastmcp.config import FINETUNE_DIR, read_yaml
from forecastmcp.data.loaders import load_dataset
from forecastmcp.data.requests import build_window_request
from forecastmcp.data.splits import holdout_windows
from forecastmcp.eval.metrics import (
    crps_from_quantiles,
    interval_coverage,
    mase,
    weighted_quantile_loss,
)
from forecastmcp.finetune.dataset import WindowSample, build_training_windows, tokenize_windows
from forecastmcp.forecasters.base import resolve_revision


@dataclass
class TrainConfig:
    """Resolved fine-tuning configuration."""

    tag: str
    base_checkpoint: str
    dataset: str
    window_id: int
    context_length: int
    horizon: int
    max_windows: int
    val_fraction: float
    series_limit: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    patience: int
    seed: int
    device: str
    max_grad_norm: float

    @classmethod
    def load(cls, config_path: Path | str, tag: str | None = None,
             dataset: str | None = None) -> TrainConfig:
        """Load a YAML config and apply CLI overrides."""
        raw = read_yaml(config_path)
        if not raw:
            raise FileNotFoundError(f"fine-tune config not found: {config_path}")
        lora = raw.get("lora", {})
        training = raw.get("training", {})
        return cls(
            tag=tag or str(raw.get("tag", "v1")),
            base_checkpoint=str(raw.get("base_checkpoint", "amazon/chronos-t5-tiny")),
            dataset=dataset or str(raw.get("dataset", "synthetic_supply")),
            window_id=int(raw.get("window_id", 1)),
            context_length=int(raw.get("context_length", 512)),
            horizon=int(raw.get("horizon", 64)),
            max_windows=int(raw.get("max_windows", 320)),
            val_fraction=float(raw.get("val_fraction", 0.15)),
            series_limit=int(raw.get("series_limit", 60)),
            lora_r=int(lora.get("r", 4)),
            lora_alpha=int(lora.get("alpha", 8)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q", "k", "v", "o"])),
            epochs=int(training.get("epochs", 3)),
            batch_size=int(training.get("batch_size", 8)),
            learning_rate=float(training.get("learning_rate", 5e-4)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            warmup_ratio=float(training.get("warmup_ratio", 0.05)),
            patience=int(training.get("patience", 2)),
            seed=int(training.get("seed", 42)),
            device=str(training.get("device", "cpu")),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        )


class QuantileEvalCallback(TrainerCallback):
    """Measure val quantile loss by sampling forecasts, and stop when it stalls."""

    def __init__(
        self,
        pipeline: Any,
        val_windows: list[WindowSample],
        horizon: int,
        quantile_levels: list[float],
        patience: int,
        best_dir: Path,
    ):
        self.pipeline = pipeline
        self.windows = val_windows
        self.horizon = horizon
        self.levels = quantile_levels
        self.patience = patience
        self.best_dir = best_dir
        self.best = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.history: list[dict[str, Any]] = []
        self._epoch_losses: list[float] = []
        self.stopped_early = False

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        """Collect per-step training loss."""
        if logs and "loss" in logs:
            self._epoch_losses.append(float(logs["loss"]))

    def on_epoch_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        """Evaluate val quantile loss and implement patience-based early stopping."""
        contexts = [
            torch.tensor(np.asarray(w.context, dtype=np.float32)[-self.pipeline.tokenizer.config.context_length :])
            for w in self.windows
        ]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                q, _ = self.pipeline.predict_quantiles(
                    contexts,
                    prediction_length=self.horizon,
                    quantile_levels=self.levels,
                    num_samples=20,
                )
        finally:
            model.train(was_training)
        q = q.detach().cpu().numpy()
        losses = []
        for i, window in enumerate(self.windows):
            future = np.asarray(window.future, dtype=float)
            for j, level in enumerate(self.levels):
                diff = future - q[i, :, j]
                losses.append(np.where(diff >= 0, level * diff, (level - 1) * diff).mean())
        val_loss = float(np.mean(losses))
        train_loss = float(np.mean(self._epoch_losses)) if self._epoch_losses else float("nan")
        self._epoch_losses = []
        epoch = int(round(float(state.epoch or 0)))
        improved = val_loss < self.best - 1e-6
        if improved:
            self.best = val_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(self.best_dir)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True
                self.stopped_early = True
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_quantile_loss": val_loss,
                "improved": improved,
            }
        )


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def finetune(config: TrainConfig, config_path: Path | str | None = None) -> dict[str, Any]:
    """Train a LoRA adapter and write the adapter plus ``model_card.json``."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    started = time.perf_counter()

    from chronos import ChronosPipeline
    from peft import LoraConfig, get_peft_model

    out_dir = FINETUNE_DIR / config.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(config.dataset)
    windows = build_training_windows(
        df,
        dataset=config.dataset,
        window_id=config.window_id,
        context_length=config.context_length,
        horizon=config.horizon,
        max_windows=config.max_windows,
        val_fraction=config.val_fraction,
        series_limit=config.series_limit,
        seed=config.seed,
    )
    pipeline = ChronosPipeline.from_pretrained(
        config.base_checkpoint, device_map=config.device, dtype=torch.float32
    )
    tokenizer = pipeline.tokenizer
    if int(tokenizer.config.prediction_length) != config.horizon:
        raise ValueError(
            f"horizon {config.horizon} must equal the checkpoint prediction_length "
            f"{tokenizer.config.prediction_length} for teacher-forced training"
        )
    train_ds = tokenize_windows(windows.train, tokenizer)

    base_model = pipeline.model.model
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    peft_model = get_peft_model(base_model, lora)
    pipeline.model.model = peft_model
    trainable = int(sum(p.numel() for p in peft_model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in peft_model.parameters()))

    best_dir = out_dir / "best_adapter"
    levels = [0.1, 0.5, 0.9]
    callback = QuantileEvalCallback(
        pipeline, windows.val, config.horizon, levels, config.patience, best_dir
    )
    steps_per_epoch = max(1, (len(train_ds) + config.batch_size - 1) // config.batch_size)
    warmup_steps = int(round(steps_per_epoch * config.epochs * config.warmup_ratio))
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        use_cpu=config.device == "cpu",
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_ds,
        data_collator=_collate,
        callbacks=[callback],
    )
    train_result = trainer.train()
    wall = time.perf_counter() - started

    adapter_source = best_dir if (best_dir / "adapter_config.json").exists() else None
    if adapter_source is not None:
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            src = adapter_source / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
        shutil.rmtree(best_dir, ignore_errors=True)
    else:
        peft_model.save_pretrained(out_dir)

    revision = resolve_revision(config.base_checkpoint, fallback="main")
    card = {
        "tag": config.tag,
        "base_checkpoint": config.base_checkpoint,
        "revision": revision,
        "dataset": config.dataset,
        "holdout_window_id": config.window_id,
        "train_end": windows.meta["train_end"],
        "context_length": config.context_length,
        "horizon": config.horizon,
        "lora_config": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "task_type": "SEQ_2_SEQ_LM",
        },
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4),
        "n_train": len(windows.train),
        "n_val": len(windows.val),
        "n_series": windows.meta["n_series"],
        "training": {
            "epochs_configured": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "scheduler": "cosine",
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "patience": config.patience,
            "seed": config.seed,
            "device": config.device,
            "train_loss": float(train_result.training_loss),
        },
        "history": callback.history,
        "best_val_quantile_loss": callback.best,
        "best_epoch": callback.best_epoch,
        "stopped_early": callback.stopped_early,
        "wall_clock_seconds": round(wall, 3),
        "device": config.device,
        "covariates_used": windows.meta["covariates_used"],
        "config_path": str(config_path) if config_path else None,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    (out_dir / "loss_curve.json").write_text(json.dumps(callback.history, indent=2))
    return card


def _holdout_eval(model_name: str, dataset: str, series_ids: list[str],
                  n_windows: int) -> dict[str, float]:
    """Evaluate one registry model on frozen windows for a set of series."""
    from forecastmcp.forecasters.registry import get_forecaster

    forecaster = get_forecaster(model_name)
    ok, reason = forecaster.available()
    if not ok:
        raise RuntimeError(f"{model_name} unavailable: {reason}")
    df = load_dataset(dataset)
    windows = holdout_windows(dataset)[:n_windows]
    levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    collected: dict[str, list[float]] = {
        "WQL": [], "CRPS": [], "MASE": [], "coverage_80": [], "coverage_90": []
    }
    for window in windows:
        requests = [
            build_window_request(
                df, dataset, sid, horizon=window.horizon, window_id=window.window_id,
                covariates=False, quantiles=levels,
            )
            for sid in series_ids
        ]
        results = forecaster.predict_many([r.request for r in requests])
        for wr, res in zip(requests, results, strict=True):
            q = np.asarray(res.quantiles)
            y = np.asarray(wr.actual)
            lo80, hi80 = res.lower_upper(0.8)
            lo90, hi90 = res.lower_upper(0.9)
            collected["WQL"].append(weighted_quantile_loss(y, q, res.quantile_levels))
            collected["CRPS"].append(crps_from_quantiles(y, q, res.quantile_levels))
            collected["MASE"].append(mase(y, res.point, wr.request.context,
                                         24 if wr.window.freq.startswith("h") else 7))
            collected["coverage_80"].append(interval_coverage(y, lo80, hi80))
            collected["coverage_90"].append(interval_coverage(y, lo90, hi90))
    return {key: float(np.mean(vals)) for key, vals in collected.items()}


def compare_finetuned(tag: str, dataset: str | None = None, n_series: int = 20,
                      n_windows: int = 6) -> dict[str, Any]:
    """Compare zero-shot Chronos with the tagged adapter on the frozen holdout."""
    card_path = FINETUNE_DIR / tag / "model_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"no adapter/model card for tag {tag!r}")
    card = json.loads(card_path.read_text())
    dataset = dataset or card["dataset"]
    df = load_dataset(dataset)
    from forecastmcp.data.splits import eligible_series

    windows = holdout_windows(dataset)[:n_windows]
    eligible = eligible_series(df, windows)
    series_ids = eligible[: max(1, int(n_series))]
    if not series_ids:
        raise RuntimeError("no series cover the frozen holdout windows")
    adapter_name = f"chronos-lora-{tag}"
    zero = _holdout_eval("chronos", dataset, series_ids, n_windows)
    tuned = _holdout_eval(adapter_name, dataset, series_ids, n_windows)
    payload = {
        "tag": tag,
        "dataset": dataset,
        "n_series": len(series_ids),
        "n_windows": len(windows),
        "series_ids": series_ids,
        "models": {"chronos": zero, adapter_name: tuned},
        "delta": {
            key: tuned[key] - zero[key]
            for key in zero
        },
    }
    out = FINETUNE_DIR / tag / "compare.json"
    out.write_text(json.dumps(payload, indent=2))
    return payload
