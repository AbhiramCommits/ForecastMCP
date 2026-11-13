"""Leakage-checked training windows and Chronos tokenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from forecastmcp.data.splits import holdout_windows


@dataclass
class WindowSample:
    """One (context, horizon) training example with its timestamps."""

    series_id: str
    context: list[float]
    future: list[float]
    context_start: str
    context_end: str
    future_start: str
    future_end: str


@dataclass
class WindowSet:
    """Train/validation windows plus the provenance needed in the model card."""

    train: list[WindowSample]
    val: list[WindowSample]
    meta: dict[str, Any] = field(default_factory=dict)

    def leakage_violations(self, train_end: str) -> list[str]:
        """Return samples whose future extends past the frozen train boundary."""
        boundary = pd.Timestamp(train_end)
        return [
            f"{s.series_id}:{s.future_end}"
            for s in [*self.train, *self.val]
            if pd.Timestamp(s.future_end) > boundary
        ]


def build_training_windows(
    df: pd.DataFrame,
    dataset: str = "synthetic_supply",
    window_id: int = 1,
    context_length: int = 512,
    horizon: int = 64,
    max_windows: int = 320,
    val_fraction: float = 0.15,
    series_limit: int = 60,
    seed: int = 42,
) -> WindowSet:
    """Build windows ONLY from the train portion of a frozen holdout split.

    Chronos T5 accepts no exogenous channels, so covariates are deliberately not
    fed; the model card records ``covariates_supported: false``. All future
    timestamps are asserted to be at or before the frozen ``train_end``.
    """
    windows = holdout_windows(dataset)
    window = next((w for w in windows if w.window_id == int(window_id)), None)
    if window is None:
        raise ValueError(f"window {window_id} not in frozen holdout")
    train_end = window.train_end_ts
    ts = pd.to_datetime(df["timestamp"])
    train_df = df[ts <= train_end].sort_values(["series_id", "timestamp"])

    ids = sorted(train_df["series_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    if series_limit and len(ids) > series_limit:
        picked = sorted(rng.choice(ids, size=series_limit, replace=False).tolist())
    else:
        picked = ids

    samples: list[WindowSample] = []
    for series_id in picked:
        series = train_df[train_df["series_id"].astype(str) == series_id].reset_index(drop=True)
        n = len(series)
        if n < context_length + horizon:
            continue
        for start in range(0, n - context_length - horizon + 1, horizon):
            ctx = series.iloc[start : start + context_length]
            fut = series.iloc[start + context_length : start + context_length + horizon]
            if len(fut) < horizon:
                break
            if pd.Timestamp(fut["timestamp"].iloc[-1]) > train_end:
                break
            samples.append(
                WindowSample(
                    series_id=series_id,
                    context=ctx["target"].to_numpy(dtype=float).tolist(),
                    future=fut["target"].to_numpy(dtype=float).tolist(),
                    context_start=str(pd.Timestamp(ctx["timestamp"].iloc[0])),
                    context_end=str(pd.Timestamp(ctx["timestamp"].iloc[-1])),
                    future_start=str(pd.Timestamp(fut["timestamp"].iloc[0])),
                    future_end=str(pd.Timestamp(fut["timestamp"].iloc[-1])),
                )
            )
    if len(samples) < 10:
        raise ValueError(f"only {len(samples)} training windows could be built")
    if len(samples) > max_windows:
        idx = np.sort(rng.choice(len(samples), size=max_windows, replace=False))
        samples = [samples[i] for i in idx]
    samples.sort(key=lambda s: (s.future_end, s.series_id))

    n_val = max(8, int(round(len(samples) * val_fraction)))
    n_val = min(n_val, len(samples) // 3)
    val = samples[-n_val:]
    train = samples[:-n_val]
    result = WindowSet(
        train=train,
        val=val,
        meta={
            "dataset": dataset,
            "holdout_window_id": int(window_id),
            "train_end": str(train_end),
            "context_length": int(context_length),
            "horizon": int(horizon),
            "n_series": len(picked),
            "n_train": len(train),
            "n_val": len(val),
            "max_windows": int(max_windows),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "covariates_supported": False,
            "covariates_used": [],
        },
    )
    violations = result.leakage_violations(str(train_end))
    if violations:
        raise AssertionError(f"leakage: {violations[:5]}")
    return result


def tokenize_windows(windows: list[WindowSample], tokenizer: Any) -> list[dict[str, torch.Tensor]]:
    """Tokenize contexts and futures with Chronos' mean-scale uniform bins."""
    if not windows:
        return []
    contexts = torch.tensor(np.asarray([w.context for w in windows], dtype=np.float32))
    futures = torch.tensor(np.asarray([w.future for w in windows], dtype=np.float32))
    input_ids, attention_mask, scale = tokenizer.context_input_transform(contexts)
    labels, label_mask = tokenizer.label_input_transform(futures, scale)
    labels = labels.masked_fill(~label_mask, -100)
    return [
        {
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
            "labels": labels[i],
        }
        for i in range(len(windows))
    ]
