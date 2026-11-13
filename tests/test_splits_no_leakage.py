"""Frozen holdout integrity: no leakage and stable boundaries."""

from __future__ import annotations

import pandas as pd
import pytest

from forecastmcp.data.loaders import load_dataset
from forecastmcp.data.splits import (
    Window,
    holdout_windows,
    load_holdout,
    split_frame,
)


def test_windows_are_ordered_and_disjoint():
    """Windows step forward in time with no overlap between test spans."""
    windows = holdout_windows("synthetic_supply")
    assert len(windows) >= 2
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert prev.test_end_ts < nxt.test_start_ts
        assert prev.test_start_ts <= prev.test_end_ts
        assert prev.train_end_ts < prev.test_start_ts


def test_train_never_reaches_test():
    """Split frames respect the frozen boundary exactly."""
    df = load_dataset("synthetic_supply")
    for window in holdout_windows("synthetic_supply"):
        train, test = split_frame(df, window)
        assert pd.to_datetime(train["timestamp"]).max() <= window.train_end_ts
        assert pd.to_datetime(test["timestamp"]).min() >= window.test_start_ts
        assert pd.to_datetime(test["timestamp"]).max() <= window.test_end_ts
        assert len(test) == window.horizon * df["series_id"].nunique()


def test_frozen_yaml_is_the_source_of_truth():
    """Typed windows exactly match the committed YAML boundaries."""
    payload = load_holdout("synthetic_supply")
    windows = holdout_windows("synthetic_supply")
    assert payload["n_windows"] == len(windows)
    for entry, window in zip(payload["windows"], windows, strict=True):
        assert entry["train_end"] == window.train_end
        assert entry["test_start"] == window.test_start
        assert entry["test_end"] == window.test_end


def test_window_dataclass_types():
    """Window properties expose Timestamps for slicing."""
    window = holdout_windows("synthetic_supply")[0]
    assert isinstance(window, Window)
    assert isinstance(window.test_start_ts, pd.Timestamp)


def test_finetune_windows_never_cross_train_end():
    """Fine-tuning windows are built only from the frozen train portion."""
    from forecastmcp.finetune.dataset import build_training_windows

    df = load_dataset("synthetic_supply")
    built = build_training_windows(
        df,
        dataset="synthetic_supply",
        window_id=1,
        context_length=128,
        horizon=32,
        max_windows=40,
        series_limit=4,
        val_fraction=0.25,
    )
    boundary = pd.Timestamp(built.meta["train_end"])
    for sample in [*built.train, *built.val]:
        assert pd.Timestamp(sample.future_end) <= boundary
        assert pd.Timestamp(sample.context_end) < pd.Timestamp(sample.future_start)
    assert built.leakage_violations(built.meta["train_end"]) == []
    assert built.meta["covariates_supported"] is False
    with pytest.raises(ValueError):
        build_training_windows(df, window_id=99)
