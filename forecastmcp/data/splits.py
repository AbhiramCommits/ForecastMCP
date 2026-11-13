"""Frozen rolling-origin holdout splits.

Split boundaries are written once to ``configs/holdout.yaml`` and thereafter read
from disk by every experiment. Nothing in the benchmark derives its own splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecastmcp.config import (
    CONFIGS_DIR,
    DEFAULT_HORIZON,
    DEFAULT_N_WINDOWS,
    dataset_spec,
    read_yaml,
    write_yaml,
)

HOLDOUT_PATH = CONFIGS_DIR / "holdout.yaml"


def _step(freq: str) -> pd.Timedelta:
    return pd.Timedelta(hours=1) if freq.lower().startswith("h") else pd.Timedelta(days=1)


@dataclass(frozen=True)
class Window:
    """A single rolling-origin split with absolute boundaries."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon: int
    freq: str

    @property
    def test_start_ts(self) -> pd.Timestamp:
        """Test start as a Timestamp."""
        return pd.Timestamp(self.test_start)

    @property
    def test_end_ts(self) -> pd.Timestamp:
        """Test end as a Timestamp."""
        return pd.Timestamp(self.test_end)

    @property
    def train_end_ts(self) -> pd.Timestamp:
        """Train end as a Timestamp."""
        return pd.Timestamp(self.train_end)

    def to_dict(self) -> dict:
        """Serialise to the on-disk YAML representation."""
        return {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def make_holdout(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    n_windows: int = DEFAULT_N_WINDOWS,
    freq: str | None = None,
) -> list[Window]:
    """Build non-overlapping rolling-origin windows anchored at the panel end."""
    if horizon <= 0 or n_windows <= 0:
        raise ValueError("horizon and n_windows must be positive")
    spec_freq = freq or dataset_spec(str(df["dataset"].iloc[0])).freq
    step = _step(spec_freq)
    end = pd.to_datetime(df["timestamp"]).max().normalize()
    start = pd.to_datetime(df["timestamp"]).min().normalize()

    windows: list[Window] = []
    for w in range(1, n_windows + 1):
        test_end = end - (n_windows - w) * horizon * step
        test_start = test_end - (horizon - 1) * step
        train_end = test_start - step
        if train_end <= start:
            raise ValueError("not enough history for the requested holdout geometry")
        windows.append(
            Window(
                window_id=w,
                train_start=str(start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
                horizon=horizon,
                freq=spec_freq,
            )
        )
    return windows


def freeze_holdout(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    n_windows: int = DEFAULT_N_WINDOWS,
    path: Path | str = HOLDOUT_PATH,
) -> dict:
    """Write split boundaries for a dataset if not already frozen.

    Returns the dataset's entry from the frozen file. Existing entries are never
    rewritten: the holdout is immutable once committed.
    """
    path = Path(path)
    name = str(df["dataset"].iloc[0])
    payload = read_yaml(path, default=None) or {"version": 1, "frozen": True, "datasets": {}}
    payload.setdefault("datasets", {})
    if name in payload["datasets"]:
        return payload["datasets"][name]

    windows = make_holdout(df, horizon=horizon, n_windows=n_windows)
    entry = {
        "freq": windows[0].freq,
        "horizon": horizon,
        "n_windows": n_windows,
        "n_series": int(df["series_id"].nunique()),
        "history_start": str(pd.to_datetime(df["timestamp"]).min()),
        "history_end": str(pd.to_datetime(df["timestamp"]).max()),
        "windows": [w.to_dict() for w in windows],
    }
    payload["datasets"][name] = entry
    write_yaml(path, payload)
    return entry


def load_holdout(dataset: str | None = None, path: Path | str = HOLDOUT_PATH) -> dict:
    """Read the frozen holdout file (optionally one dataset's entry)."""
    payload = read_yaml(path, default=None)
    if not payload or "datasets" not in payload:
        raise FileNotFoundError(
            f"frozen holdout not found at {path}; run `forecastmcp data prepare` first"
        )
    if dataset is None:
        return payload
    if dataset not in payload["datasets"]:
        raise KeyError(f"dataset {dataset!r} is not present in the frozen holdout")
    return payload["datasets"][dataset]


def holdout_windows(dataset: str, path: Path | str = HOLDOUT_PATH) -> list[Window]:
    """Return typed Window objects for a frozen dataset."""
    entry = load_holdout(dataset, path)
    return [
        Window(
            window_id=int(w["window_id"]),
            train_start=str(w["train_start"]),
            train_end=str(w["train_end"]),
            test_start=str(w["test_start"]),
            test_end=str(w["test_end"]),
            horizon=int(entry["horizon"]),
            freq=str(entry["freq"]),
        )
        for w in entry["windows"]
    ]


def split_frame(df: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long frame into train and test portions for one window."""
    ts = pd.to_datetime(df["timestamp"])
    train = df[ts <= window.train_end_ts].copy()
    test = df[(ts >= window.test_start_ts) & (ts <= window.test_end_ts)].copy()
    return train, test


def eligible_series(df: pd.DataFrame, windows: list[Window]) -> list[str]:
    """Series covering every frozen test window (keeps evaluation balanced)."""
    first_test = min(w.test_start_ts for w in windows)
    last_test = max(w.test_end_ts for w in windows)
    ts = pd.to_datetime(df["timestamp"])
    covered = df[(ts >= first_test) & (ts <= last_test)]
    counts = covered.groupby(covered["series_id"].astype(str))["timestamp"].nunique()
    needed = sum(w.horizon for w in windows)
    return sorted(counts[counts >= needed].index.tolist())
