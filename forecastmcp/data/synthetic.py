"""Deterministic synthetic supply-chain demand generator.

The generator is intentionally seeded and dependency-free beyond numpy/pandas so
that ``FORECASTMCP_OFFLINE=1`` runs, CI and tests are fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastmcp.config import DEFAULT_SEED, SYNTHETIC_N_SERIES

START = "2022-01-01"
END = "2025-12-31"
CATEGORIES = ("staples", "seasonal", "electronics", "apparel", "industrial")


def _sku_parameters(rng: np.random.Generator, idx: int) -> dict[str, float | str]:
    category = CATEGORIES[idx % len(CATEGORIES)]
    weekend_boost = {"staples": 0.02, "seasonal": 0.12, "electronics": 0.22,
                     "apparel": 0.18, "industrial": -0.25}[category]
    return {
        "series_id": f"S{idx + 1:03d}",
        "category": category,
        "base": float(rng.uniform(25, 480)),
        "trend": float(rng.normal(0.00018, 0.00022)),
        "weekly_amp": float(rng.uniform(0.03, 0.16)),
        "weekend_boost": float(weekend_boost),
        "annual_amp": float(rng.uniform(0.04, 0.30)),
        "annual_phase": float(rng.uniform(0, 2 * np.pi)),
        "noise_sigma": float(rng.uniform(0.04, 0.18)),
        "promo_prob": float(rng.uniform(0.012, 0.045)),
        "promo_lift": float(rng.uniform(0.25, 1.1)),
        "intermittent": 1.0 if rng.random() < 0.25 else 0.0,
        "zero_prob": float(rng.uniform(0.15, 0.55)),
        "lead_base": float(rng.integers(3, 15)),
        "price_base": float(rng.uniform(4, 120)),
        "noise_scale": float(rng.uniform(0.0, 1.0)),
    }


def generate_synthetic_supply(
    n_series: int = SYNTHETIC_N_SERIES,
    seed: int = DEFAULT_SEED,
    start: str = START,
    end: str = END,
) -> pd.DataFrame:
    """Generate a long-format panel of SKU-level daily demand.

    Each series combines a level, a linear trend, weekly and annual seasonality,
    occasional promotion spikes, optional intermittency (zero-inflated counts) and
    two business covariates: ``promo`` and ``lead_time_days``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    frames: list[pd.DataFrame] = []
    for idx in range(n_series):
        p = _sku_parameters(rng, idx)
        base = float(p["base"])
        lead = np.clip(
            p["lead_base"] + 2.0 * np.sin(2 * np.pi * t / 90.0) + rng.normal(0, 1.2, n),
            2,
            30,
        ).round()
        price = p["price_base"] * (1 + 0.0001 * t) * (1 + rng.normal(0, 0.01, n))
        promo = (rng.random(n) < p["promo_prob"]).astype(float)

        trend = base * (1 + p["trend"] * t)
        annual = 1 + p["annual_amp"] * np.sin(2 * np.pi * doy / 365.25 + p["annual_phase"])
        weekly = 1 + p["weekly_amp"] * np.sin(2 * np.pi * dow / 7.0 + idx)
        weekend = np.where(dow >= 5, 1 + p["weekend_boost"], 1.0)
        promo_mult = 1 + p["promo_lift"] * promo
        noise = rng.lognormal(mean=0, sigma=p["noise_sigma"], size=n)
        # Lead-time shock: long lead times correlate with mild demand suppression.
        lead_effect = 1 - 0.004 * np.maximum(0, lead - p["lead_base"])

        target = trend * annual * weekly * weekend * promo_mult * noise * lead_effect
        if p["intermittent"]:
            target = target * (rng.random(n) > p["zero_prob"])
        target = np.maximum(0, np.round(target))

        frame = pd.DataFrame(
            {
                "series_id": p["series_id"],
                "timestamp": dates,
                "target": target,
                "promo": promo,
                "lead_time_days": lead,
                "unit_price": np.round(price, 2),
                "category": p["category"],
            }
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
