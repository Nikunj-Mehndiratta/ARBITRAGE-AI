"""Shared fixtures: a synthetic, genuinely cointegrated pair (no network needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbitrage.config import load_config  # noqa: E402

TRUE_BETA = 1.5


def make_frames(n: int = 600, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Raw bars for AAA, BBB (= cointegrated with AAA) and MKT, in data_loader's format."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    log_a = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for i in range(1, n):  # stationary AR(1) spread
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.01)
    log_b = 0.3 + TRUE_BETA * log_a + noise
    log_m = np.log(300) + np.cumsum(rng.normal(0, 0.008, n))

    frames = {}
    for ticker, lp in [("AAA", log_a), ("BBB", log_b), ("MKT", log_m)]:
        price = np.exp(lp)
        frames[ticker] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "adj_close": price,
                "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            },
            index=pd.Index(dates, name="date"),
        )
    return frames


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return make_frames()


@pytest.fixture
def cfg() -> dict:
    cfg = load_config()
    cfg["data"]["candidate_pairs"] = [["AAA", "BBB"]]
    cfg["data"]["market_ticker"] = "MKT"
    cfg["pair_selection"]["force_pair"] = None
    return cfg


@pytest.fixture
def panel(frames) -> pd.DataFrame:
    from arbitrage.preprocessing import build_panel

    return build_panel(frames)[0]
