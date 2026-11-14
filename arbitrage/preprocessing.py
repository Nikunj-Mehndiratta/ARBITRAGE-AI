"""Validation and alignment of raw bars, and the chronological train / validation / test split."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def build_panel(
    frames: dict[str, pd.DataFrame], max_abs_daily_return: float = 0.25
) -> tuple[pd.DataFrame, dict]:
    """Align all tickers on common trading dates and remove invalid observations.

    Returns a wide panel with columns ``{TICKER}_adj_close``, ``{TICKER}_close`` and
    ``{TICKER}_volume`` plus a boolean ``quality_ok`` column, and a validation report.
    """
    report: dict = {"tickers": {}, "rows_per_ticker_raw": {}}
    cleaned = {}
    for ticker, frame in frames.items():
        f = frame.copy()
        f.index = pd.to_datetime(f.index)
        raw_rows = len(f)
        f = f[~f.index.duplicated(keep="last")].sort_index()
        duplicates = raw_rows - len(f)

        price_ok = f["adj_close"].notna() & (f["adj_close"] > 0) & f["close"].notna() & (f["close"] > 0)
        volume_ok = f["volume"].notna() & (f["volume"] > 0)
        invalid = int((~(price_ok & volume_ok)).sum())
        f = f[price_ok & volume_ok]

        cleaned[ticker] = f[["adj_close", "close", "volume"]].add_prefix(f"{ticker}_")
        report["tickers"][ticker] = {
            "raw_rows": raw_rows,
            "duplicate_dates_removed": duplicates,
            "invalid_rows_removed": invalid,
        }

    # Inner join: keep only sessions where every ticker has a valid bar.
    panel = pd.concat(cleaned.values(), axis=1, join="inner").sort_index()
    panel.index.name = "date"

    # Flag (do not silently delete) suspicious one-day moves; they are excluded from signals.
    quality_ok = pd.Series(True, index=panel.index)
    suspicious = {}
    for ticker in frames:
        ret = np.log(panel[f"{ticker}_adj_close"]).diff().abs()
        bad = ret > max_abs_daily_return
        suspicious[ticker] = int(bad.sum())
        quality_ok &= ~bad
    panel["quality_ok"] = quality_ok

    gaps = panel.index.to_series().diff().dt.days.dropna()
    report.update(
        {
            "aligned_rows": int(len(panel)),
            "first_date": panel.index.min().strftime("%Y-%m-%d"),
            "last_date": panel.index.max().strftime("%Y-%m-%d"),
            "max_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
            "suspicious_moves_flagged": suspicious,
            "quality_flagged_rows": int((~quality_ok).sum()),
        }
    )
    return panel, report


@dataclass(frozen=True)
class Split:
    """Chronological split expressed as integer row positions.

    train = [0, train_end)   validation = [train_end, val_end)   test = [val_end, n)

    Leakage across a boundary is prevented by *purging*: a sample is only used for
    fitting if its forward-looking label is fully resolved before the boundary
    (see ``label_end_pos`` in ``feature_engineering``). That leaves an embargo of
    ``horizon + execution_lag`` sessions on the earlier side of each boundary.
    """

    n: int
    train_end: int
    val_end: int

    def name_of(self, pos: int) -> str:
        if pos < self.train_end:
            return "train"
        if pos < self.val_end:
            return "validation"
        return "test"

    def labels(self) -> np.ndarray:
        out = np.empty(self.n, dtype=object)
        out[: self.train_end] = "train"
        out[self.train_end : self.val_end] = "validation"
        out[self.val_end :] = "test"
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def chronological_split(n: int, train_frac: float, val_frac: float) -> Split:
    """Split ``n`` ordered observations into train / validation / test without shuffling."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("Not enough observations for a three-way split")
    return Split(n=n, train_end=train_end, val_end=val_end)
