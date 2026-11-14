"""Market data layer: download daily bars from Yahoo Finance and cache them on disk.

The cache in ``data/raw`` is the research dataset. ``metadata.json`` records where the
data came from and when it was retrieved, so an experiment can be reproduced exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def _download_one_batch(tickers: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,  # keep both the raw close and the adjusted close
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            sub = data.xs(ticker, axis=1, level=-1)
        else:  # single ticker download
            sub = data
        sub = sub.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
        sub = sub[[c for c in RAW_COLUMNS if c in sub.columns]].dropna(how="all")
        sub.index = pd.to_datetime(sub.index).tz_localize(None)
        sub.index.name = "date"
        if sub.empty:
            raise RuntimeError(f"No rows returned for {ticker}")
        frames[ticker] = sub
    return frames


def load_cached(tickers: list[str], raw_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Return cached bars, or None if any ticker is missing from the cache."""
    frames = {}
    for ticker in tickers:
        path = raw_dir / f"{ticker}.csv"
        if not path.exists():
            return None
        frames[ticker] = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return frames


def load_metadata(raw_dir: Path) -> dict:
    path = raw_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    raw_dir: Path,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load daily bars for ``tickers``; download only if the cache is missing or ``refresh``.

    Returns ``(frames, metadata)`` where frames maps ticker -> DataFrame indexed by date.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        cached = load_cached(tickers, raw_dir)
        if cached is not None:
            return cached, load_metadata(raw_dir)

    try:
        frames = _download_one_batch(tickers, start, end)
    except Exception as exc:  # network down, rate limit, ...
        cached = load_cached(tickers, raw_dir)
        if cached is not None:
            meta = load_metadata(raw_dir)
            meta["warning"] = f"Download failed ({exc}); using cached data."
            return cached, meta
        raise RuntimeError(
            f"Could not download market data and no cache exists in {raw_dir}: {exc}"
        ) from exc

    for ticker, frame in frames.items():
        frame.to_csv(raw_dir / f"{ticker}.csv")

    try:
        import yfinance as yf

        yf_version = yf.__version__
    except Exception:
        yf_version = "unknown"

    metadata = {
        "source": "Yahoo Finance via the yfinance Python package",
        "yfinance_version": yf_version,
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "interval": "1d",
        "requested_start": start,
        "requested_end": end,
        "price_field_used": "adj_close (split- and dividend-adjusted)",
        "tickers": {
            t: {
                "rows": int(len(f)),
                "first_date": f.index.min().strftime("%Y-%m-%d"),
                "last_date": f.index.max().strftime("%Y-%m-%d"),
            }
            for t, f in frames.items()
        },
        "note": "Research dataset for a paper-trading prototype. Not a live or institutional feed.",
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frames, metadata
