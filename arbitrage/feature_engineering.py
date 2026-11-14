"""Feature engineering and label generation.

Every feature at row t is computed from data up to and including the close of day t.
Labels look forward (that is their job) and are therefore never used as features; each
labelled row records ``label_end_pos``, the last row its outcome depends on, so the
pipeline can purge samples whose outcome would straddle a split boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtester import TRADING_DAYS, TradeParams, simulate_trade
from .statistical_engine import HedgeModel, compute_spread, rolling_zscore

FEATURE_COLUMNS = [
    "z_score",            # current spread deviation
    "abs_z",              # size of the deviation regardless of side
    "z_change_1d",        # speed of divergence (1 session)
    "z_change_5d",        # speed of divergence (5 sessions)
    "spread_vol_20",      # stability of the relationship
    "spread_vol_ratio",   # short-term vs long-term spread volatility
    "rolling_corr_60",    # relationship between the two stocks
    "rolling_beta_120",   # recent hedge ratio (relative exposure)
    "rsi_a",              # recent momentum of stock A
    "rsi_b",              # recent momentum of stock B
    "rel_volume_a",       # unusual trading activity in A
    "rel_volume_b",       # unusual trading activity in B
    "market_vol_20",      # broader market conditions
    "spread_momentum_5",  # direction and speed of the spread move
]

FEATURE_DESCRIPTIONS = {
    "z_score": "Current spread z-score",
    "abs_z": "Absolute z-score",
    "z_change_1d": "Z-score change, 1 session",
    "z_change_5d": "Z-score change, 5 sessions",
    "spread_vol_20": "Spread volatility, 20 sessions",
    "spread_vol_ratio": "Spread volatility, 20 vs 60 sessions",
    "rolling_corr_60": "Return correlation, 60 sessions",
    "rolling_beta_120": "Rolling hedge ratio, 120 sessions",
    "rsi_a": "RSI(14) of stock A",
    "rsi_b": "RSI(14) of stock B",
    "rel_volume_a": "Relative volume of stock A",
    "rel_volume_b": "Relative volume of stock B",
    "market_vol_20": "Market volatility, 20 sessions (annualised)",
    "spread_momentum_5": "Spread momentum, 5 sessions",
}


def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = price.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0.0, 100.0).where(gain.notna() & loss.notna())


def build_features(panel: pd.DataFrame, hedge: HedgeModel, market_ticker: str, cfg: dict) -> pd.DataFrame:
    """Spread, z-score and the model's input features for every session in ``panel``."""
    a, b = hedge.ticker_a, hedge.ticker_b
    window = int(cfg["statistical"]["zscore_window"])

    out = pd.DataFrame(index=panel.index)
    out["price_a"] = panel[f"{a}_adj_close"]
    out["price_b"] = panel[f"{b}_adj_close"]
    out["log_a"] = np.log(out["price_a"])
    out["log_b"] = np.log(out["price_b"])
    out["spread"] = compute_spread(out["log_a"], out["log_b"], hedge)
    out = out.join(rolling_zscore(out["spread"], window))

    ret_a = out["log_a"].diff()
    ret_b = out["log_b"].diff()
    spread_diff = out["spread"].diff()

    out["abs_z"] = out["z_score"].abs()
    out["z_change_1d"] = out["z_score"].diff(1)
    out["z_change_5d"] = out["z_score"].diff(5)
    out["spread_vol_20"] = spread_diff.rolling(20).std()
    out["spread_vol_ratio"] = out["spread_vol_20"] / spread_diff.rolling(60).std()
    out["rolling_corr_60"] = ret_a.rolling(60).corr(ret_b)
    out["rolling_beta_120"] = ret_a.rolling(120).cov(ret_b) / ret_a.rolling(120).var()
    out["rsi_a"] = rsi(out["price_a"])
    out["rsi_b"] = rsi(out["price_b"])
    out["rel_volume_a"] = panel[f"{a}_volume"] / panel[f"{a}_volume"].rolling(20).mean()
    out["rel_volume_b"] = panel[f"{b}_volume"] / panel[f"{b}_volume"].rolling(20).mean()
    market_ret = np.log(panel[f"{market_ticker}_adj_close"]).diff()
    out["market_vol_20"] = market_ret.rolling(20).std() * np.sqrt(TRADING_DAYS)
    out["spread_momentum_5"] = out["spread"].diff(5) / out["spread_std"]

    out["quality_ok"] = panel["quality_ok"].astype(bool)
    out["pos"] = np.arange(len(out))
    return out.replace([np.inf, -np.inf], np.nan)


def build_labels(features: pd.DataFrame, beta: float, params: TradeParams) -> pd.DataFrame:
    """Label every session with the outcome of the hypothetical mean-reversion trade.

    At row t the hypothetical trade fades the current deviation (short the spread if
    z > 0, long if z < 0), is filled ``execution_lag`` sessions later, and follows the
    same exit policy and costs as the backtest. ``label`` = 1 if net P&L > 0, else 0.
    """
    price_a = features["price_a"].to_numpy(dtype=float)
    price_b = features["price_b"].to_numpy(dtype=float)
    z = features["z_score"].to_numpy(dtype=float)
    n = len(features)

    direction = np.zeros(n)
    net = np.full(n, np.nan)
    gross = np.full(n, np.nan)
    hold = np.full(n, np.nan)
    end_pos = np.full(n, np.nan)
    reason = np.full(n, None, dtype=object)

    for t in range(n):
        if np.isnan(z[t]) or z[t] == 0.0:
            continue
        d = -1 if z[t] > 0 else 1
        trade = simulate_trade(t, d, price_a, price_b, z, beta, params)
        if trade is None:  # not enough future data: leave unlabelled
            continue
        direction[t] = d
        net[t], gross[t] = trade["net_return"], trade["gross_return"]
        hold[t], reason[t] = trade["holding_days"], trade["exit_reason"]
        # Use the full possible span (not the realised exit) so purging is conservative.
        end_pos[t] = t + params.label_span

    labels = pd.DataFrame(
        {
            "trade_direction": direction,
            "trade_net_return": net,
            "trade_gross_return": gross,
            "trade_holding_days": hold,
            "trade_exit_reason": reason,
            "label_end_pos": end_pos,
        },
        index=features.index,
    )
    labels["label"] = np.where(np.isnan(net), np.nan, (net > 0).astype(float))
    return labels


def build_dataset(panel: pd.DataFrame, hedge: HedgeModel, cfg: dict) -> pd.DataFrame:
    """Features plus labels for the whole history, using one fixed hedge model."""
    params = TradeParams.from_config(cfg)
    features = build_features(panel, hedge, cfg["data"]["market_ticker"], cfg)
    return features.join(build_labels(features, hedge.beta, params))


def candidate_mask(dataset: pd.DataFrame, cfg: dict) -> pd.Series:
    """Rows that count as ML samples: a real deviation, inside the stop zone, clean data."""
    lo = float(cfg["model"]["train_min_abs_z"])
    hi = float(cfg["statistical"]["stop_z"])
    return (
        dataset["abs_z"].between(lo, hi, inclusive="left")
        & dataset["label"].notna()
        & dataset["quality_ok"]
    )
