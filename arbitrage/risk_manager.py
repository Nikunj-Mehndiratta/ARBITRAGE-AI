"""Risk filters applied before any hybrid signal is allowed through.

The limits are quantiles of the TRAINING period, so "unusually volatile" is always
defined without reference to validation or test data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_market_vol: float
    max_spread_vol: float
    min_rolling_corr: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_risk_limits(train_features: pd.DataFrame, cfg: dict) -> RiskLimits:
    """Derive volatility caps from the training rows only."""
    rk = cfg["risk"]
    return RiskLimits(
        max_market_vol=float(train_features["market_vol_20"].quantile(rk["max_market_vol_quantile"])),
        max_spread_vol=float(train_features["spread_vol_20"].quantile(rk["max_spread_vol_quantile"])),
        min_rolling_corr=float(rk["min_rolling_correlation"]),
    )


def risk_check(features: pd.DataFrame, limits: RiskLimits) -> pd.DataFrame:
    """Row-by-row risk verdict. ``risk_reason`` is empty when every check passes."""
    checks = pd.DataFrame(index=features.index)
    checks["data_ok"] = features["quality_ok"].astype(bool) & features[
        ["z_score", "market_vol_20", "spread_vol_20", "rolling_corr_60"]
    ].notna().all(axis=1)
    checks["market_vol_ok"] = features["market_vol_20"] <= limits.max_market_vol
    checks["spread_vol_ok"] = features["spread_vol_20"] <= limits.max_spread_vol
    checks["correlation_ok"] = features["rolling_corr_60"] >= limits.min_rolling_corr

    labels = {
        "data_ok": "data quality",
        "market_vol_ok": "market volatility too high",
        "spread_vol_ok": "spread volatility too high",
        "correlation_ok": "pair correlation broke down",
    }
    reason = np.full(len(checks), "", dtype=object)
    for col in ["correlation_ok", "spread_vol_ok", "market_vol_ok", "data_ok"]:  # last wins
        reason = np.where(~checks[col].to_numpy(), labels[col], reason)
    checks["risk_ok"] = checks[list(labels)].all(axis=1)
    checks["risk_reason"] = reason
    return checks
