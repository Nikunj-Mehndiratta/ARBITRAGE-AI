"""Signal fusion: combine the statistical signal, the model probability, costs and risk.

    baseline :  |z| >= entry_z                                    -> trade
    hybrid   :  |z| >= entry_z  AND  p >= threshold
                AND expected net return > 0  AND risk filters ok  -> trade

Signals are decisions about a *simulated* portfolio. Nothing here places an order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_SPREAD = "LONG_SPREAD"
SHORT_SPREAD = "SHORT_SPREAD"
NO_TRADE = "NO_TRADE"


def expected_net_return(probability, avg_win: float, avg_loss: float):
    """p * (average winning net return) + (1 - p) * (average losing net return).

    ``avg_win`` (> 0) and ``avg_loss`` (< 0) come from training-period hypothetical
    trades, already net of transaction costs.
    """
    p = np.asarray(probability, dtype=float)
    return p * avg_win + (1.0 - p) * avg_loss


def fuse_signal(
    z_score: float,
    probability: float,
    expected_return: float,
    risk_ok: bool,
    entry_z: float,
    stop_z: float,
    threshold: float,
    risk_reason: str = "",
) -> tuple[str, str]:
    """Decision for a single observation. Returns ``(signal, reason)``."""
    if z_score is None or np.isnan(z_score):
        return NO_TRADE, "z-score not available"
    if abs(z_score) < entry_z:
        return NO_TRADE, f"|z| below entry threshold {entry_z:g}"
    if abs(z_score) >= stop_z:
        return NO_TRADE, f"|z| beyond stop threshold {stop_z:g}"
    if not risk_ok:
        return NO_TRADE, f"risk filter: {risk_reason or 'limit breached'}"
    if probability is None or np.isnan(probability):
        return NO_TRADE, "model probability not available"
    if probability < threshold:
        return NO_TRADE, f"model probability {probability:.2f} below threshold {threshold:.2f}"
    if not expected_return > 0:
        return NO_TRADE, "expected net return not positive after costs"
    side = LONG_SPREAD if z_score < 0 else SHORT_SPREAD
    return side, f"|z| >= {entry_z:g}, probability {probability:.2f} >= {threshold:.2f}, risk filters passed"


def baseline_signals(z: pd.Series, entry_z: float, stop_z: float) -> pd.Series:
    """Rule-based baseline: fade the spread whenever entry_z <= |z| < stop_z."""
    active = z.abs().ge(entry_z) & z.abs().lt(stop_z)
    return pd.Series(np.where(active, -np.sign(z), 0.0), index=z.index, name="baseline_signal").fillna(0.0)


def hybrid_signals(
    dataset: pd.DataFrame,
    probability: pd.Series,
    risk: pd.DataFrame,
    threshold: float,
    avg_win: float,
    avg_loss: float,
    entry_z: float,
    stop_z: float,
) -> pd.DataFrame:
    """Hybrid decision for every row: ``signal`` (+1/-1/0), ``signal_name`` and ``reason``."""
    exp_ret = pd.Series(expected_net_return(probability, avg_win, avg_loss), index=dataset.index)
    names, reasons = [], []
    for z, p, er, ok, why in zip(
        dataset["z_score"].to_numpy(),
        probability.to_numpy(dtype=float),
        exp_ret.to_numpy(),
        risk["risk_ok"].to_numpy(),
        risk["risk_reason"].to_numpy(),
    ):
        name, reason = fuse_signal(z, p, er, bool(ok), entry_z, stop_z, threshold, why)
        names.append(name)
        reasons.append(reason)
    out = pd.DataFrame({"signal_name": names, "reason": reasons}, index=dataset.index)
    out["signal"] = out["signal_name"].map({LONG_SPREAD: 1.0, SHORT_SPREAD: -1.0, NO_TRADE: 0.0})
    out["expected_net_return"] = exp_ret
    return out
