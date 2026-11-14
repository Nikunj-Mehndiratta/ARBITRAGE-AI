"""Mathematical engine: hedge ratio, spread, rolling z-score and mean-reversion diagnostics.

    P_A(t) = ln(A_t),  P_B(t) = ln(B_t)
    P_B(t) = alpha + beta * P_A(t) + eps_t        (fitted on the fit window only)
    S_t    = P_B(t) - alpha - beta * P_A(t)       (the spread)
    Z_t    = (S_t - mu_t) / sigma_t               (rolling mean / std, past data only)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


@dataclass(frozen=True)
class HedgeModel:
    """OLS relationship between the two log-price series."""

    ticker_a: str
    ticker_b: str
    alpha: float
    beta: float
    r_squared: float
    n_obs: int
    fit_start: str
    fit_end: str

    def to_dict(self) -> dict:
        return asdict(self)


def fit_hedge_ratio(
    log_a: pd.Series, log_b: pd.Series, ticker_a: str = "A", ticker_b: str = "B"
) -> HedgeModel:
    """Estimate ``P_B = alpha + beta * P_A`` by OLS on the supplied (fit-window) data."""
    if len(log_a) != len(log_b) or len(log_a) < 30:
        raise ValueError("Need at least 30 aligned observations to fit a hedge ratio")
    x = add_constant(np.asarray(log_a, dtype=float))
    res = OLS(np.asarray(log_b, dtype=float), x).fit()
    alpha, beta = (float(v) for v in res.params)
    return HedgeModel(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        alpha=alpha,
        beta=beta,
        r_squared=float(res.rsquared),
        n_obs=int(res.nobs),
        fit_start=str(pd.Timestamp(log_a.index[0]).date()),
        fit_end=str(pd.Timestamp(log_a.index[-1]).date()),
    )


def compute_spread(log_a: pd.Series, log_b: pd.Series, hedge: HedgeModel) -> pd.Series:
    """Residual of the hedge regression, applied to any period (in or out of sample)."""
    return (log_b - hedge.alpha - hedge.beta * log_a).rename("spread")


def rolling_zscore(spread: pd.Series, window: int) -> pd.DataFrame:
    """Rolling z-score of the spread.

    The window ends at (and includes) the current observation, so Z_t depends only on
    S_{t-window+1} ... S_t. Changing any future value cannot change Z_t.
    """
    mu = spread.rolling(window, min_periods=window).mean()
    sigma = spread.rolling(window, min_periods=window).std(ddof=1)
    z = (spread - mu) / sigma.replace(0.0, np.nan)
    return pd.DataFrame({"spread_mean": mu, "spread_std": sigma, "z_score": z})


def half_life(spread: pd.Series) -> float:
    """Half-life (in sessions) of mean reversion from an AR(1) fit; inf if not mean-reverting."""
    s = spread.dropna()
    if len(s) < 30:
        return float("nan")
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    x = add_constant(lagged.values)
    phi = float(OLS(delta.values, x).fit().params[1])
    if phi >= 0:
        return float("inf")
    return float(-np.log(2) / np.log1p(phi)) if phi > -1 else 1.0


def classify_z(z: float, entry_z: float, exit_z: float, stop_z: float) -> str:
    """Plain-language reading of a z-score under the experimental thresholds."""
    if z is None or np.isnan(z):
        return "undefined"
    a = abs(z)
    if a >= stop_z:
        return "risk / stop zone"
    if a >= entry_z:
        return "spread unusually high" if z > 0 else "spread unusually low"
    if a < exit_z:
        return "near its mean"
    return "inside normal range"
