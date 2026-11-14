"""Pair selection on TRAINING data only.

A pair is screened with return correlation, liquidity, an Engle-Granger cointegration
test, and a stability check that repeats the test on sub-windows of the training period.
Nothing here may see validation or test data: callers pass in the training slice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .statistical_engine import fit_hedge_ratio


def _coint_pvalue(log_b: pd.Series, log_a: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(coint(log_b.values, log_a.values, trend="c")[1])
        except Exception:
            return float("nan")


def evaluate_pair(train_panel: pd.DataFrame, a: str, b: str, cfg: dict) -> dict:
    """Compute every screening statistic for one candidate pair on the training slice."""
    ps = cfg["pair_selection"]
    log_a = np.log(train_panel[f"{a}_adj_close"])
    log_b = np.log(train_panel[f"{b}_adj_close"])

    ret_corr = float(log_a.diff().corr(log_b.diff()))
    dollar_vol_a = float((train_panel[f"{a}_close"] * train_panel[f"{a}_volume"]).mean())
    dollar_vol_b = float((train_panel[f"{b}_close"] * train_panel[f"{b}_volume"]).mean())
    min_dollar_vol = min(dollar_vol_a, dollar_vol_b)

    p_full = _coint_pvalue(log_b, log_a)
    hedge = fit_hedge_ratio(log_a, log_b, a, b)

    # Stability: repeat the test and the hedge fit on consecutive sub-windows.
    k = int(ps["stability_windows"])
    window_p, window_beta = [], []
    for idx in np.array_split(np.arange(len(train_panel)), k):
        wa, wb = log_a.iloc[idx], log_b.iloc[idx]
        window_p.append(_coint_pvalue(wb, wa))
        window_beta.append(fit_hedge_ratio(wa, wb, a, b).beta)
    window_p_arr = np.array(window_p, dtype=float)
    stability_share = float(np.nanmean(window_p_arr < ps["stability_pvalue"]))

    passes_corr = ret_corr >= ps["min_return_correlation"]
    passes_liquidity = min_dollar_vol >= ps["min_avg_dollar_volume"]
    passes_coint = bool(p_full <= ps["max_coint_pvalue"])

    return {
        "ticker_a": a,
        "ticker_b": b,
        "return_correlation": ret_corr,
        "min_avg_dollar_volume": min_dollar_vol,
        "coint_pvalue": p_full,
        "hedge_ratio_beta": hedge.beta,
        "hedge_r_squared": hedge.r_squared,
        "window_pvalues": [round(float(p), 4) for p in window_p],
        "window_betas": [round(float(v), 4) for v in window_beta],
        "stability_share": stability_share,
        "passes_correlation": bool(passes_corr),
        "passes_liquidity": bool(passes_liquidity),
        "passes_cointegration": passes_coint,
        "passes_all": bool(passes_corr and passes_liquidity and passes_coint),
    }


def select_pair(train_panel: pd.DataFrame, cfg: dict) -> tuple[tuple[str, str], pd.DataFrame, dict]:
    """Pick the pair to study. Returns ``(pair, report_table, decision)``.

    If no candidate passes every screen, the pair with the lowest cointegration p-value
    is still used so the experiment can run, and the decision carries an explicit warning.
    """
    force = cfg["pair_selection"].get("force_pair")
    candidates = [tuple(p) for p in cfg["data"]["candidate_pairs"]]
    if force and tuple(force) not in candidates:
        candidates.append(tuple(force))

    rows = [evaluate_pair(train_panel, a, b, cfg) for a, b in candidates]
    report = pd.DataFrame(rows)

    if force:
        pair = tuple(force)
        row = report[(report.ticker_a == pair[0]) & (report.ticker_b == pair[1])].iloc[0]
        decision = {"method": "forced in config.yaml", "warning": None}
    else:
        passing = report[report.passes_all]
        if len(passing):
            row = passing.sort_values("coint_pvalue").iloc[0]
            decision = {"method": "lowest Engle-Granger p-value among pairs passing all screens", "warning": None}
        else:
            row = report.sort_values("coint_pvalue").iloc[0]
            decision = {
                "method": "fallback: lowest Engle-Granger p-value (no pair passed all screens)",
                "warning": None,
            }
        pair = (row.ticker_a, row.ticker_b)

    if not bool(row.passes_all):
        failed = [
            name
            for name, ok in [
                ("return correlation", row.passes_correlation),
                ("liquidity", row.passes_liquidity),
                (f"cointegration at p <= {cfg['pair_selection']['max_coint_pvalue']}", row.passes_cointegration),
            ]
            if not ok
        ]
        decision["warning"] = (
            f"{pair[0]}/{pair[1]} did not pass: {', '.join(failed)} "
            f"(Engle-Granger p = {row.coint_pvalue:.3f} on training data). "
            "The spread is not statistically confirmed to be mean-reverting; treat every result as exploratory."
        )
    decision["pair"] = list(pair)
    decision["selected_on"] = "training data only"
    return pair, report, decision
