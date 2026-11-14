"""Statistical engine, look-ahead checks, risk filters and signal fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arbitrage.feature_engineering import FEATURE_COLUMNS, build_features
from arbitrage.pair_selection import select_pair
from arbitrage.risk_manager import fit_risk_limits, risk_check
from arbitrage.signal_fusion import (
    LONG_SPREAD,
    NO_TRADE,
    SHORT_SPREAD,
    baseline_signals,
    expected_net_return,
    fuse_signal,
)
from arbitrage.statistical_engine import compute_spread, fit_hedge_ratio, half_life, rolling_zscore
from tests.conftest import TRUE_BETA

FUSE = dict(entry_z=2.0, stop_z=3.5, threshold=0.65)


def _hedge(panel):
    return fit_hedge_ratio(np.log(panel["AAA_adj_close"]), np.log(panel["BBB_adj_close"]), "AAA", "BBB")


def test_hedge_ratio_recovers_the_true_relationship(panel):
    hedge = _hedge(panel)
    assert abs(hedge.beta - TRUE_BETA) < 0.1
    spread = compute_spread(np.log(panel["AAA_adj_close"]), np.log(panel["BBB_adj_close"]), hedge)
    assert abs(spread.mean()) < 1e-8  # OLS residuals are centred on the fit window
    assert 1 < half_life(spread) < 30  # the synthetic spread is AR(1) with phi = 0.9


def test_zscore_matches_its_definition():
    s = pd.Series(np.arange(10, dtype=float))
    out = rolling_zscore(s, window=5)
    assert out["z_score"].iloc[:4].isna().all()
    window = s.iloc[5:10]
    assert np.isclose(out["z_score"].iloc[9], (s.iloc[9] - window.mean()) / window.std(ddof=1))


def test_zscore_has_no_look_ahead():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=300))
    before = rolling_zscore(s, 60)["z_score"]
    tampered = s.copy()
    tampered.iloc[200:] += 5.0  # rewrite the future
    after = rolling_zscore(tampered, 60)["z_score"]
    pd.testing.assert_series_equal(before.iloc[:200], after.iloc[:200])


def test_features_have_no_look_ahead(panel, cfg):
    """Changing prices and volumes after day t must not change any feature up to day t."""
    hedge = _hedge(panel.iloc[:400])
    before = build_features(panel, hedge, "MKT", cfg)
    tampered = panel.copy()
    cut = 450
    price_cols = [c for c in panel.columns if c != "quality_ok"]
    tampered.iloc[cut:, [panel.columns.get_loc(c) for c in price_cols]] *= 1.37
    after = build_features(tampered, hedge, "MKT", cfg)
    pd.testing.assert_frame_equal(before[FEATURE_COLUMNS].iloc[:cut], after[FEATURE_COLUMNS].iloc[:cut])


def test_pair_selection_accepts_a_cointegrated_pair(panel, cfg):
    cfg["pair_selection"]["min_avg_dollar_volume"] = 0
    pair, report, decision = select_pair(panel.iloc[:400], cfg)
    assert pair == ("AAA", "BBB")
    assert bool(report.iloc[0]["passes_cointegration"]) and decision["warning"] is None


def test_pair_selection_warns_when_screens_fail(panel, cfg):
    cfg["pair_selection"]["min_avg_dollar_volume"] = 1e15  # nothing is this liquid
    _, _, decision = select_pair(panel.iloc[:400], cfg)
    assert decision["warning"] and "liquidity" in decision["warning"]


def test_baseline_signal_fades_the_spread():
    z = pd.Series([0.0, 1.9, 2.0, 2.6, -2.1, -3.4, 3.5, -4.0, np.nan])
    sig = baseline_signals(z, entry_z=2.0, stop_z=3.5)
    assert sig.tolist() == [0, 0, -1, -1, 1, 1, 0, 0, 0]  # short when high, long when low, none in stop zone


def test_fusion_requires_every_condition():
    ok = dict(probability=0.70, expected_return=0.004, risk_ok=True, **FUSE)
    assert fuse_signal(z_score=-2.4, **ok)[0] == LONG_SPREAD
    assert fuse_signal(z_score=2.4, **ok)[0] == SHORT_SPREAD
    assert fuse_signal(z_score=1.5, **ok)[0] == NO_TRADE                           # z not extreme
    assert fuse_signal(z_score=3.8, **ok)[0] == NO_TRADE                           # stop zone
    assert fuse_signal(z_score=2.4, **{**ok, "probability": 0.64})[0] == NO_TRADE  # model not convinced
    assert fuse_signal(z_score=2.4, **{**ok, "expected_return": -0.001})[0] == NO_TRADE
    assert fuse_signal(z_score=2.4, **{**ok, "risk_ok": False})[0] == NO_TRADE
    assert fuse_signal(z_score=float("nan"), **ok)[0] == NO_TRADE
    assert fuse_signal(z_score=2.4, **{**ok, "probability": float("nan")})[0] == NO_TRADE


def test_fusion_explains_itself():
    signal, reason = fuse_signal(2.4, 0.5, 0.01, True, **FUSE)
    assert signal == NO_TRADE and "below threshold" in reason


def test_expected_net_return_is_probability_weighted():
    assert np.isclose(expected_net_return(0.5, 0.01, -0.01), 0.0)
    assert expected_net_return(0.7, 0.01, -0.01) > 0 > expected_net_return(0.3, 0.01, -0.01)


def test_risk_limits_come_from_the_fitting_rows_only(panel, cfg):
    feats = build_features(panel, _hedge(panel.iloc[:400]), "MKT", cfg)
    limits = fit_risk_limits(feats.iloc[:400], cfg)
    tampered = feats.copy()
    tampered.iloc[400:, tampered.columns.get_loc("market_vol_20")] = 99.0  # an extreme future
    assert fit_risk_limits(tampered.iloc[:400], cfg) == limits
    checks = risk_check(tampered, limits)
    assert not checks["risk_ok"].iloc[400:].any()
    assert (checks["risk_reason"].iloc[400:] == "market volatility too high").all()
