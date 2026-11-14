"""Trade simulation: both legs, costs, exits, timing and the sequential backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arbitrage.backtester import TradeParams, performance_metrics, run_backtest, simulate_trade
from arbitrage.feature_engineering import build_labels

COST = 0.0003  # 3 bps per side


def params(lag: int = 1, horizon: int = 5) -> TradeParams:
    return TradeParams(entry_z=2.0, exit_z=0.5, stop_z=3.5, horizon_days=horizon, execution_lag=lag, cost_per_side=COST)


def flat(n: int = 30, a: float = 100.0, b: float = 50.0):
    return np.full(n, a), np.full(n, b)


def test_flat_prices_lose_exactly_the_round_trip_cost():
    a, b = flat()
    z = np.full(30, 2.5)
    trade = simulate_trade(3, -1, a, b, z, beta=1.2, params=params())
    assert trade["gross_return"] == 0.0
    assert np.isclose(trade["net_return"], -2 * COST)
    assert trade["exit_reason"] == "time" and trade["holding_days"] == 5


def test_entry_is_filled_after_the_execution_lag():
    a, b = flat()
    z = np.full(30, 2.5)
    assert simulate_trade(3, -1, a, b, z, 1.0, params(lag=1))["entry_pos"] == 4
    assert simulate_trade(3, -1, a, b, z, 1.0, params(lag=0))["entry_pos"] == 3


def test_both_legs_drive_the_pnl():
    a, b = flat()
    b = b.copy()
    b[5:] = 55.0  # B rises 10% after the entry fill at row 4; A is unchanged
    z = np.full(30, -2.5)
    beta = 1.0
    long_spread = simulate_trade(3, 1, a, b, z, beta, params())
    assert np.isclose(long_spread["gross_return"], 0.10 / (1 + beta))  # long B on half the gross notional
    short_spread = simulate_trade(3, -1, a, b, np.full(30, 2.5), beta, params())
    assert np.isclose(short_spread["gross_return"], -0.10 / (1 + beta))

    a2 = a.copy()
    a2[5:] = 110.0  # now A rises 10% as well: a beta-1 hedge cancels it out
    assert np.isclose(simulate_trade(3, 1, a2, b, z, beta, params())["gross_return"], 0.0)


def test_exit_when_spread_reverts():
    a, b = flat()
    z = np.full(30, 2.5)
    z[6:] = 0.3  # back near the mean on row 6
    trade = simulate_trade(3, -1, a, b, z, 1.0, params(lag=1))
    assert trade["exit_reason"] == "reverted"
    assert trade["exit_pos"] == 7  # seen at the close of row 6, filled one session later


def test_exit_on_stop_when_spread_moves_further_away():
    a, b = flat()
    z = np.full(30, 2.5)
    z[5:] = 3.9
    trade = simulate_trade(3, -1, a, b, z, 1.0, params(lag=1))
    assert trade["exit_reason"] == "stop" and trade["exit_pos"] == 6


def test_stop_is_direction_aware():
    a, b = flat()
    z = np.full(30, -2.5)
    z[5:] = -3.9  # further below the mean: adverse for a long-spread position
    assert simulate_trade(3, 1, a, b, z, 1.0, params())["exit_reason"] == "stop"


def test_no_partial_labels_at_the_end_of_data():
    a, b = flat(n=10)
    z = np.full(10, 2.5)
    assert simulate_trade(6, -1, a, b, z, 1.0, params()) is None  # needs rows up to 12
    truncated = simulate_trade(6, -1, a, b, z, 1.0, params(), last_pos=9, allow_truncation=True)
    assert truncated["exit_reason"] == "end_of_data" and truncated["exit_pos"] == 9


def test_invalid_direction_is_rejected():
    a, b = flat()
    with pytest.raises(ValueError):
        simulate_trade(3, 0, a, b, np.full(30, 2.5), 1.0, params())


def _dataset(a, b, z) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(a))
    return pd.DataFrame({"price_a": a, "price_b": b, "z_score": z}, index=idx)


def test_labels_agree_with_the_trade_simulator():
    rng = np.random.default_rng(1)
    n = 120
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    z = rng.normal(0, 1.5, n)
    p = params()
    labels = build_labels(_dataset(a, b, z), beta=1.1, params=p)
    for t in (10, 40, 77):
        d = -1 if z[t] > 0 else 1
        trade = simulate_trade(t, d, a, b, z, 1.1, p)
        assert np.isclose(labels["trade_net_return"].iloc[t], trade["net_return"])
        assert labels["label"].iloc[t] == float(trade["net_return"] > 0)
        assert labels["label_end_pos"].iloc[t] == t + p.label_span
    assert labels["label"].iloc[-p.label_span :].isna().all()


def test_backtest_without_signals_is_flat():
    a, b = flat(n=50)
    ds = _dataset(a, b, np.zeros(50))
    trades, equity = run_backtest(ds, np.zeros(50), 1.0, params(), 0, 50, 100_000)
    assert trades.empty and (equity["equity"] == 100_000).all()
    assert performance_metrics(trades, equity, 100_000)["n_trades"] == 0


def test_backtest_books_costs_and_holds_one_position_at_a_time():
    a, b = flat(n=60)
    z = np.full(60, 2.5)  # a signal every single day
    ds = _dataset(a, b, z)
    signal = np.full(60, -1.0)
    trades, equity = run_backtest(ds, signal, 1.0, params(), 0, 60, 100_000)

    # Flat prices: every trade loses exactly its costs, and equity falls by their sum.
    assert np.allclose(trades["gross_pnl"], 0.0)
    assert np.isclose(equity["equity"].iloc[-1], 100_000 - trades["costs"].sum())
    assert np.isclose(trades["net_pnl"].sum(), equity["daily_pnl"].sum())
    # Positions never overlap: each entry is on or after the previous exit.
    assert (trades["entry_date"].iloc[1:].to_numpy() >= trades["exit_date"].iloc[:-1].to_numpy()).all()
    # 60 sessions cannot hold more than ~10 five-session trades.
    assert 8 <= len(trades) <= 11


def test_backtest_pnl_matches_hand_calculation():
    a, b = flat(n=40)
    b = b.copy()
    b[11:] = 52.5  # +5% in B one session after the fill
    z = np.zeros(40)
    z[9] = -2.5  # a single long-spread decision on row 9, filled on row 10
    signal = np.zeros(40)
    signal[9] = 1.0
    ds = _dataset(a, b, z)
    trades, equity = run_backtest(ds, signal, 1.0, params(), 0, 40, 100_000, gross_exposure=1.0)

    assert len(trades) == 1
    gross = 50_000 * 0.05  # half the gross notional is long B, which rose 5%
    costs = COST * 100_000 + COST * (52_500 + 50_000)
    assert np.isclose(trades["gross_pnl"].iloc[0], gross)
    assert np.isclose(trades["net_pnl"].iloc[0], gross - costs)
    assert np.isclose(equity["equity"].iloc[-1], 100_000 + gross - costs)
    assert equity["drawdown"].min() <= 0
