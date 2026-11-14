"""Backtesting and simulation. No real orders exist anywhere in this module.

A *spread position* always has two legs:

    LONG_SPREAD  (direction +1, taken when z is very low):  long B, short beta * A
    SHORT_SPREAD (direction -1, taken when z is very high): short B, long beta * A

For 1 unit of gross notional, 1/(1+|beta|) goes into B and |beta|/(1+|beta|) into A.

Timing: a decision made on the close of day ``t`` is filled on the close of day
``t + execution_lag``. Exit policy, checked on each later close:

    1. the spread has returned near its mean  (|z| <= exit_z, or it crossed zero)
    2. the stop threshold is breached         (z moved further out, beyond stop_z)
    3. ``horizon_days`` sessions have passed since the entry fill

``simulate_trade`` is the single source of truth for one trade. It is used both to
label hypothetical trades for the ML model and to run the sequential backtest, so the
model is trained on exactly the outcome the backtest later realises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class TradeParams:
    entry_z: float
    exit_z: float
    stop_z: float
    horizon_days: int
    execution_lag: int
    cost_per_side: float  # (commission + slippage) as a fraction of traded notional, per leg, per side

    @classmethod
    def from_config(cls, cfg: dict) -> "TradeParams":
        st, tr = cfg["statistical"], cfg["trading"]
        return cls(
            entry_z=float(st["entry_z"]),
            exit_z=float(st["exit_z"]),
            stop_z=float(st["stop_z"]),
            horizon_days=int(tr["horizon_days"]),
            execution_lag=int(tr["execution_lag"]),
            cost_per_side=(float(tr["commission_bps"]) + float(tr["slippage_bps"])) / 1e4,
        )

    @property
    def label_span(self) -> int:
        """Sessions of future data a label needs after the decision day."""
        return self.execution_lag + self.horizon_days


def simulate_trade(
    t: int,
    direction: int,
    price_a: np.ndarray,
    price_b: np.ndarray,
    z: np.ndarray,
    beta: float,
    params: TradeParams,
    last_pos: int | None = None,
    allow_truncation: bool = False,
) -> dict | None:
    """Simulate one spread trade decided on the close of row ``t``.

    Returns None if there is not enough future data. With ``allow_truncation`` the trade
    is force-closed on ``last_pos`` (used at the end of a backtest period); without it
    the full horizon must be available (used for labels, so no label is ever partial).
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long spread) or -1 (short spread)")
    n = len(price_a)
    last_pos = n - 1 if last_pos is None else min(last_pos, n - 1)

    entry = t + params.execution_lag
    final = entry + params.horizon_days
    truncated = False
    if final > last_pos:
        if not allow_truncation or entry + 1 > last_pos:
            return None
        final, truncated = last_pos, True

    exit_pos, reason = final, ("end_of_data" if truncated else "time")
    for fill in range(entry + 1, final + 1):
        decision_day = fill - params.execution_lag
        zd = z[decision_day]
        if np.isnan(zd):
            continue
        adverse = -direction * zd  # positive while the spread is still stretched against the mean
        if adverse <= params.exit_z:
            exit_pos, reason = fill, "reverted"
            break
        if adverse >= params.stop_z:
            exit_pos, reason = fill, "stop"
            break

    ret_a = price_a[exit_pos] / price_a[entry] - 1.0
    ret_b = price_b[exit_pos] / price_b[entry] - 1.0
    gross_notional = 1.0 + abs(beta)
    gross_return = direction * (ret_b - beta * ret_a) / gross_notional
    net_return = gross_return - 2.0 * params.cost_per_side  # entry + exit, both legs

    return {
        "decision_pos": t,
        "entry_pos": entry,
        "exit_pos": exit_pos,
        "direction": direction,
        "exit_reason": reason,
        "holding_days": exit_pos - entry,
        "gross_return": float(gross_return),
        "net_return": float(net_return),
    }


def run_backtest(
    dataset: pd.DataFrame,
    signal: pd.Series | np.ndarray,
    beta: float,
    params: TradeParams,
    start_pos: int,
    end_pos: int,
    initial_capital: float = 100_000.0,
    gross_exposure: float = 1.0,
    strategy: str = "strategy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential paper-trading simulation over rows ``[start_pos, end_pos)``.

    ``signal`` holds the decision for every row: +1 long spread, -1 short spread, 0 none.
    One position at a time. Returns ``(trades, equity)``; equity is marked to market daily.
    """
    price_a = dataset["price_a"].to_numpy(dtype=float)
    price_b = dataset["price_b"].to_numpy(dtype=float)
    z = dataset["z_score"].to_numpy(dtype=float)
    sig = np.asarray(signal, dtype=float)
    dates = dataset.index
    last_pos = end_pos - 1

    daily_pnl = np.zeros(len(dataset))
    in_position = np.zeros(len(dataset), dtype=bool)
    equity_now = float(initial_capital)
    trades: list[dict] = []

    i = start_pos
    while i < end_pos:
        direction = int(sig[i]) if not np.isnan(sig[i]) else 0
        if direction == 0:
            i += 1
            continue
        trade = simulate_trade(i, direction, price_a, price_b, z, beta, params, last_pos, allow_truncation=True)
        if trade is None:
            i += 1
            continue

        e, x = trade["entry_pos"], trade["exit_pos"]
        gross = equity_now * gross_exposure
        dollars_b = gross / (1.0 + abs(beta))
        dollars_a = gross - dollars_b
        shares_b = direction * dollars_b / price_b[e]
        shares_a = -direction * np.sign(beta) * dollars_a / price_a[e]

        entry_cost = params.cost_per_side * gross
        exit_cost = params.cost_per_side * (abs(shares_b) * price_b[x] + abs(shares_a) * price_a[x])
        daily_pnl[e] -= entry_cost
        for j in range(e + 1, x + 1):
            daily_pnl[j] += shares_b * (price_b[j] - price_b[j - 1]) + shares_a * (price_a[j] - price_a[j - 1])
            in_position[j] = True
        daily_pnl[x] -= exit_cost

        gross_pnl = shares_b * (price_b[x] - price_b[e]) + shares_a * (price_a[x] - price_a[e])
        net_pnl = gross_pnl - entry_cost - exit_cost
        equity_now += net_pnl

        trades.append(
            {
                "strategy": strategy,
                "decision_date": dates[i],
                "entry_date": dates[e],
                "exit_date": dates[x],
                "side": "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD",
                "entry_z": float(z[i]),
                "exit_z": float(z[x - params.execution_lag]) if x - params.execution_lag >= 0 else np.nan,
                "exit_reason": trade["exit_reason"],
                "holding_days": trade["holding_days"],
                "gross_notional": gross,
                "gross_pnl": gross_pnl,
                "costs": entry_cost + exit_cost,
                "net_pnl": net_pnl,
                "net_return": net_pnl / gross,
                "decision_pos": i,
            }
        )
        i = x  # the next decision can be taken on the close the position was exited

    period = slice(start_pos, end_pos)
    equity = pd.DataFrame(
        {
            "strategy": strategy,
            "daily_pnl": daily_pnl[period],
            "in_position": in_position[period],
        },
        index=dates[period],
    )
    equity["equity"] = initial_capital + equity["daily_pnl"].cumsum()
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    return pd.DataFrame(trades), equity


def performance_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict:
    """Trading-simulation metrics. Reported separately from classification metrics."""
    n_trades = int(len(trades))
    final_equity = float(equity["equity"].iloc[-1]) if len(equity) else float(initial_capital)
    prev_equity = equity["equity"].shift(1).fillna(initial_capital)
    daily_ret = equity["daily_pnl"] / prev_equity
    vol = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else 0.0
    sharpe = float(daily_ret.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0

    out = {
        "n_trades": n_trades,
        "net_pnl": final_equity - float(initial_capital),
        "total_return": final_equity / float(initial_capital) - 1.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(equity["drawdown"].min()) if len(equity) else 0.0,
        "time_in_market": float(equity["in_position"].mean()) if len(equity) else 0.0,
        "sessions": int(len(equity)),
    }
    if n_trades:
        wins = trades[trades.net_pnl > 0]
        losses = trades[trades.net_pnl <= 0]
        gross_win, gross_loss = float(wins.net_pnl.sum()), float(-losses.net_pnl.sum())
        out.update(
            {
                "win_rate": float(len(wins) / n_trades),
                "avg_trade_return": float(trades.net_return.mean()),
                "best_trade_return": float(trades.net_return.max()),
                "worst_trade_return": float(trades.net_return.min()),
                "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
                "total_costs": float(trades.costs.sum()),
                "avg_holding_days": float(trades.holding_days.mean()),
                "exit_reasons": {k: int(v) for k, v in trades.exit_reason.value_counts().items()},
            }
        )
    else:
        out.update(
            {
                "win_rate": None,
                "avg_trade_return": None,
                "best_trade_return": None,
                "worst_trade_return": None,
                "profit_factor": None,
                "total_costs": 0.0,
                "avg_holding_days": None,
                "exit_reasons": {},
            }
        )
    return out
