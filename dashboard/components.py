"""Reusable dashboard pieces: styling, header, stat tiles, signal panel, formatters."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

CSS = """
<style>
  [data-testid="stMainBlockContainer"] { padding-top: 4.5rem !important; max-width: 1280px; }
  .aa-header { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
               border-bottom:1px solid #2c2c2a; padding-bottom:14px; margin-bottom:18px; }
  .aa-title { font-size:1.55rem; font-weight:700; letter-spacing:.08em; color:#f0efec; line-height:1.1; }
  .aa-sub { font-size:.72rem; letter-spacing:.22em; color:#898781; margin-top:4px; }
  .aa-badge { font-size:.7rem; letter-spacing:.18em; font-weight:600; color:#f0efec; border:1px solid #898781;
              border-radius:999px; padding:5px 12px; white-space:nowrap; }
  .aa-tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:12px; }
  .aa-tile { background:#1a1a19; border:1px solid rgba(255,255,255,.10); border-radius:10px; padding:14px 16px; }
  .aa-label { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:#898781; }
  .aa-value { font-size:1.7rem; font-weight:650; color:#f0efec; margin-top:4px; line-height:1.15; }
  .aa-note { font-size:.78rem; color:#c3c2b7; margin-top:4px; }
  .aa-signal { background:#1a1a19; border:1px solid rgba(255,255,255,.10); border-left:4px solid #898781;
               border-radius:10px; padding:14px 18px; margin-bottom:18px; }
  .aa-signal.active { border-left-color:#3987e5; }
  .aa-signal .aa-value { font-size:1.35rem; letter-spacing:.06em; }
  div[data-testid="stMetricValue"] { font-size:1.35rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def money(x, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{'+' if signed and x > 0 else ''}${x:,.0f}".replace("$-", "-$")


def pct(x, digits: int = 1, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{'+' if signed and x > 0 else ''}{x * 100:.{digits}f}%"


def num(x, digits: int = 2, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def header() -> None:
    st.markdown(
        """
        <div class="aa-header">
          <div>
            <div class="aa-title">ARBITRAGE AI</div>
            <div class="aa-sub">QUANTITATIVE RESEARCH TERMINAL</div>
          </div>
          <div class="aa-badge">SIMULATION · PAPER TRADING ONLY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tiles(items: list[tuple[str, str, str]]) -> None:
    """Row of stat tiles. Each item is (label, value, note)."""
    cells = "".join(
        f'<div class="aa-tile"><div class="aa-label">{html.escape(label)}</div>'
        f'<div class="aa-value">{html.escape(value)}</div>'
        f'<div class="aa-note">{html.escape(note)}</div></div>'
        for label, value, note in items
    )
    st.markdown(f'<div class="aa-tiles">{cells}</div>', unsafe_allow_html=True)


SIGNAL_TEXT = {
    "LONG_SPREAD": "LONG SPREAD",
    "SHORT_SPREAD": "SHORT SPREAD",
    "NO_TRADE": "NO SIGNAL",
}


def signal_panel(signal: str, reason: str, date: str, legs: str = "") -> None:
    active = signal in ("LONG_SPREAD", "SHORT_SPREAD")
    detail = f"{html.escape(reason)}{' · ' + html.escape(legs) if legs else ''}"
    st.markdown(
        f"""
        <div class="aa-signal {'active' if active else ''}">
          <div class="aa-label">Current signal · as of {html.escape(date)} close</div>
          <div class="aa-value">{SIGNAL_TEXT.get(signal, signal)}</div>
          <div class="aa-note">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def strategy_table(baseline: dict, hybrid: dict) -> pd.DataFrame:
    """Side-by-side trading metrics for the two systems."""

    def ci(m: dict) -> str:
        c = m.get("avg_trade_return_ci95")
        return f"{c[0] * 100:+.2f}% to {c[1] * 100:+.2f}%" if c else "—"

    rows = [
        ("Net cumulative P&L", money(baseline["net_pnl"]), money(hybrid["net_pnl"])),
        ("Total return", pct(baseline["total_return"], 2, True), pct(hybrid["total_return"], 2, True)),
        ("Trades", str(baseline["n_trades"]), str(hybrid["n_trades"])),
        ("Win rate", pct(baseline["win_rate"], 0), pct(hybrid["win_rate"], 0)),
        ("Average trade return (net)", pct(baseline["avg_trade_return"], 2, True), pct(hybrid["avg_trade_return"], 2, True)),
        ("95% bootstrap interval, avg trade", ci(baseline), ci(hybrid)),
        ("Sharpe ratio (annualised)", num(baseline["sharpe_ratio"]), num(hybrid["sharpe_ratio"])),
        ("Maximum drawdown", pct(baseline["max_drawdown"], 1), pct(hybrid["max_drawdown"], 1)),
        ("Profit factor", num(baseline["profit_factor"]), num(hybrid["profit_factor"])),
        ("Time in market", pct(baseline["time_in_market"], 0), pct(hybrid["time_in_market"], 0)),
        ("Transaction costs paid", money(baseline["total_costs"], False), money(hybrid["total_costs"], False)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Baseline (rules only)", "Hybrid (rules + ML + risk)"])


def classification_table(metrics: dict) -> pd.DataFrame:
    """Classifier metrics next to the no-skill reference, so they cannot be over-read."""
    rows = [
        ("Samples", f"{metrics['n_samples']}", "—"),
        ("Share of profitable trades (base rate)", pct(metrics["base_rate"], 1), "—"),
        (f"Precision at threshold {metrics['threshold']:.2f}", pct(metrics["precision"], 1), pct(metrics["base_rate"], 1)),
        (f"Recall at threshold {metrics['threshold']:.2f}", pct(metrics["recall"], 1), "—"),
        ("PR-AUC", num(metrics["pr_auc"], 3), num(metrics["pr_auc_no_skill"], 3)),
        ("ROC-AUC", num(metrics["roc_auc"], 3), "0.500"),
        ("Brier score (lower is better)", num(metrics["brier_score"], 4), num(metrics["brier_no_skill"], 4)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Model", "No-skill reference"])
