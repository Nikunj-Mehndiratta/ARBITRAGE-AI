"""Plotly charts for the dashboard.

Colour is assigned by role and stays fixed: stock A is always blue, stock B always
orange; the hybrid strategy is always aqua, the baseline always violet. No chart uses
two y-scales: series of different scale are indexed to a common base or stacked in
separate panels that share the time axis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dark-surface palette (each pair checked for colour-vision-deficiency separation).
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
BLUE = "#3987e5"     # stock A / primary series
ORANGE = "#d95926"   # stock B / secondary series
AQUA = "#199e70"     # hybrid strategy
VIOLET = "#9085e9"   # baseline strategy
CRITICAL = "#d03b3b"  # stop level only
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(fig: go.Figure, height: int = 380, unified: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=12),
        margin=dict(l=56, r=24, t=44, b=40),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor=PAGE, bordercolor=AXIS, font=dict(color=INK, family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def _level(fig: go.Figure, y: float, label: str, color: str = MUTED, dash: str = "dash", row=None, col=None) -> None:
    kwargs = dict(row=row, col=col) if row else {}
    fig.add_hline(
        y=y, line=dict(color=color, width=1, dash=dash),
        annotation_text=label, annotation_position="right",
        annotation_font=dict(color=color, size=10), **kwargs,
    )


def _split_markers(fig: go.Figure, ds: pd.DataFrame, rows: int = 1) -> None:
    """Thin vertical rules where the validation and test periods begin."""
    for name in ("validation", "test"):
        part = ds[ds["split"] == name]
        if part.empty or part.index[0] == ds.index[0]:
            continue
        x = part.index[0]
        for r in range(1, rows + 1):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=0, y1=1, yref="y domain" if r == 1 else f"y{r} domain",
                xref="x" if r == 1 else f"x{r}", line=dict(color=AXIS, width=1, dash="dot"),
            )
        fig.add_annotation(
            x=x, y=1, yref="y domain", text=f"{name} starts", showarrow=False,
            xanchor="left", yanchor="top", font=dict(color=MUTED, size=10), xshift=4,
        )


def spread_zscore_chart(ds: pd.DataFrame, params: dict, trades: pd.DataFrame | None = None) -> go.Figure:
    """Spread with its rolling mean (top) and the rolling z-score with thresholds (bottom)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.58])
    fig.add_trace(
        go.Scatter(x=ds.index, y=ds["spread"], name="Spread", line=dict(color=BLUE, width=2),
                   hovertemplate="%{y:.4f}"), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ds.index, y=ds["spread_mean"], name=f"Rolling mean ({params['zscore_window']} sessions)",
                   line=dict(color=INK_2, width=1.5, dash="dot"), hovertemplate="%{y:.4f}"), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ds.index, y=ds["z_score"], name="Z-score", line=dict(color=ORANGE, width=2),
                   hovertemplate="%{y:.2f}"), row=2, col=1,
    )
    entry, exit_, stop = params["entry_z"], params["exit_z"], params["stop_z"]
    for sign in (1, -1):
        _level(fig, sign * entry, f"entry {sign * entry:+g}", MUTED, "dash", 2, 1)
        _level(fig, sign * exit_, f"exit {sign * exit_:+g}", MUTED, "dot", 2, 1)
        _level(fig, sign * stop, f"stop {sign * stop:+g}", CRITICAL, "dash", 2, 1)

    if trades is not None and len(trades):
        t = trades[(trades["decision_date"] >= ds.index[0]) & (trades["decision_date"] <= ds.index[-1])]
        if len(t):
            fig.add_trace(
                go.Scatter(
                    x=t["decision_date"], y=t["entry_z"], mode="markers", name="Hybrid trade decision",
                    marker=dict(
                        color=AQUA, size=11, line=dict(color=SURFACE, width=2),
                        symbol=["triangle-up" if s == "LONG_SPREAD" else "triangle-down" for s in t["side"]],
                    ),
                    customdata=t[["side", "net_pnl", "exit_reason"]].to_numpy(),
                    hovertemplate="%{customdata[0]} · net P&L $%{customdata[1]:,.0f} · exit: %{customdata[2]}<extra>Hybrid trade</extra>",
                ), row=2, col=1,
            )
    _split_markers(fig, ds, rows=2)
    fig.update_yaxes(title_text="Log spread", row=1, col=1, title_font=dict(color=MUTED, size=11))
    fig.update_yaxes(title_text="Z-score", row=2, col=1, title_font=dict(color=MUTED, size=11))
    _style(fig, height=560)
    fig.update_layout(margin=dict(l=56, r=72, t=44, b=40))
    return fig


def price_chart(ds: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    """Both stocks indexed to 100 at the start of the visible window (one shared scale)."""
    base_a, base_b = ds["price_a"].iloc[0], ds["price_b"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ds.index, y=ds["price_a"] / base_a * 100, name=ticker_a, line=dict(color=BLUE, width=2),
        customdata=ds["price_a"], hovertemplate="%{y:.1f} (adj. close $%{customdata:.2f})"))
    fig.add_trace(go.Scatter(
        x=ds.index, y=ds["price_b"] / base_b * 100, name=ticker_b, line=dict(color=ORANGE, width=2),
        customdata=ds["price_b"], hovertemplate="%{y:.1f} (adj. close $%{customdata:.2f})"))
    _split_markers(fig, ds)
    fig.update_yaxes(title_text="Indexed adjusted close (window start = 100)", title_font=dict(color=MUTED, size=11))
    return _style(fig, height=400)


def probability_chart(ds: pd.DataFrame, threshold: float, entry_z: float) -> go.Figure:
    """Out-of-sample model probability over time, with entry-zone days marked."""
    d = ds[ds["probability"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["probability"], name="Model probability", line=dict(color=BLUE, width=2),
        hovertemplate="%{y:.2f}"))
    zone = d[d["abs_z"] >= entry_z]
    fig.add_trace(go.Scatter(
        x=zone.index, y=zone["probability"], mode="markers", name=f"Day with |z| ≥ {entry_z:g}",
        marker=dict(color=ORANGE, size=8, line=dict(color=SURFACE, width=2)),
        customdata=zone["z_score"], hovertemplate="%{y:.2f} (z = %{customdata:+.2f})"))
    _level(fig, threshold, f"threshold {threshold:.2f}", INK_2, "dash")
    _split_markers(fig, d)
    fig.update_yaxes(title_text="P(trade profitable after costs)", range=[0, 1], title_font=dict(color=MUTED, size=11))
    _style(fig, height=400)
    fig.update_layout(margin=dict(l=56, r=96, t=44, b=40))
    return fig


def _strategy_lines(equity: pd.DataFrame, column: str, fmt: str) -> go.Figure:
    fig = go.Figure()
    for name, color, label in [("baseline", VIOLET, "Baseline (rules only)"), ("hybrid", AQUA, "Hybrid (rules + ML + risk)")]:
        e = equity[equity["strategy"] == name]
        fig.add_trace(go.Scatter(x=e.index, y=e[column], name=label, line=dict(color=color, width=2), hovertemplate=fmt))
    return fig


def equity_chart(equity: pd.DataFrame, initial_capital: float) -> go.Figure:
    fig = _strategy_lines(equity, "equity", "$%{y:,.0f}")
    _level(fig, initial_capital, "starting capital", MUTED, "dot")
    fig.update_yaxes(title_text="Simulated equity ($)", tickformat="$,.0f", title_font=dict(color=MUTED, size=11))
    _style(fig, height=380)
    fig.update_layout(margin=dict(l=72, r=110, t=44, b=40))
    return fig


def drawdown_chart(equity: pd.DataFrame) -> go.Figure:
    fig = _strategy_lines(equity, "drawdown", "%{y:.2%}")
    fig.update_yaxes(title_text="Drawdown from peak", tickformat=".0%", title_font=dict(color=MUTED, size=11))
    return _style(fig, height=280)


def calibration_chart(calibration: list[dict]) -> go.Figure:
    """Predicted probability vs observed win rate. A calibrated model sits on the diagonal."""
    c = pd.DataFrame(calibration)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines",
                             line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"))
    if len(c):
        fig.add_trace(go.Scatter(
            x=c["mean_predicted"], y=c["observed_win_rate"], name="Model (test period)", mode="lines+markers+text",
            line=dict(color=BLUE, width=2), marker=dict(color=BLUE, size=10, line=dict(color=SURFACE, width=2)),
            text=[f"n={n}" for n in c["n"]],
            textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(c))], textfont=dict(color=INK_2, size=11),
            hovertemplate="predicted %{x:.2f} → observed %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Mean predicted probability", range=[0, 1], title_font=dict(color=MUTED, size=11))
    fig.update_yaxes(title_text="Observed win rate", range=[0, 1], title_font=dict(color=MUTED, size=11))
    return _style(fig, height=360, unified=False)


def zbucket_chart(buckets: list[dict]) -> go.Figure:
    """Observed win rate vs average predicted probability, by size of the deviation."""
    b = pd.DataFrame(buckets)
    fig = go.Figure()
    if len(b):
        labels = [f"|z| {x}<br>n={n}" for x, n in zip(b["abs_z_bucket"], b["n"])]
        fig.add_trace(go.Bar(x=labels, y=b["observed_win_rate"], name="Observed win rate",
                             marker=dict(color=BLUE, cornerradius=4, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y:.0%}"))
        fig.add_trace(go.Bar(x=labels, y=b["mean_predicted_probability"], name="Mean predicted probability",
                             marker=dict(color=ORANGE, cornerradius=4, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y:.0%}"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.04)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="Share of hypothetical trades", title_font=dict(color=MUTED, size=11))
    return _style(fig, height=360)


def importance_chart(importance: list[dict], descriptions: dict) -> go.Figure:
    """Permutation importance measured on the validation period (single series, no legend)."""
    imp = pd.DataFrame(importance).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=imp["importance"], y=[descriptions.get(f, f) for f in imp["feature"]], orientation="h",
        marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Drop in PR-AUC when the feature is shuffled", title_font=dict(color=MUTED, size=11),
                     zeroline=True, zerolinecolor=AXIS)
    _style(fig, height=440, unified=False)
    fig.update_layout(margin=dict(l=250, r=24, t=20, b=44), showlegend=False)
    return fig


def threshold_chart(search: pd.DataFrame, chosen: float) -> go.Figure:
    """Validation-period net P&L for each candidate probability threshold."""
    fig = go.Figure()
    for eligible, color, label in [(True, BLUE, "Enough validation trades"), (False, MUTED, "Too few trades (not eligible)")]:
        s = search[search["eligible"] == eligible]
        if len(s):
            fig.add_trace(go.Bar(
                x=[f"{t:.2f}" for t in s["threshold"]], y=s["net_pnl"], name=label,
                marker=dict(color=color, cornerradius=4),
                text=[f"{int(n)} trades" for n in s["n_trades"]], textposition="outside",
                textfont=dict(color=INK_2, size=11), hovertemplate="$%{y:,.0f}"))
    order = [f"{t:.2f}" for t in sorted(search["threshold"])]
    if f"{chosen:.2f}" in order:  # category axes are positioned by index, not by label
        fig.add_annotation(x=order.index(f"{chosen:.2f}"), y=0, yref="paper", yshift=-40, text="▲ selected",
                           showarrow=False, font=dict(color=INK, size=11))
    fig.update_xaxes(title_text="Probability threshold", type="category",
                     categoryorder="array", categoryarray=order,
                     title_font=dict(color=MUTED, size=11), title_standoff=36)
    fig.update_yaxes(title_text="Validation net P&L ($)", tickformat="$,.0f", title_font=dict(color=MUTED, size=11))
    _style(fig, height=360)
    fig.update_layout(bargap=0.4, margin=dict(l=72, r=24, t=44, b=84))
    return fig
