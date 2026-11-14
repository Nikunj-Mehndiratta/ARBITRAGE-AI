"""Build the one-page research paper (reports/ARBITRAGE_AI_paper.pdf) from the saved results.

    python reports/build_paper.py

Every number in the paper is read from reports/metrics.json and reports/equity_curves.csv,
so the paper cannot drift from the experiment. Needs two extra packages:
    pip install matplotlib reportlab
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

REPORTS = Path(__file__).resolve().parent
REPO_URL = sys.argv[1] if len(sys.argv) > 1 else "github.com/Nikunj-Mehndiratta/ARBITRAGE-AI"

INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#c3c2b7")
HYBRID, BASELINE = "#1baf7a", "#6250d6"  # same roles as the dashboard, light-surface steps


def register_fonts() -> None:
    """DejaVu Sans ships with matplotlib and has the Greek and maths glyphs the paper needs."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(font_dir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")


def money(x: float) -> str:
    return f"{'+' if x > 0 else '−'}${abs(x):,.0f}"


def pct(x: float, digits: int = 1, signed: bool = False) -> str:
    sign = "+" if (signed and x > 0) else ("−" if x < 0 else "")
    return f"{sign}{abs(x) * 100:.{digits}f}%"


def equity_figure(path: Path, capital: float) -> None:
    eq = pd.read_csv(REPORTS / "equity_curves.csv", parse_dates=["date"], index_col="date")
    eq = eq[eq["period"] == "test"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, ax = plt.subplots(figsize=(4.1, 2.25), dpi=300)
    for name, color, label in [("baseline", BASELINE, "Baseline (rules only)"), ("hybrid", HYBRID, "Hybrid (rules + ML + risk)")]:
        e = eq[eq["strategy"] == name]
        ax.plot(e.index, e["equity"] / 1000, color=color, linewidth=1.3, label=label)
    ax.axhline(capital / 1000, color="#898781", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Simulated equity ($ thousand)", color="#52514e")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(color="#e1e0d9", linewidth=0.5)
    ax.tick_params(colors="#898781", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def build() -> Path:
    register_fonts()
    m = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    p, t, v = m["parameters"], m["test"], m["validation"]
    base, hyb = t["baseline"], t["hybrid"]
    cls, cls_zone = t["classification_all_candidates"], t["classification_entry_zone"]
    ea = t["error_analysis"]
    a, b = m["pair"]["ticker_a"], m["pair"]["ticker_b"]
    sp, hedge, q = m["split"], m["hedge"], m["data"]["quality"]
    chosen = next(r for r in m["pair"]["screening"] if r["ticker_a"] == a)
    other = next((r for r in m["pair"]["screening"] if r["ticker_a"] != a), None)
    accepted = next(r for r in ea["decision_attribution"] if r["hybrid_decision"] == "accepted")
    rejected = next(r for r in ea["decision_attribution"] if r["hybrid_decision"] == "rejected")
    top_bin = max(t["calibration"], key=lambda r: r["mean_predicted"])
    ci = hyb["avg_trade_return_ci95"]
    thr = m["model"]["probability_threshold"]
    cost_bps = p["round_trip_cost_fraction_of_gross"] * 1e4
    max_val_trades = max(r["n_trades"] for r in v["threshold_search"])
    betas = chosen["window_betas"]
    betas = json.loads(betas) if isinstance(betas, str) else betas  # stored as text in the CSV

    fig_path = REPORTS / "figures" / "equity_test.png"
    fig_path.parent.mkdir(exist_ok=True)
    equity_figure(fig_path, p["initial_capital"])

    body = ParagraphStyle("body", fontName="Body", fontSize=8.3, leading=10.9, textColor=INK, spaceAfter=3.2)
    head = ParagraphStyle("head", parent=body, fontName="Body-Bold", fontSize=9.2, leading=11.5, spaceBefore=3.5, spaceAfter=2)
    title = ParagraphStyle("title", parent=body, fontName="Body-Bold", fontSize=15, leading=18.5, alignment=TA_CENTER, spaceAfter=2.5)
    sub = ParagraphStyle("sub", parent=body, fontSize=8.3, textColor=INK_2, alignment=TA_CENTER, spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=7.3, leading=9.2, textColor=INK_2, spaceAfter=0)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)

    story = [
        Paragraph("ARBITRAGE AI: Can a Machine-Learning Filter Improve a Pairs-Trading Rule?", title),
        Paragraph(
            f"A leakage-controlled paper-trading study on {a}/{b} &nbsp;·&nbsp; Research prototype, simulated portfolio only, "
            f"not investment advice &nbsp;·&nbsp; Code and data: {REPO_URL}",
            sub,
        ),
        Paragraph(
            f"<b>Abstract.</b> We build a hybrid statistical-arbitrage system: a mathematical engine detects when two related "
            f"stocks drift unusually far apart, and a gradient-boosting classifier estimates whether betting on the gap closing "
            f"will be profitable after costs. On {sp['test']['sessions']} unseen trading sessions ({sp['test']['start']} to "
            f"{sp['test']['end']}) the hybrid made {money(hyb['net_pnl'])} on ${p['initial_capital']:,.0f} of simulated capital, "
            f"against {money(base['net_pnl'])} for the rule-only baseline. The result rests on {hyb['n_trades']} trades and a "
            f"classifier barely above no-skill, so we report it as suggestive, not proven.",
            body,
        ),
        Paragraph("1. Problem", head),
        Paragraph(
            "Pairs trading assumes the prices of two related companies return to a stable relationship. A plain z-score rule "
            "trades every large deviation, including the ones that keep widening. We ask whether a model can tell in advance "
            "which deviations will revert within five sessions, and whether filtering on that improves simulated trading.",
            body,
        ),
        Paragraph("2. Method", head),
        Paragraph(
            f"<b>Statistical engine.</b> With log prices, we fit ln {b} = α + β·ln {a} by least squares and define the spread "
            f"S<sub>t</sub> = ln {b} − α − β·ln {a} and its rolling z-score Z<sub>t</sub> = (S<sub>t</sub> − μ<sub>t</sub>)/σ<sub>t</sub> "
            f"over {p['zscore_window']} sessions. <b>Baseline:</b> fade the spread when {p['entry_z']:g} ≤ |Z| &lt; {p['stop_z']:g}; "
            f"exit when |Z| ≤ {p['exit_z']:g}, when |Z| ≥ {p['stop_z']:g} (stop), or after {p['horizon_days']} sessions. "
            f"<b>Hybrid:</b> the same rule, but a trade is taken only if the model's probability p ≥ τ, the expected net return is "
            f"positive, and volatility, correlation and data-quality filters pass. <b>Model:</b> a small "
            f"HistGradientBoostingClassifier on {len(m['model']['features'])} features (z-score and its changes, spread volatility "
            f"and momentum, rolling correlation and hedge ratio, RSI, relative volume, market volatility). The label for day t is "
            f"whether the hypothetical trade decided at t ends with positive P&amp;L after a {cost_bps:.0f} bps round-trip cost, "
            f"filled {p['execution_lag']} session after the signal. Both legs are simulated ($1 of {b} against $β of {a}).",
            body,
        ),
        Paragraph("3. Protocol against data leakage", head),
        Paragraph(
            f"Daily adjusted closes from Yahoo Finance, {q['first_date']} to {q['last_date']} ({q['aligned_rows']:,} sessions), "
            f"split chronologically 70/15/15 with no shuffling. Pair choice, α, β and risk limits use the training period only "
            f"(to {sp['train']['end']}). The threshold τ = {thr:.2f} was chosen from {len(v['threshold_search'])} values on the "
            f"validation period; the model was then refit on training + validation and the test period evaluated once. A sample "
            f"is used for fitting only if its {sp['embargo_sessions']}-session forward label ends before the boundary. Unit tests "
            f"rewrite future prices and confirm that past features do not change.",
            body,
        ),
        Paragraph("4. Results on the unseen test period", head),
    ]

    rows = [
        ["", "Baseline", "Hybrid"],
        ["Net P&L", money(base["net_pnl"]), money(hyb["net_pnl"])],
        ["Trades", str(base["n_trades"]), str(hyb["n_trades"])],
        ["Win rate", pct(base["win_rate"], 0), pct(hyb["win_rate"], 0)],
        ["Avg net return / trade", pct(base["avg_trade_return"], 2, True), pct(hyb["avg_trade_return"], 2, True)],
        ["Sharpe ratio", f"{base['sharpe_ratio']:.2f}".replace("-", "−"), f"{hyb['sharpe_ratio']:.2f}".replace("-", "−")],
        ["Max drawdown", pct(base["max_drawdown"]), pct(hyb["max_drawdown"])],
        ["Classifier", "Model", "No-skill"],
        ["PR-AUC", f"{cls['pr_auc']:.3f}", f"{cls['pr_auc_no_skill']:.3f}"],
        ["ROC-AUC", f"{cls['roc_auc']:.3f}", "0.500"],
        ["Brier (lower is better)", f"{cls['brier_score']:.4f}", f"{cls['brier_no_skill']:.4f}"],
    ]
    results = Table(rows, colWidths=[3.9 * cm, 2.0 * cm, 2.0 * cm], rowHeights=0.43 * cm)
    results.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Body", 7.6),
                ("FONT", (0, 0), (-1, 0), "Body-Bold", 7.6),
                ("FONT", (0, 7), (-1, 7), "Body-Bold", 7.6),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                ("LINEABOVE", (0, 7), (-1, 7), 0.6, RULE),
                ("LINEBELOW", (0, 7), (-1, 7), 0.6, RULE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    figure = Image(str(fig_path), width=9.6 * cm, height=9.6 * cm * 2.25 / 4.1)
    side = Table([[results, figure]], colWidths=[8.1 * cm, 9.9 * cm])
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [
        side,
        Paragraph(
            f"Table 1 and Figure 1. Test period, {sp['test']['start']} to {sp['test']['end']}; ${p['initial_capital']:,.0f} starting "
            f"capital. Classifier metrics are for the {cls['n_samples']} test days with 1 ≤ |Z| &lt; {p['stop_z']:g}.",
            small,
        ),
        Paragraph("5. Where it fails", head),
        Paragraph(
            f"<b>The pair is not confirmed as mean-reverting.</b> The Engle–Granger test on training data gives p = "
            f"{chosen['coint_pvalue']:.2f} for {a}/{b}"
            + (f" ({other['ticker_a']}/{other['ticker_b']}: {other['coint_pvalue']:.2f})" if other else "")
            + f"; the spread's half-life is about {hedge['spread_half_life_sessions_train']:.0f} sessions, far longer than the "
            f"{p['horizon_days']}-session holding period, and β ranges from {min(betas):.2f} to {max(betas):.2f} across thirds of "
            f"the training period. <b>The classifier adds little.</b> Its Brier score is worse than always predicting the base "
            f"rate, and the days it scored highest (mean prediction {top_bin['mean_predicted']:.2f}) were profitable only "
            f"{pct(top_bin['observed_win_rate'], 0)} of the time. <b>The filter mostly trades less.</b> Among entry-zone signals, "
            f"those it accepted won {pct(accepted['win_rate'], 0)} of the time against {pct(rejected['win_rate'], 0)} for those it "
            f"rejected; the gain comes from a higher average return ({pct(accepted['avg_net_return'], 2)} vs "
            f"{pct(rejected['avg_net_return'], 2)}) on only {accepted['n_signals']} signals, and from avoiding the baseline's worst "
            f"trade ({pct(base['worst_trade_return'])}). <b>The uncertainty is large.</b> A 95% bootstrap interval for the hybrid's "
            f"average trade is {pct(ci[0], 2, True)} to {pct(ci[1], 2, True)}, which includes zero. Every losing trade is listed "
            f"with the model's probability in reports/failure_cases.csv.",
            body,
        ),
        Paragraph("6. Limitations and next steps", head),
        Paragraph(
            f"One pair and a few dozen trades cannot establish an edge. Costs are a flat assumption with no short-borrow fees, "
            f"labels from neighbouring days overlap, and τ was chosen on a validation period with at most {max_val_trades} trades "
            f"per candidate. The honest conclusion is that the pipeline and its safeguards work, and that the trading result is "
            f"not yet distinguishable from luck. Next: screen many pairs and keep only those that pass cointegration, retrain "
            f"walk-forward, calibrate the probabilities, and collect far more out-of-sample trades before making any claim.",
            body,
        ),
    ]

    out = REPORTS / "ARBITRAGE_AI_paper.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
        title="ARBITRAGE AI: Can a Machine-Learning Filter Improve a Pairs-Trading Rule?", author="ARBITRAGE AI",
    )
    doc.build(story)
    print(f"Wrote {out} ({doc.page} page{'s' if doc.page != 1 else ''})")
    if doc.page != 1:
        raise SystemExit("The paper no longer fits on one page; shorten the text.")
    return out


if __name__ == "__main__":
    build()
