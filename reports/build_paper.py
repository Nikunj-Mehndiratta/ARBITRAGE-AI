"""Build the one-page write-up (reports/ARBITRAGE_AI_paper.pdf) from the saved results.

    python reports/build_paper.py
    python reports/build_paper.py --team "First Name and Second Name"

The layout follows the hackathon's sample submission: the problem, what we built, what we
added, how it works, results, where it fails, what we'd do next, sources. Every number is
read from reports/metrics.json and the saved trades, so the write-up cannot drift from the
experiment. Needs two extra packages:  pip install matplotlib reportlab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPORTS = Path(__file__).resolve().parent
ROOT = REPORTS.parent
sys.path.insert(0, str(ROOT))

from arbitrage.config import load_config  # noqa: E402

INK = colors.HexColor("#1a1a19")
INK_2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d6cc")
SHADE = colors.HexColor("#f5f2ea")


def register_fonts() -> None:
    """DejaVu ships with matplotlib and has the Greek and maths glyphs the write-up needs."""
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in [
        ("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"), ("Body-Italic", "DejaVuSans-Oblique.ttf"),
        ("Body-BoldItalic", "DejaVuSans-BoldOblique.ttf"), ("Head", "DejaVuSerif.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(d / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")


def money(x: float) -> str:
    return f"{'+' if x > 0 else '−'}${abs(x):,.0f}"


def pct(x: float, digits: int = 1, signed: bool = False) -> str:
    sign = "+" if (signed and x > 0) else ("−" if x < 0 else "")
    return f"{sign}{abs(x) * 100:.{digits}f}%"


def minus(text: str) -> str:
    return text.replace("-", "−")


def best_hybrid_trade(m: dict) -> dict:
    """A real example for 'What we built': the hybrid's best trade in the test period."""
    trades = pd.read_csv(REPORTS / "experiment_results.csv")
    for col in ("decision_date", "exit_date"):
        trades[col] = pd.to_datetime(trades[col], format="mixed")
    t = trades[(trades.period == "test") & (trades.strategy == "hybrid")].sort_values("net_return").iloc[-1]
    ds = pd.read_csv(ROOT / load_config()["data"]["processed_dir"] / "dataset.csv", parse_dates=["date"], index_col="date")
    a, b = m["pair"]["ticker_a"], m["pair"]["ticker_b"]
    short = t.side == "SHORT_SPREAD"
    return {
        "date": f"{t.decision_date.day} {t.decision_date:%b %Y}",
        "z": float(t.entry_z),
        "prob": float(ds.loc[t.decision_date, "probability"]),
        "signal": "SHORT SPREAD" if short else "LONG SPREAD",
        "legs": f"short {b}, long {a}" if short else f"long {b}, short {a}",
        "ret": float(t.net_return),
        "days": int(t.holding_days),
    }


def build(team: str, track: str, repo_url: str) -> Path:
    register_fonts()
    m = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    p, t, v = m["parameters"], m["test"], m["validation"]
    base, hyb = t["baseline"], t["hybrid"]
    cls, ea = t["classification_all_candidates"], t["error_analysis"]
    a, b = m["pair"]["ticker_a"], m["pair"]["ticker_b"]
    sp, q = m["split"], m["data"]["quality"]
    chosen = next(r for r in m["pair"]["screening"] if r["ticker_a"] == a)
    other = next((r for r in m["pair"]["screening"] if r["ticker_a"] != a), None)
    accepted = next(r for r in ea["decision_attribution"] if r["hybrid_decision"] == "accepted")
    rejected = next(r for r in ea["decision_attribution"] if r["hybrid_decision"] == "rejected")
    top_bin = max(t["calibration"], key=lambda r: r["mean_predicted"])
    ci = hyb["avg_trade_return_ci95"]
    thr = m["model"]["probability_threshold"]
    cost_bps = p["round_trip_cost_fraction_of_gross"] * 1e4
    latest = m["latest"]
    ex = best_hybrid_trade(m)
    tickers = ", ".join(m["data"].get("tickers", {}).keys()) or f"{a}, {b}"
    retrieved = str(m["data"].get("retrieved_at_utc", ""))[:10]
    failures = pd.read_csv(REPORTS / "failure_cases.csv")
    hyb_losses = failures[failures.strategy == "hybrid"]

    body = ParagraphStyle("body", fontName="Body", fontSize=8.3, leading=10.75, textColor=INK, spaceAfter=2.8)
    head = ParagraphStyle("head", parent=body, fontName="Head", fontSize=11, leading=13.5, spaceBefore=3.6, spaceAfter=2)
    title = ParagraphStyle("title", parent=body, fontName="Head", fontSize=15.5, leading=19, spaceAfter=3.5)
    meta = ParagraphStyle("meta", parent=body, spaceAfter=1.5)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=1.6)
    quote = ParagraphStyle("quote", parent=body, spaceAfter=0, textColor=INK)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.1, leading=10.2, spaceAfter=0)
    cell_head = ParagraphStyle("cell_head", parent=cell, textColor=INK_2)

    def bullets(items: list[str]) -> list:
        return [Paragraph(text, bullet, bulletText="•") for text in items]

    example = Table(
        [[Paragraph(
            f"{a} / {b} on {ex['date']}: gap z-score {ex['z']:+.2f}, model probability {ex['prob']:.2f}<br/>"
            f"→ <b>{ex['signal']}</b> ({ex['legs']}). The gap is unusually wide, the model's probability is above "
            f"{thr:.2f} and the risk checks passed. Simulated result: {pct(ex['ret'], 2, True)} after costs in {ex['days']} sessions.",
            quote,
        )]],
        colWidths=[17.6 * cm],
    )
    example.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))

    def row(label: str, mm: dict) -> list:
        return [
            Paragraph(label, cell), Paragraph(money(mm["net_pnl"]), cell), Paragraph(str(mm["n_trades"]), cell),
            Paragraph(pct(mm["win_rate"], 0), cell), Paragraph(minus(f"{mm['sharpe_ratio']:.2f}"), cell),
            Paragraph(pct(mm["max_drawdown"]), cell),
        ]

    results = Table(
        [
            [Paragraph(h, cell_head) for h in
             ["System", f"Profit or loss on ${p['initial_capital']:,.0f}", "Trades", "Winning trades", "Sharpe ratio", "Worst drawdown"]],
            row("Baseline: z-score rule only", base),
            row("Hybrid: + ML filter and risk checks (final)", hyb),
        ],
        colWidths=[7.0 * cm, 3.1 * cm, 1.5 * cm, 2.1 * cm, 1.8 * cm, 2.1 * cm],
    )
    results.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE), ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))

    story = [
        Paragraph("ARBITRAGE AI: does machine learning make a pairs-trading rule any better?", title),
        Paragraph(f"<b>Track:</b> {track} · <b>Team:</b> {team} · <b>Code and data:</b> {repo_url}", meta),
        Paragraph("<i>This project is a research prototype that trades only in simulation. It is not financial advice.</i>", body),

        Paragraph("The problem", head),
        Paragraph(
            "Pairs trading sounds simple: when two related stocks such as Coca-Cola and PepsiCo drift apart, bet that they "
            "come back together. Many tutorials now add machine learning on top. A student "
            "learning quantitative finance has no easy way to tell whether the AI part really helps, or whether the backtest "
            "was lucky or quietly used future data. They need a tool that tests the idea fairly and shows its mistakes.",
            body,
        ),

        Paragraph("What we built", head),
        Paragraph(
            f"ARBITRAGE AI watches one pair of stocks and each day returns LONG SPREAD, SHORT SPREAD or NO SIGNAL, with the "
            f"reason. For example:",
            body,
        ),
        example,
        Spacer(1, 3),
        Paragraph(
            f"It trades only in a simulated portfolio. A Streamlit dashboard shows the gap between the two stocks, the model's "
            f"confidence, every simulated trade and every mistake. On {latest['date']} it says NO SIGNAL (z = "
            f"{minus(f'{latest['z_score']:.2f}')}).",
            body,
        ),

        Paragraph("What we added", head),
        Paragraph(
            f"A basic pairs-trading project uses a z-score rule and reports its profit. We added three things. First, a "
            f"machine-learning filter that estimates the chance each trade makes money after costs. Second, a fair-testing "
            f"protocol: data split by time, a {sp['embargo_sessions']}-session gap at each boundary, every setting chosen before "
            f"the test period, and unit tests that rewrite future prices to prove past inputs don't change. Third, an honest "
            f"error report: every losing trade, a comparison with plain guessing, and how uncertain the result is.",
            body,
        ),

        Paragraph("How it works", head),
        *bullets([
            f"<b>Data:</b> {q['aligned_rows']:,} days of adjusted closing prices for {a} and {b} from Yahoo Finance "
            f"({q['first_date']} to {q['last_date']}), split by time into 70% training, 15% validation and 15% final test.",
            f"<b>Baseline:</b> we fit ln {b} = α + β·ln {a}, call what is left over the spread, and measure how unusual it is "
            f"with a {p['zscore_window']}-day z-score. The rule bets on the gap closing when {p['entry_z']:g} ≤ |z| &lt; "
            f"{p['stop_z']:g}, and exits when |z| ≤ {p['exit_z']:g}, at a stop of {p['stop_z']:g}, or after {p['horizon_days']} days.",
            f"<b>Model:</b> a gradient-boosting classifier in scikit-learn with {len(m['model']['features'])} inputs (the "
            f"z-score and how fast it is moving, spread volatility, correlation, RSI, trading volume, market volatility). It "
            f"predicts whether the trade would end in profit after a {cost_bps:.0f} bps cost, with both stocks traded and the "
            f"order filled {p['execution_lag']} day after the signal.",
            f"<b>Our improvement:</b> the hybrid trades only when the model's probability is at least {thr:.2f} and the "
            f"volatility and correlation checks pass. We chose {thr:.2f} on the validation period, then ran the test period once.",
        ]),

        Paragraph("Results", head),
        results,
        Spacer(1, 4),
        Paragraph(
            f"Final test period, {sp['test']['start']} to {sp['test']['end']} ({sp['test']['sessions']} days the system had never "
            f"seen). Our biggest finding is that the hybrid made money where the baseline lost it, but mostly because it traded "
            f"less, not because the model predicts well. On {cls['n_samples']} test days the model's PR-AUC was "
            f"{cls['pr_auc']:.3f} against {cls['pr_auc_no_skill']:.3f} for guessing, and its Brier score ({cls['brier_score']:.4f}) "
            f"was slightly worse than always predicting the average ({cls['brier_no_skill']:.4f}).",
            body,
        ),
        Paragraph(
            f"<b>Where it fails.</b> The signals the model accepted won {pct(accepted['win_rate'], 0)} of the time and the ones "
            f"it rejected won {pct(rejected['win_rate'], 0)}, so it is barely choosing better trades. When it was most confident "
            f"(about {top_bin['mean_predicted']:.2f}), only {pct(top_bin['observed_win_rate'], 0)} of trades won. All "
            f"{len(hyb_losses)} of the hybrid's losing trades had probabilities between {hyb_losses.model_probability.min():.2f} "
            f"and {hyb_losses.model_probability.max():.2f}. {a} and {b} also failed our own cointegration test (p = "
            f"{chosen['coint_pvalue']:.2f}"
            + (f"; {other['ticker_a']}/{other['ticker_b']} scored {other['coint_pvalue']:.2f}" if other else "")
            + f"), so the gap is not proven to close, and it kept widening through the test period. With only "
            f"{hyb['n_trades']} trades, the 95% range for the average trade is {pct(ci[0], 2, True)} to {pct(ci[1], 2, True)}, "
            f"which includes zero. The result could be luck.",
            body,
        ),

        Paragraph("What we'd do next", head),
        *bullets([
            "Screen many pairs and keep only those that pass the cointegration test, instead of taking the better of two.",
            f"Retrain every few months and calibrate the probabilities, so that a score of {top_bin['mean_predicted']:.2f} "
            f"really means {top_bin['mean_predicted'] * 100:.0f}%.",
            "Collect many more unseen trades and add short-borrowing fees before making any claim about profit.",
        ]),

        Paragraph("Sources", head),
        *bullets([
            f"Daily prices from Yahoo Finance through the yfinance package ({tickers}; retrieved {retrieved}).",
            "scikit-learn, statsmodels, pandas, NumPy, Plotly and Streamlit. Background reading: Engle and Granger (1987) on "
            "cointegration; Gatev, Goetzmann and Rouwenhorst (2006) on pairs trading.",
            "AI tools: Claude (Anthropic) wrote the code, tests and dashboard from our plan and helped draft this write-up.",
        ]),
    ]

    out = REPORTS / "ARBITRAGE_AI_paper.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.1 * cm,
        title="ARBITRAGE AI: does machine learning make a pairs-trading rule any better?", author=team,
    )
    doc.build(story)
    print(f"Wrote {out} ({doc.page} page{'s' if doc.page != 1 else ''})")
    if doc.page != 1:
        raise SystemExit("The write-up no longer fits on one page; shorten the text.")
    return out


if __name__ == "__main__":
    project = load_config().get("project", {})
    parser = argparse.ArgumentParser(description="Build the one-page write-up from the saved results.")
    parser.add_argument("--team", default=project.get("team", "ARBITRAGE AI"), help="Team or author names for the byline")
    parser.add_argument("--track", default=project.get("track", "AI + Finance"))
    parser.add_argument("--repo-url", default=project.get("repo_url", "github.com/Nikunj-Mehndiratta/ARBITRAGE-AI"))
    args = parser.parse_args()
    build(args.team, args.track, args.repo_url)
