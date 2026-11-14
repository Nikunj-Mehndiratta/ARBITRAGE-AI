"""ARBITRAGE AI - quantitative research terminal (Streamlit dashboard).

    streamlit run app.py

The dashboard only *reads* the artefacts written by the pipeline (reports/ and
data/processed/). Every number on screen comes from a simulated portfolio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from arbitrage.config import load_config, project_path
from arbitrage.feature_engineering import FEATURE_DESCRIPTIONS
from dashboard import charts
from dashboard import components as ui

st.set_page_config(page_title="ARBITRAGE AI", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

CFG = load_config()
REPORTS = project_path(CFG["reports"]["dir"])
PROCESSED = project_path(CFG["data"]["processed_dir"])
REQUIRED = [REPORTS / "metrics.json", PROCESSED / "dataset.csv", REPORTS / "equity_curves.csv", REPORTS / "experiment_results.csv"]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _parse_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], format="mixed")
    return frame


@st.cache_data(show_spinner=False)
def load_artifacts(stamp: float) -> dict:
    """Read every pipeline output. ``stamp`` invalidates the cache when reports change."""
    date_cols = ["decision_date", "entry_date", "exit_date"]
    failures_path = REPORTS / "failure_cases.csv"
    try:
        failures = _parse_dates(pd.read_csv(failures_path), date_cols)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        failures = pd.DataFrame()
    return {
        "metrics": json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8")),
        "dataset": pd.read_csv(PROCESSED / "dataset.csv", parse_dates=["date"], index_col="date"),
        "equity": pd.read_csv(REPORTS / "equity_curves.csv", parse_dates=["date"], index_col="date"),
        "trades": _parse_dates(pd.read_csv(REPORTS / "experiment_results.csv"), date_cols),
        "failures": failures,
        "threshold_search": pd.read_csv(REPORTS / "threshold_search.csv"),
    }


def run_experiment(refresh: bool) -> None:
    from arbitrage.pipeline import run_all

    with st.spinner("Running the experiment: data → pair selection → training → test evaluation…"):
        run_all(refresh_data=refresh, verbose=False)
    st.cache_data.clear()
    st.rerun()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Experiment")
    refresh = st.checkbox("Re-download market data", value=False, help="Otherwise the cached Yahoo Finance bars in data/raw are reused.")
    if st.button("Run experiment", type="primary", width="stretch"):
        run_experiment(refresh)
    st.caption(
        "Runs the full pipeline. Pair selection, the hedge ratio, the model and the "
        "probability threshold are fitted without the test period; the test period is "
        "then evaluated once."
    )

ui.header()

# ------------------------------------------------------------------ empty state
if not all(p.exists() for p in REQUIRED):
    pair = CFG["data"]["candidate_pairs"][0]
    ui.tiles(
        [
            ("Active pair", f"{pair[0]} / {pair[1]}", "Candidate, not yet tested"),
            ("Model probability", "—", "Awaiting model output"),
            ("Spread z-score", "—", "Awaiting data"),
            ("Net simulated P&L", "—", "No backtest has been run"),
        ]
    )
    ui.signal_panel("NO_TRADE", "No market data or model has been run yet.", "—")
    st.info("No results yet. Use **Run experiment** in the sidebar, or run `python run_pipeline.py` in a terminal.")
    st.stop()

art = load_artifacts(max(_mtime(p) for p in REQUIRED))
M, ds, equity, trades = art["metrics"], art["dataset"], art["equity"], art["trades"]
A, B = M["pair"]["ticker_a"], M["pair"]["ticker_b"]
P, latest, test = M["parameters"], M["latest"], M["test"]
threshold = M["model"]["probability_threshold"]
hedge = M["hedge"]["final"]
test_trades = trades[trades["period"] == "test"]
test_equity = equity[equity["period"] == "test"]

with st.sidebar:
    st.markdown("### This run")
    st.markdown(
        f"- **Pair:** {A} / {B}\n"
        f"- **Data:** Yahoo Finance, daily, retrieved {M['data'].get('retrieved_at_utc', 'n/a')} UTC\n"
        f"- **History:** {M['data']['quality']['first_date']} → {M['data']['quality']['last_date']}\n"
        f"- **Test period:** {M['split']['test']['start']} → {M['split']['test']['end']} "
        f"({M['split']['test']['sessions']} sessions)\n"
        f"- **Model:** {M['model']['type'].replace('_', ' ')}\n"
        f"- **Generated:** {M['generated_at']}"
    )
    st.caption("Research prototype. Simulated portfolio only. Not investment advice.")

# ------------------------------------------------------------------ headline
warning = M["pair"]["decision"].get("warning")
if warning:
    st.warning(f"**Pair caveat.** {warning}")

hyb, base = test["hybrid"], test["baseline"]
ui.tiles(
    [
        ("Active pair", f"{A} / {B}", f"Hedge ratio β = {hedge['beta']:.3f} (fitted on {M['hedge']['final_fit_scope']})"),
        ("Model probability", ui.num(latest["probability"]), f"Trade threshold {threshold:.2f} · estimate, not a guarantee"),
        ("Spread z-score", ui.num(latest["z_score"], 2, True), latest["z_reading"].capitalize()),
        (
            "Net simulated P&L",
            ui.money(hyb["net_pnl"]),
            f"Hybrid, test period, {hyb['n_trades']} trades · baseline {ui.money(base['net_pnl'])}",
        ),
    ]
)
legs = ""
if latest["signal"] == "LONG_SPREAD":
    legs = f"simulated position: long {B}, short {A}"
elif latest["signal"] == "SHORT_SPREAD":
    legs = f"simulated position: short {B}, long {A}"
ui.signal_panel(latest["signal"], latest["reason"], str(latest["date"]), legs)

# ------------------------------------------------------------------ window filter
WINDOWS = ["Test period", "Last 12 months", "Validation + test", "Full history"]
window = st.segmented_control("History window", WINDOWS, default="Test period", key="window") or "Test period"
if window == "Test period":
    view = ds[ds["split"] == "test"]
elif window == "Last 12 months":
    view = ds[ds.index >= ds.index[-1] - pd.DateOffset(years=1)]
elif window == "Validation + test":
    view = ds[ds["split"] != "train"]
else:
    view = ds

tab_spread, tab_price, tab_model, tab_backtest, tab_errors, tab_protocol = st.tabs(
    ["Spread & z-score", "Stock prices", "Model probability", "Baseline vs hybrid", "Error analysis", "Protocol & data"]
)

# ------------------------------------------------------------------ spread
with tab_spread:
    st.markdown(f"#### How far is {B} from its historical relationship with {A}?")
    st.caption(
        f"Spread = ln({B}) − ({hedge['alpha']:.3f}) − {hedge['beta']:.3f} · ln({A}), with α and β fitted on "
        f"{hedge['fit_start']} → {hedge['fit_end']} only. The z-score compares the spread with its own last "
        f"{P['zscore_window']} sessions. Triangles mark hybrid trade decisions in the test period (▲ long spread, ▼ short spread)."
    )
    hybrid_test = test_trades[test_trades["strategy"] == "hybrid"]
    st.plotly_chart(charts.spread_zscore_chart(view, P, hybrid_test), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions with |z| ≥ entry (window)", f"{int((view['abs_z'] >= P['entry_z']).sum())} of {len(view)}")
    hl = M["hedge"]["spread_half_life_sessions_train"]
    c2.metric("Spread half-life (training period)", f"{hl:.0f} sessions" if hl else "not mean-reverting")
    c3.metric("Hedge regression R² (fit window)", f"{hedge['r_squared']:.3f}")
    with st.expander("Table view"):
        st.dataframe(view[["price_a", "price_b", "spread", "spread_mean", "spread_std", "z_score"]].round(4).iloc[::-1], width="stretch", height=280)

# ------------------------------------------------------------------ prices
with tab_price:
    st.markdown(f"#### {A} and {B}, indexed to the start of the window")
    st.caption("Adjusted closing prices (splits and dividends), rebased to 100 so both stocks share one scale.")
    st.plotly_chart(charts.price_chart(view, A, B), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{A} adjusted close", f"${latest['price_a']:.2f}")
    c2.metric(f"{B} adjusted close", f"${latest['price_b']:.2f}")
    c3.metric("60-session return correlation", ui.num(ds["rolling_corr_60"].iloc[-1]))

# ------------------------------------------------------------------ model
with tab_model:
    st.markdown("#### Estimated probability that fading the spread is profitable after costs")
    st.caption(
        "Out-of-sample only: validation-period probabilities come from the model trained on the training period; "
        "test-period probabilities from the final model. Training-period probabilities are not shown because they are in-sample."
    )
    prob_view = view if view["probability"].notna().any() else ds[ds["split"] != "train"]
    st.plotly_chart(charts.probability_chart(prob_view, threshold, P["entry_z"]), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Threshold chosen on validation data only")
        st.caption(f"Selected: {threshold:.2f} — {M['model']['threshold_source']}. The test period played no part.")
        st.plotly_chart(charts.threshold_chart(art["threshold_search"], threshold), width="stretch")
    with right:
        st.markdown("##### What the model relies on")
        st.caption("Permutation importance on the validation period. Small values mean the feature barely matters.")
        if M["model"]["feature_importance_validation"]:
            st.plotly_chart(charts.importance_chart(M["model"]["feature_importance_validation"], FEATURE_DESCRIPTIONS), width="stretch")

# ------------------------------------------------------------------ backtest
with tab_backtest:
    st.markdown("#### Same statistical engine, with and without the ML filter — final test period")
    st.caption(
        f"{M['split']['test']['start']} → {M['split']['test']['end']}, never used for fitting or tuning. "
        f"Starting capital ${P['initial_capital']:,.0f}; round-trip cost {P['round_trip_cost_fraction_of_gross'] * 1e4:.0f} bps of gross notional; "
        f"signals filled {P['execution_lag']} session after the decision; maximum holding {P['horizon_days']} sessions."
    )
    st.dataframe(ui.strategy_table(base, hyb), hide_index=True, width="stretch")
    st.markdown("##### Simulated equity")
    st.plotly_chart(charts.equity_chart(test_equity, P["initial_capital"]), width="stretch")
    st.markdown("##### Drawdown")
    st.plotly_chart(charts.drawdown_chart(test_equity), width="stretch")

    st.markdown("##### Simulated trades")
    which = st.segmented_control("Strategy", ["Hybrid", "Baseline"], default="Hybrid", key="trade_strategy") or "Hybrid"
    shown = test_trades[test_trades["strategy"] == which.lower()].copy()
    for col in ["decision_date", "entry_date", "exit_date"]:
        shown[col] = shown[col].dt.date
    st.dataframe(
        shown[["decision_date", "entry_date", "exit_date", "side", "entry_z", "exit_reason", "holding_days", "net_pnl", "net_return"]],
        hide_index=True, width="stretch",
        column_config={
            "entry_z": st.column_config.NumberColumn("entry z", format="%.2f"),
            "net_pnl": st.column_config.NumberColumn("net P&L ($)", format="%.0f"),
            "net_return": st.column_config.NumberColumn("net return", format="percent"),
        },
    )
    with st.expander("Validation-period results (used for tuning, shown for transparency)"):
        st.dataframe(ui.strategy_table({**M["validation"]["baseline"]}, {**M["validation"]["hybrid"]}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ errors
with tab_errors:
    cls_all, cls_entry = test["classification_all_candidates"], test["classification_entry_zone"]
    ea = test["error_analysis"]
    st.markdown("#### Where the model is wrong — test period")
    st.caption("Classification quality is reported separately from trading results: a model can predict well and still trade badly, or the reverse.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"##### All candidate days (|z| ≥ {CFG['model']['train_min_abs_z']:g})")
        st.dataframe(ui.classification_table(cls_all), hide_index=True, width="stretch")
    with right:
        st.markdown(f"##### Entry-zone days only (|z| ≥ {P['entry_z']:g})")
        st.dataframe(ui.classification_table(cls_entry), hide_index=True, width="stretch")

    skill = cls_all["pr_auc"] - cls_all["pr_auc_no_skill"]
    brier_gap = cls_all["brier_score"] - cls_all["brier_no_skill"]
    st.info(
        f"**Reading these numbers honestly.** PR-AUC is {cls_all['pr_auc']:.3f} against a no-skill level of "
        f"{cls_all['pr_auc_no_skill']:.3f} ({skill:+.3f}). The Brier score is {cls_all['brier_score']:.4f} against "
        f"{cls_all['brier_no_skill']:.4f} for always predicting the base rate "
        f"({'worse' if brier_gap > 0 else 'better'} by {abs(brier_gap):.4f}). "
        f"The hybrid's edge over the baseline rests on {hyb['n_trades']} test trades, so it may be noise."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### Calibration")
        st.caption("If the model says 0.65, do about 65% of those trades win?")
        st.plotly_chart(charts.calibration_chart(test["calibration"]), width="stretch")
    with right:
        st.markdown("##### By size of deviation")
        st.caption("Observed win rate of the hypothetical trade vs what the model predicted.")
        st.plotly_chart(charts.zbucket_chart(ea["by_abs_z_bucket"]), width="stretch")

    cm = cls_entry["confusion_matrix"]
    st.markdown("##### Did the filter keep the right signals?")
    st.caption(ea["note"])
    attribution = pd.DataFrame(ea["decision_attribution"]).rename(
        columns={"hybrid_decision": "Hybrid decision", "n_signals": "Signal-days", "win_rate": "Win rate",
                 "avg_net_return": "Avg net return", "sum_net_return": "Sum of net returns"}
    )
    st.dataframe(
        attribution, hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in ["Win rate", "Avg net return", "Sum of net returns"]},
    )
    st.caption(f"Model verdict on entry-zone days at threshold {threshold:.2f}, against what actually happened:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model said trade · it won", cm["tp"])
    c2.metric("Model said trade · it lost", cm["fp"], help="False positives")
    c3.metric("Model said skip · it would have won", cm["fn"], help="Missed opportunities")
    c4.metric("Model said skip · it would have lost", cm["tn"])
    if ea["rejection_reasons"]:
        st.caption("Why entry-zone signals were rejected: " + "; ".join(f"{k} ({v})" for k, v in ea["rejection_reasons"].items()))

    st.markdown("##### Losing trades, worst first")
    fails = art["failures"]
    if len(fails):
        f = fails.copy()
        for col in ["decision_date", "exit_date"]:
            f[col] = f[col].dt.date
        st.dataframe(
            f[["strategy", "decision_date", "exit_date", "side", "entry_z", "exit_reason", "net_pnl", "net_return",
               "model_probability", "at_decision_market_vol_20", "at_decision_rolling_corr_60"]],
            hide_index=True, width="stretch",
            column_config={
                "entry_z": st.column_config.NumberColumn("entry z", format="%.2f"),
                "net_pnl": st.column_config.NumberColumn("net P&L ($)", format="%.0f"),
                "net_return": st.column_config.NumberColumn("net return", format="percent"),
                "model_probability": st.column_config.NumberColumn("model probability", format="%.2f"),
                "at_decision_market_vol_20": st.column_config.NumberColumn("market vol", format="%.2f"),
                "at_decision_rolling_corr_60": st.column_config.NumberColumn("pair correlation", format="%.2f"),
            },
        )
    else:
        st.caption("No losing trades were recorded in the test period.")

# ------------------------------------------------------------------ protocol
with tab_protocol:
    st.markdown("#### Anti-leakage protocol and data provenance")
    sp = M["split"]
    st.dataframe(
        pd.DataFrame(
            [
                ("Training", sp["train"]["start"], sp["train"]["end"], sp["train"]["sessions"], "Pair selection, hedge ratio, model, risk limits"),
                ("Validation", sp["validation"]["start"], sp["validation"]["end"], sp["validation"]["sessions"], "Probability threshold; then merged into the final refit"),
                ("Final test", sp["test"]["start"], sp["test"]["end"], sp["test"]["sessions"], "Evaluated once. Nothing is fitted or tuned here"),
            ],
            columns=["Period", "From", "To", "Sessions", "Used for"],
        ),
        hide_index=True, width="stretch",
    )
    st.markdown(
        f"""
- **Chronological split, never shuffled.** Rows keep their time order and the three periods never interleave.
- **Embargo of {sp['embargo_sessions']} sessions at each boundary:** {sp['embargo_rule']}.
- **Hedge ratio and pair selection** use the training period only; the final refit adds validation, never test.
- **Feature transformations** live inside a scikit-learn Pipeline fitted on the fitting rows only.
- **Features at day t** use data up to the close of day t; a unit test changes future prices and checks that past z-scores do not move.
- **Risk limits** are quantiles of the fitting period.
- **Losing trades and model errors are reported**, not hidden (see *Error analysis*).
"""
    )
    st.markdown("##### Pair screening (training period only)")
    screening = pd.DataFrame(M["pair"]["screening"])[
        ["ticker_a", "ticker_b", "return_correlation", "coint_pvalue", "stability_share", "hedge_ratio_beta",
         "window_betas", "passes_correlation", "passes_liquidity", "passes_cointegration"]
    ]
    st.dataframe(
        screening, hide_index=True, width="stretch",
        column_config={
            "return_correlation": st.column_config.NumberColumn("return corr.", format="%.2f"),
            "coint_pvalue": st.column_config.NumberColumn("Engle–Granger p", format="%.3f"),
            "stability_share": st.column_config.NumberColumn("stable sub-windows", format="percent"),
            "hedge_ratio_beta": st.column_config.NumberColumn("β", format="%.3f"),
            "window_betas": "β by sub-window",
        },
    )
    st.caption(f"Selection: {M['pair']['decision']['method']}.")

    st.markdown("##### Data")
    q = M["data"]["quality"]
    st.markdown(
        f"- **Source:** {M['data'].get('source', 'Yahoo Finance via yfinance')} (retrieved {M['data'].get('retrieved_at_utc', 'n/a')} UTC)\n"
        f"- **Price field:** {M['data'].get('price_field_used', 'adjusted close')}\n"
        f"- **Aligned sessions:** {q['aligned_rows']:,} ({q['first_date']} → {q['last_date']}); longest calendar gap {q['max_calendar_gap_days']} days\n"
        f"- **Rows flagged as suspect and excluded from signals:** {q['quality_flagged_rows']}"
    )
    st.markdown("##### Limitations")
    st.markdown("\n".join(f"- {c}" for c in M["caveats"]))
