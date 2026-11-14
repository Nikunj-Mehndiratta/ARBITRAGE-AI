"""End-to-end experiment orchestration with the anti-leakage protocol built in.

``train``     data -> pair selection (train only) -> stage-1 model (train only) ->
              validation metrics + threshold tuning (validation only) -> final refit on
              train + validation -> saved model bundle. The test period is never read.

``evaluate``  loads the frozen bundle and touches the final test period exactly once:
              classification metrics, baseline vs hybrid backtest, failure analysis.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtester import TradeParams, performance_metrics, run_backtest
from .config import all_tickers, load_config, project_path
from .data_loader import get_prices
from .feature_engineering import FEATURE_COLUMNS, build_dataset, candidate_mask
from .pair_selection import select_pair
from .prediction_engine import (
    calibration_table,
    classification_metrics,
    feature_importance,
    fit_model,
    load_bundle,
    predict_probability,
    save_bundle,
)
from .preprocessing import Split, build_panel, chronological_split
from .risk_manager import RiskLimits, fit_risk_limits, risk_check
from .signal_fusion import baseline_signals, fuse_signal, hybrid_signals
from .statistical_engine import HedgeModel, classify_z, fit_hedge_ratio, half_life


# --------------------------------------------------------------------------- helpers
def _clean(obj):
    """Make an object JSON-serialisable (numpy scalars, NaN, timestamps)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def prepare_panel(cfg: dict, refresh: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    """Market data layer: load (or download) bars, validate, align, cache the panel."""
    d = cfg["data"]
    frames, meta = get_prices(all_tickers(cfg), d["start_date"], d["end_date"], project_path(d["raw_dir"]), refresh)
    panel, quality = build_panel(frames, float(d["max_abs_daily_return"]))
    processed = project_path(d["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed / "prices.csv")
    return panel, meta, quality


def _bootstrap_mean_ci(trades: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> list | None:
    """95% bootstrap interval for the mean net return per trade (None if under 3 trades)."""
    if len(trades) < 3:
        return None
    r = trades["net_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _outcome_stats(dataset: pd.DataFrame, rows: pd.Series, entry_z: float) -> dict:
    """Average winning / losing net return of training-period hypothetical trades."""
    sample = dataset[rows & (dataset["abs_z"] >= entry_z)]
    scope = "training samples with |z| >= entry threshold"
    if len(sample) < 30:
        sample, scope = dataset[rows], "all training samples"
    wins = sample.loc[sample.trade_net_return > 0, "trade_net_return"]
    losses = sample.loc[sample.trade_net_return <= 0, "trade_net_return"]
    return {
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "n": int(len(sample)),
        "scope": scope,
    }


def _run_strategies(
    dataset: pd.DataFrame,
    probability: pd.Series,
    limits: RiskLimits,
    stats: dict,
    threshold: float,
    beta: float,
    cfg: dict,
    start: int,
    end: int,
    period: str,
) -> dict:
    """Run the rule-based baseline and the hybrid strategy over the same rows."""
    params = TradeParams.from_config(cfg)
    capital = float(cfg["trading"]["initial_capital"])
    exposure = float(cfg["trading"]["gross_exposure"])

    risk = risk_check(dataset, limits)
    base_sig = baseline_signals(dataset["z_score"], params.entry_z, params.stop_z)
    hyb = hybrid_signals(
        dataset, probability, risk, threshold, stats["avg_win"], stats["avg_loss"], params.entry_z, params.stop_z
    )

    out = {"risk": risk, "baseline_signal": base_sig, "hybrid_signals": hyb}
    for name, sig in [("baseline", base_sig), ("hybrid", hyb["signal"])]:
        trades, equity = run_backtest(dataset, sig, beta, params, start, end, capital, exposure, name)
        trades["period"], equity["period"] = period, period
        out[name] = {
            "trades": trades,
            "equity": equity,
            "metrics": performance_metrics(trades, equity, capital),
        }
    return out


def _split_summary(panel: pd.DataFrame, split: Split, params: TradeParams) -> dict:
    idx = panel.index

    def part(lo: int, hi: int) -> dict:
        return {"start": idx[lo], "end": idx[hi - 1], "sessions": hi - lo}

    return {
        "method": "chronological, no shuffling",
        "train": part(0, split.train_end),
        "validation": part(split.train_end, split.val_end),
        "test": part(split.val_end, split.n),
        "embargo_sessions": params.label_span,
        "embargo_rule": (
            "a sample is used for fitting only if its forward-looking label "
            f"({params.execution_lag} execution-lag + {params.horizon_days} holding sessions) "
            "is fully resolved before the split boundary"
        ),
    }


# ----------------------------------------------------------------------------- train
def train(cfg: dict | None = None, refresh_data: bool = False, verbose: bool = True) -> dict:
    """Fit everything that may be fitted without looking at the test period."""
    cfg = cfg or load_config()
    params = TradeParams.from_config(cfg)
    m = cfg["model"]
    reports_dir = project_path(cfg["reports"]["dir"])
    processed_dir = project_path(cfg["data"]["processed_dir"])

    panel, data_meta, quality = prepare_panel(cfg, refresh_data)
    split = chronological_split(len(panel), cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    pos = np.arange(len(panel))
    in_val = (pos >= split.train_end) & (pos < split.val_end)

    # 1. Pair selection - training slice only.
    train_panel = panel.iloc[: split.train_end]
    pair, pair_report, decision = select_pair(train_panel, cfg)
    a, b = pair

    # 2. Stage 1: hedge ratio and model from the training period only.
    hedge1 = fit_hedge_ratio(np.log(train_panel[f"{a}_adj_close"]), np.log(train_panel[f"{b}_adj_close"]), a, b)
    ds1 = build_dataset(panel, hedge1, cfg)
    cand1 = candidate_mask(ds1, cfg)
    train_rows = cand1 & (ds1["label_end_pos"] < split.train_end)
    val_rows = cand1 & in_val & (ds1["label_end_pos"] < split.val_end)

    model1 = fit_model(ds1.loc[train_rows, FEATURE_COLUMNS], ds1.loc[train_rows, "label"], m["type"], m["random_state"])
    prob1 = pd.Series(np.nan, index=ds1.index, name="probability")
    prob1[in_val] = predict_probability(model1, ds1.loc[in_val, FEATURE_COLUMNS])

    limits1 = fit_risk_limits(ds1.iloc[: split.train_end], cfg)
    stats1 = _outcome_stats(ds1, train_rows, params.entry_z)

    # 3. Threshold tuning - validation period only.
    search_rows = []
    grid = sorted(set(m["threshold_grid"]) | {m["probability_threshold"]}) if m["tune_threshold_on_validation"] else [
        m["probability_threshold"]
    ]
    for thr in grid:
        res = _run_strategies(ds1, prob1, limits1, stats1, thr, hedge1.beta, cfg, split.train_end, split.val_end, "validation")
        hm = res["hybrid"]["metrics"]
        search_rows.append(
            {
                "threshold": thr,
                "n_trades": hm["n_trades"],
                "net_pnl": hm["net_pnl"],
                "win_rate": hm["win_rate"],
                "sharpe_ratio": hm["sharpe_ratio"],
                "max_drawdown": hm["max_drawdown"],
                "eligible": hm["n_trades"] >= m["min_validation_trades"],
            }
        )
    search = pd.DataFrame(search_rows)
    eligible = search[search.eligible]
    if m["tune_threshold_on_validation"] and len(eligible):
        threshold = float(eligible.sort_values(["net_pnl", "threshold"], ascending=[False, True]).iloc[0].threshold)
        threshold_source = "highest validation net P&L among thresholds with enough validation trades"
    else:
        threshold = float(m["probability_threshold"])
        threshold_source = (
            "config default (tuning disabled)"
            if not m["tune_threshold_on_validation"]
            else "config default (no threshold produced enough validation trades)"
        )

    val = _run_strategies(ds1, prob1, limits1, stats1, threshold, hedge1.beta, cfg, split.train_end, split.val_end, "validation")
    entry_zone = val_rows & (ds1["abs_z"] >= params.entry_z)
    validation_report = {
        "classification_all_candidates": classification_metrics(ds1.loc[val_rows, "label"], prob1[val_rows], threshold),
        "classification_entry_zone": classification_metrics(ds1.loc[entry_zone, "label"], prob1[entry_zone], threshold),
        "baseline": val["baseline"]["metrics"],
        "hybrid": val["hybrid"]["metrics"],
        "threshold_search": search.to_dict("records"),
    }
    importance = feature_importance(model1, ds1.loc[val_rows, FEATURE_COLUMNS], ds1.loc[val_rows, "label"], m["random_state"])

    # 4. Final fit. Refit on train + validation (still no test data) or keep stage 1.
    if m["refit_on_train_plus_validation"]:
        fit_end = split.val_end
        fit_panel = panel.iloc[:fit_end]
        hedge = fit_hedge_ratio(np.log(fit_panel[f"{a}_adj_close"]), np.log(fit_panel[f"{b}_adj_close"]), a, b)
        ds = build_dataset(panel, hedge, cfg)
        final_rows = candidate_mask(ds, cfg) & (ds["label_end_pos"] < fit_end)
        model = fit_model(ds.loc[final_rows, FEATURE_COLUMNS], ds.loc[final_rows, "label"], m["type"], m["random_state"])
        limits = fit_risk_limits(ds.iloc[:fit_end], cfg)
        stats = _outcome_stats(ds, final_rows, params.entry_z)
        fit_scope = "train + validation"
    else:
        fit_end, hedge, ds, final_rows, model, limits, stats = (
            split.train_end, hedge1, ds1, train_rows, model1, limits1, stats1
        )
        fit_scope = "train only"

    bundle = {
        "model": model,
        "model_type": m["type"],
        "feature_columns": FEATURE_COLUMNS,
        "hedge": hedge.to_dict(),
        "risk_limits": limits.to_dict(),
        "outcome_stats": stats,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "pair": list(pair),
        "pair_decision": decision,
        "split": split.to_dict(),
        "fit_end_pos": int(fit_end),
        "fit_scope": fit_scope,
        "n_fit_samples": int(final_rows.sum()),
        "fit_base_rate": float(ds.loc[final_rows, "label"].mean()),
        "stage1": {
            "hedge": hedge1.to_dict(),
            "n_train_samples": int(train_rows.sum()),
            "train_base_rate": float(ds1.loc[train_rows, "label"].mean()),
            "half_life_train": half_life(ds1["spread"].iloc[: split.train_end]),
        },
        "validation": validation_report,
        "feature_importance": importance,
        "config": cfg,
        "data_meta": data_meta,
        "data_quality": quality,
        "panel_fingerprint": {"rows": int(len(panel)), "last_date": str(panel.index[-1].date())},
        "trained_at": _now(),
    }
    model_path = save_bundle(bundle, project_path(cfg["reports"]["model_dir"]))

    # Validation artefacts (used by the dashboard's validation-only threshold explorer).
    reports_dir.mkdir(parents=True, exist_ok=True)
    pair_report.to_csv(reports_dir / "pair_selection.csv", index=False)
    search.to_csv(reports_dir / "threshold_search.csv", index=False)
    val_trades = pd.concat([val["baseline"]["trades"], val["hybrid"]["trades"]], ignore_index=True)
    val_trades.to_csv(reports_dir / "validation_trades.csv", index=False)
    pd.concat([val["baseline"]["equity"], val["hybrid"]["equity"]]).to_csv(reports_dir / "validation_equity.csv")

    val_ds = ds1.loc[in_val].copy()
    val_ds["probability"] = prob1[in_val]
    val_ds = val_ds.join(val["risk"].loc[in_val, ["risk_ok", "risk_reason"]])
    val_ds.to_csv(processed_dir / "validation_dataset.csv")
    _write_json(
        reports_dir / "validation_metrics.json",
        {"generated_at": _now(), "threshold": threshold, "threshold_source": threshold_source, **validation_report},
    )

    if verbose:
        print(f"Pair            : {a}/{b}  ({decision['method']})")
        if decision.get("warning"):
            print(f"  WARNING       : {decision['warning']}")
        print(f"Stage-1 samples : {int(train_rows.sum())} train, {int(val_rows.sum())} validation")
        print(f"Threshold       : {threshold:.2f}  ({threshold_source})")
        vb, vh = validation_report["baseline"], validation_report["hybrid"]
        print(f"Validation P&L  : baseline {vb['net_pnl']:+,.0f} ({vb['n_trades']} trades) | hybrid {vh['net_pnl']:+,.0f} ({vh['n_trades']} trades)")
        print(f"Final model     : {m['type']} fitted on {fit_scope} ({int(final_rows.sum())} samples)")
        print(f"Saved           : {model_path}")
    return bundle


# -------------------------------------------------------------------------- evaluate
def evaluate(verbose: bool = True) -> dict:
    """Run the frozen model on the final test period and write every report."""
    base_cfg = load_config()
    bundle = load_bundle(project_path(base_cfg["reports"]["model_dir"]))
    cfg = bundle["config"]  # evaluate with exactly the configuration the model was trained under
    params = TradeParams.from_config(cfg)
    reports_dir = project_path(cfg["reports"]["dir"])
    processed_dir = project_path(cfg["data"]["processed_dir"])

    panel = pd.read_csv(processed_dir / "prices.csv", parse_dates=["date"], index_col="date")
    fp = bundle["panel_fingerprint"]
    if len(panel) != fp["rows"] or str(panel.index[-1].date()) != fp["last_date"]:
        raise RuntimeError("Price panel changed since training. Re-run `python -m models.train` first.")

    split = Split(**bundle["split"])
    hedge = HedgeModel(**bundle["hedge"])
    limits = RiskLimits(**bundle["risk_limits"])
    stats, threshold = bundle["outcome_stats"], float(bundle["threshold"])
    a, b = bundle["pair"]

    ds = build_dataset(panel, hedge, cfg)
    pos = np.arange(len(ds))
    in_test = pos >= split.val_end

    prob = pd.Series(np.nan, index=ds.index, name="probability")
    prob[in_test] = predict_probability(bundle["model"], ds.loc[in_test, FEATURE_COLUMNS])

    test = _run_strategies(ds, prob, limits, stats, threshold, hedge.beta, cfg, split.val_end, split.n, "test")
    for name in ("baseline", "hybrid"):  # how uncertain is the average trade? (bootstrap over trades)
        test[name]["metrics"]["avg_trade_return_ci95"] = _bootstrap_mean_ci(test[name]["trades"])

    # --- classification, reported separately from the trading simulation ---
    cand = candidate_mask(ds, cfg) & in_test
    entry_zone = cand & (ds["abs_z"] >= params.entry_z)
    cls_all = classification_metrics(ds.loc[cand, "label"], prob[cand], threshold)
    cls_entry = classification_metrics(ds.loc[entry_zone, "label"], prob[entry_zone], threshold)
    calibration = calibration_table(ds.loc[cand, "label"], prob[cand])

    # --- error analysis ---
    fused = test["risk"].join(test["hybrid_signals"])  # per-row risk verdict + hybrid decision

    zone = ds.loc[entry_zone, ["z_score", "trade_net_return", "label"]].join(fused.loc[entry_zone, ["signal_name", "reason"]])
    zone["probability"] = prob[entry_zone]
    zone["accepted"] = zone["signal_name"] != "NO_TRADE"
    attribution = []
    for accepted, grp in zone.groupby("accepted"):
        attribution.append(
            {
                "hybrid_decision": "accepted" if accepted else "rejected",
                "n_signals": int(len(grp)),
                "win_rate": float(grp["label"].mean()),
                "avg_net_return": float(grp["trade_net_return"].mean()),
                "sum_net_return": float(grp["trade_net_return"].sum()),
            }
        )
    rejection_reasons = (
        zone.loc[~zone.accepted, "reason"].str.replace(r"\s\d+\.\d+", "", regex=True).value_counts().to_dict()
    )

    buckets = []
    edges = [cfg["model"]["train_min_abs_z"], 1.5, 2.0, 2.5, cfg["statistical"]["stop_z"]]
    cand_df = ds.loc[cand, ["abs_z", "label", "trade_net_return"]].assign(probability=prob[cand])
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = cand_df[(cand_df.abs_z >= lo) & (cand_df.abs_z < hi)]
        if len(g):
            buckets.append(
                {
                    "abs_z_bucket": f"{lo:g}-{hi:g}",
                    "n": int(len(g)),
                    "observed_win_rate": float(g.label.mean()),
                    "mean_predicted_probability": float(g.probability.mean()),
                    "avg_net_return": float(g.trade_net_return.mean()),
                }
            )

    feature_view = ["z_score", "z_change_5d", "spread_vol_20", "rolling_corr_60", "market_vol_20", "rsi_a", "rsi_b"]
    failures = []
    for strategy in ("hybrid", "baseline"):
        trades = test[strategy]["trades"]
        if len(trades):
            losers = trades[trades.net_pnl <= 0].sort_values("net_return").copy()
            losers["model_probability"] = prob.iloc[losers["decision_pos"].to_numpy()].to_numpy()
            ctx = ds.iloc[losers["decision_pos"].to_numpy()][feature_view].reset_index(drop=True)
            failures.append(pd.concat([losers.reset_index(drop=True), ctx.add_prefix("at_decision_")], axis=1))
    failure_cases = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()

    false_pos = zone[(zone.accepted) & (zone.label == 0)]
    error_summary = {
        "decision_attribution": attribution,
        "rejection_reasons": rejection_reasons,
        "false_positive_signals": int(len(false_pos)),
        "accepted_signals": int(zone.accepted.sum()),
        "by_abs_z_bucket": buckets,
        "note": (
            "Attribution is per signal-day in the entry zone (overlapping hypothetical trades), "
            "so counts exceed the number of sequential backtest trades."
        ),
    }

    # --- latest observation: what the system says right now ---
    last = ds.iloc[-1]
    last_prob = float(prob.iloc[-1])
    last_risk = test["risk"].iloc[-1]
    last_exp = float(last_prob * stats["avg_win"] + (1 - last_prob) * stats["avg_loss"])
    sig, why = fuse_signal(
        float(last.z_score), last_prob, last_exp, bool(last_risk.risk_ok),
        params.entry_z, params.stop_z, threshold, str(last_risk.risk_reason),
    )
    latest = {
        "date": ds.index[-1],
        "price_a": float(last.price_a),
        "price_b": float(last.price_b),
        "spread": float(last.spread),
        "z_score": float(last.z_score),
        "z_reading": classify_z(float(last.z_score), params.entry_z, params.exit_z, params.stop_z),
        "probability": last_prob,
        "expected_net_return": last_exp,
        "signal": sig,
        "reason": why,
    }

    # --- persist everything ---
    trade_dates = ["decision_date", "entry_date", "exit_date"]
    val_trades = pd.read_csv(reports_dir / "validation_trades.csv", parse_dates=trade_dates)
    test_trades = pd.concat([test["baseline"]["trades"], test["hybrid"]["trades"]], ignore_index=True)
    pd.concat([val_trades, test_trades], ignore_index=True).drop(columns=["decision_pos"], errors="ignore").to_csv(
        reports_dir / "experiment_results.csv", index=False
    )
    val_equity = pd.read_csv(reports_dir / "validation_equity.csv", parse_dates=["date"], index_col="date")
    pd.concat([val_equity, test["baseline"]["equity"], test["hybrid"]["equity"]]).to_csv(reports_dir / "equity_curves.csv")
    failure_cases.drop(columns=["decision_pos"], errors="ignore").to_csv(reports_dir / "failure_cases.csv", index=False)

    out_ds = ds.copy()
    out_ds["split"] = split.labels()
    out_ds["probability"] = prob
    val_path = processed_dir / "validation_dataset.csv"
    if val_path.exists():  # out-of-sample validation probabilities from the stage-1 model
        val_prob = pd.read_csv(val_path, parse_dates=["date"], index_col="date")["probability"]
        out_ds.loc[val_prob.index, "probability"] = val_prob
    out_ds["baseline_signal"] = test["baseline_signal"]
    out_ds = out_ds.join(fused[["risk_ok", "risk_reason", "signal_name", "reason", "expected_net_return"]])
    out_ds.to_csv(processed_dir / "dataset.csv")

    metrics = {
        "generated_at": _now(),
        "mode": "paper trading only - simulated portfolio, no real orders",
        "data": {**bundle["data_meta"], "quality": bundle["data_quality"]},
        "pair": {
            "ticker_a": a,
            "ticker_b": b,
            "decision": bundle["pair_decision"],
            "screening": pd.read_csv(reports_dir / "pair_selection.csv").to_dict("records"),
        },
        "split": _split_summary(panel, split, params),
        "hedge": {
            "final": bundle["hedge"],
            "final_fit_scope": bundle["fit_scope"],
            "stage1_train_only": bundle["stage1"]["hedge"],
            "spread_half_life_sessions_train": bundle["stage1"]["half_life_train"],
        },
        "parameters": {
            "zscore_window": cfg["statistical"]["zscore_window"],
            "entry_z": params.entry_z,
            "exit_z": params.exit_z,
            "stop_z": params.stop_z,
            "horizon_days": params.horizon_days,
            "execution_lag": params.execution_lag,
            "round_trip_cost_fraction_of_gross": 2 * params.cost_per_side,
            "initial_capital": cfg["trading"]["initial_capital"],
        },
        "model": {
            "type": bundle["model_type"],
            "features": FEATURE_COLUMNS,
            "fit_scope": bundle["fit_scope"],
            "n_fit_samples": bundle["n_fit_samples"],
            "fit_base_rate": bundle["fit_base_rate"],
            "stage1_train_samples": bundle["stage1"]["n_train_samples"],
            "probability_threshold": threshold,
            "threshold_source": bundle["threshold_source"],
            "outcome_stats": stats,
            "risk_limits": bundle["risk_limits"],
            "feature_importance_validation": bundle["feature_importance"],
            "trained_at": bundle["trained_at"],
        },
        "validation": bundle["validation"],
        "test": {
            "classification_all_candidates": cls_all,
            "classification_entry_zone": cls_entry,
            "calibration": calibration,
            "baseline": test["baseline"]["metrics"],
            "hybrid": test["hybrid"]["metrics"],
            "error_analysis": error_summary,
        },
        "latest": latest,
        "caveats": [
            "Statistical arbitrage is not risk-free: related stocks can stay mispriced or diverge further.",
            "One pair, daily bars, a few dozen test trades: every number has wide uncertainty.",
            "Costs are a flat bps assumption; short-borrow fees, dividends on shorts and market impact are not modelled.",
            "Labels from consecutive days overlap, so samples are not independent.",
            "This is a research prototype for paper trading. It is not investment advice.",
        ],
    }
    _write_json(reports_dir / "metrics.json", metrics)

    if verbose:
        tb, th = metrics["test"]["baseline"], metrics["test"]["hybrid"]
        print(f"Test period     : {metrics['split']['test']['start'].date()} -> {metrics['split']['test']['end'].date()}")
        for name, mm in [("baseline", tb), ("hybrid", th)]:
            wr = f"{mm['win_rate']:.0%}" if mm["win_rate"] is not None else "n/a"
            print(
                f"  {name:<9}: net P&L {mm['net_pnl']:+,.0f} | trades {mm['n_trades']} | win rate {wr} | "
                f"Sharpe {mm['sharpe_ratio']:.2f} | max DD {mm['max_drawdown']:.1%}"
            )
        if cls_all.get("n_samples"):
            pr = cls_all["pr_auc"]
            print(
                f"  classifier: PR-AUC {pr:.3f} (no-skill {cls_all['pr_auc_no_skill']:.3f}) | "
                f"Brier {cls_all['brier_score']:.3f} (no-skill {cls_all['brier_no_skill']:.3f}) | n={cls_all['n_samples']}"
            )
        print(f"Latest signal   : {latest['signal']} on {latest['date'].date()} (z = {latest['z_score']:+.2f}) - {latest['reason']}")
        print(f"Reports written : {reports_dir}")
    return metrics


def run_all(refresh_data: bool = False, verbose: bool = True) -> dict:
    train(refresh_data=refresh_data, verbose=verbose)
    return evaluate(verbose=verbose)
