"""Market data layer: validation, alignment, chronological split and purging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arbitrage.backtester import TradeParams
from arbitrage.feature_engineering import build_dataset, candidate_mask
from arbitrage.preprocessing import build_panel, chronological_split
from arbitrage.statistical_engine import fit_hedge_ratio


def test_panel_keeps_only_dates_common_to_all_tickers(frames):
    frames["BBB"] = frames["BBB"].drop(frames["BBB"].index[[10, 11, 200]])
    panel, report = build_panel(frames)
    assert len(panel) == len(frames["AAA"]) - 3
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert report["aligned_rows"] == len(panel)


def test_invalid_rows_are_removed(frames):
    bad = frames["AAA"].index[[5, 6, 7]]
    frames["AAA"].loc[bad[0], "adj_close"] = np.nan
    frames["AAA"].loc[bad[1], "adj_close"] = -1.0
    frames["AAA"].loc[bad[2], "volume"] = 0.0
    panel, report = build_panel(frames)
    assert not panel.index.isin(bad).any()
    assert report["tickers"]["AAA"]["invalid_rows_removed"] == 3
    assert (panel.filter(like="adj_close") > 0).all().all()


def test_duplicate_dates_are_removed(frames):
    frames["AAA"] = pd.concat([frames["AAA"], frames["AAA"].iloc[[20]]])
    panel, report = build_panel(frames)
    assert panel.index.is_unique
    assert report["tickers"]["AAA"]["duplicate_dates_removed"] == 1


def test_suspicious_jump_is_flagged_not_hidden(frames):
    day = frames["AAA"].index[100]
    frames["AAA"].loc[day:, "adj_close"] *= 2.0  # a 100% one-day "move" is a data problem
    panel, report = build_panel(frames, max_abs_daily_return=0.25)
    assert not panel.loc[day, "quality_ok"]
    assert report["suspicious_moves_flagged"]["AAA"] == 1
    assert panel["quality_ok"].sum() == len(panel) - 1


def test_split_is_chronological_and_complete():
    split = chronological_split(1000, 0.70, 0.15)
    assert (split.train_end, split.val_end, split.n) == (700, 850, 1000)
    labels = split.labels()
    assert list(dict.fromkeys(labels)) == ["train", "validation", "test"]  # order never interleaves
    assert (labels == "train").sum() == 700 and (labels == "test").sum() == 150


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        chronological_split(100, 0.9, 0.2)


def test_purged_training_labels_never_reach_past_the_boundary(panel, cfg):
    """No training sample's outcome may depend on a price from the next period."""
    params = TradeParams.from_config(cfg)
    split = chronological_split(len(panel), 0.70, 0.15)
    train = panel.iloc[: split.train_end]
    hedge = fit_hedge_ratio(np.log(train["AAA_adj_close"]), np.log(train["BBB_adj_close"]), "AAA", "BBB")
    ds = build_dataset(panel, hedge, cfg)

    train_rows = candidate_mask(ds, cfg) & (ds["label_end_pos"] < split.train_end)
    assert train_rows.sum() > 50
    assert ds.loc[train_rows, "pos"].max() < split.train_end - params.label_span  # the embargo
    # The last `label_span` sessions of the period are labelled but must not be used for fitting.
    tail = ds.iloc[split.train_end - params.label_span : split.train_end]
    assert not train_rows.loc[tail.index].any()


def test_last_sessions_have_no_label(panel, cfg):
    params = TradeParams.from_config(cfg)
    hedge = fit_hedge_ratio(np.log(panel["AAA_adj_close"]), np.log(panel["BBB_adj_close"]), "AAA", "BBB")
    ds = build_dataset(panel, hedge, cfg)
    assert ds["label"].iloc[-params.label_span :].isna().all()  # their future does not exist yet
