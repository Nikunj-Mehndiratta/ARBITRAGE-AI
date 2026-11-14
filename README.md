# ARBITRAGE AI

**Hybrid statistical arbitrage intelligence — a research prototype. Paper trading only.**

ARBITRAGE AI looks for mispricing between two related stocks, measures it mathematically,
and then asks a machine-learning model how likely that mispricing is to turn into a
profitable mean-reversion trade after costs.

> This is *statistical* arbitrage, not risk-free arbitrage. Related stocks can stay
> mispriced, diverge further, or stop being related. Nothing here places a real order,
> and nothing here is investment advice.

## The idea in two questions

1. **Mathematical engine** — is the relationship between the two stocks unusually far from
   where it has recently been? (hedge ratio → spread → rolling z-score)
2. **AI engine** — given the z-score, volatility, momentum and market conditions, how likely
   is fading that deviation to make money over the next five sessions, net of costs?

The first engine *detects* a candidate. The second *evaluates* it. A trade is simulated
only when both agree and the risk filters pass.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

```bash
streamlit run app.py
```

`run_pipeline.py` downloads daily bars from Yahoo Finance (cached in `data/raw/`), trains,
and evaluates. To run the two halves separately:

```bash
python -m models.train
```

```bash
python -m models.evaluate
```

```bash
python -m pytest tests -q
```

Add `--refresh-data` to `run_pipeline.py` or `models.train` to re-download prices.

## Results of the included run

Pair **KO / PEP**, daily adjusted closes 2010-01-04 → 2026-11-13 (4,243 aligned sessions,
retrieved 2026-11-14 UTC). Final test period **2024-05-02 → 2026-11-13** (637 sessions),
never used for fitting or tuning. Starting capital $100,000, 6 bps round-trip cost on gross
notional, fills one session after the signal.

| Test period | Baseline (rules only) | Hybrid (rules + ML + risk) |
|---|---|---|
| Net cumulative P&L | −$2,907 | +$3,229 |
| Trades | 28 | 10 |
| Win rate | 57% | 60% |
| Sharpe ratio (annualised) | −0.26 | 0.53 |
| Maximum drawdown | −6.3% | −2.8% |
| Average net return per trade | −0.09% | +0.33% |
| 95% bootstrap interval, average trade | −0.73% to +0.43% | −0.47% to +1.15% |

Classifier on the test period (430 candidate days with 1 ≤ |z| < 3.5):

| Metric | Model | No-skill reference |
|---|---|---|
| PR-AUC | 0.460 | 0.414 |
| ROC-AUC | 0.535 | 0.500 |
| Brier score (lower is better) | 0.2468 | 0.2426 |
| Precision at the 0.60 threshold | 41.0% | 41.4% (base rate) |

### What these numbers do and do not show

- **The hybrid beat the baseline in this test period, but the evidence is weak.** It rests on
  10 trades; the bootstrap interval for the average trade comfortably includes zero.
- **The classifier has little measurable skill.** PR-AUC is only slightly above the no-skill
  level, the Brier score is slightly *worse* than always predicting the base rate, and its
  probabilities are poorly calibrated (predictions near 0.65 won about 41% of the time across
  all candidate days).
- **Most of the hybrid's improvement comes from trading less.** On entry-zone days the
  signals it accepted won 61% of the time against 60% for the ones it rejected. It did avoid
  the baseline's worst trade (−5.6%), and its accepted signals had a higher average return
  (0.37% vs 0.09%), but with 18 accepted signal-days that difference is not reliable.
- **The pair is not statistically confirmed as cointegrated.** Engle–Granger p = 0.17 on the
  training period (XOM/CVX was worse at 0.75), the hedge ratio changes a lot between
  sub-windows, and the spread drifted steadily through the test period. The pipeline says so
  up front instead of picking a friendlier window.
- **Failures are reported.** `reports/failure_cases.csv` lists every losing trade with the
  model's probability at the time; the hybrid's four losers all had probabilities of 0.60–0.70.

A one-page write-up is in [`reports/ARBITRAGE_AI_paper.pdf`](reports/ARBITRAGE_AI_paper.pdf) (regenerate it from the saved results with `python reports/build_paper.py`).

Full details: `reports/metrics.json`, `reports/experiment_results.csv` (every simulated
trade), `reports/equity_curves.csv`, and the dashboard's *Error analysis* tab.

## How it works

### 1. Market data layer — `arbitrage/data_loader.py`, `preprocessing.py`
Daily bars from Yahoo Finance via `yfinance`. Adjusted closes, dates aligned across all
tickers, invalid rows removed, suspicious one-day moves flagged. `data/raw/metadata.json`
records the source and retrieval time.

### 2. Statistical engine — `statistical_engine.py`, `pair_selection.py`

```
P_A = ln(A)      P_B = ln(B)
P_B = α + β·P_A + ε            (OLS, fit window only)
S_t = P_B − α − β·P_A          (the spread)
Z_t = (S_t − μ_t) / σ_t        (rolling 60-session mean and standard deviation)
```

| Condition | Reading |
|---|---|
| Z > 2 | spread unusually high → candidate short-spread |
| Z < −2 | spread unusually low → candidate long-spread |
| \|Z\| < 0.5 | back near the mean → exit |
| \|Z\| > 3.5 | stop / risk zone |

Pairs are screened on the training period only: return correlation, liquidity,
Engle–Granger cointegration, and a stability check over sub-windows.

### 3. ML prediction engine — `feature_engineering.py`, `prediction_engine.py`
A `HistGradientBoostingClassifier` (small and heavily regularised) estimates
`p = P(trade is profitable after costs | features)`. Features: z-score and its changes,
spread volatility, rolling correlation, rolling hedge ratio, RSI of each stock, relative
volume, market volatility and spread momentum. The label for day *t* is the outcome of the
hypothetical trade decided at *t*, simulated with the same fills, exits and costs as the
backtest. `random_forest` and `logistic_regression` are available via `config.yaml`.

### 4. Signal fusion and risk — `signal_fusion.py`, `risk_manager.py`

```python
if entry_z <= abs(z) < stop_z and risk_ok and p >= threshold and expected_net_return > 0:
    signal = "LONG_SPREAD" if z < 0 else "SHORT_SPREAD"
else:
    signal = "NO_TRADE"
```

Risk filters reject signals when market or spread volatility is above its training-period
95th percentile, when the pair's rolling correlation breaks down, or when data is flagged.

### 5. Backtesting and simulation — `backtester.py`
Both legs are simulated: long spread = long B, short β·A; short spread is the reverse.
Costs and slippage are charged per leg, per side. Exits: spread reverts, stop is breached,
or five sessions pass. One position at a time, marked to market daily. The baseline and the
hybrid share everything except the ML and risk filters, so the comparison isolates their effect.

### 6. Dashboard — `app.py`, `dashboard/`
Spread and z-score history, price comparison, model probability over time, baseline vs
hybrid backtest, equity and drawdown, error analysis, and the protocol and data provenance.

## Anti-leakage protocol

| Period | Dates in this run | Used for |
|---|---|---|
| Training (70%) | 2010-01-04 → 2021-10-19 | pair selection, hedge ratio, model, risk limits |
| Validation (15%) | 2021-10-20 → 2024-05-01 | probability threshold; then merged into the final refit |
| Final test (15%) | 2024-05-02 → 2026-11-13 | evaluated once |

1. Time-series rows are never shuffled.
2. Imputation and scaling live inside a scikit-learn `Pipeline` fitted on fitting rows only.
3. The hedge ratio and pair choice never see test data.
4. A sample is used for fitting only if its forward-looking label (1 lag + 5 holding
   sessions) is fully resolved before the split boundary — a 6-session embargo.
5. The probability threshold is tuned on validation only (0.60 was selected there).
6. Losing trades and model errors are written to the reports.

`tests/` checks the pieces this depends on: rewriting future prices does not change past
z-scores or features, purged training labels never reach past a boundary, risk limits ignore
future rows, and the trade simulator's P&L matches hand calculations.

## Repository layout

```
ARBITRAGE-AI/
├── app.py                  Streamlit dashboard
├── run_pipeline.py         train + evaluate in one command
├── config.yaml             every experimental parameter
├── requirements.txt
├── arbitrage/
│   ├── data_loader.py      download + cache
│   ├── preprocessing.py    validation, alignment, chronological split
│   ├── pair_selection.py   correlation, cointegration, stability
│   ├── statistical_engine.py  hedge ratio, spread, z-score
│   ├── feature_engineering.py features + labels
│   ├── prediction_engine.py   model, metrics, calibration
│   ├── signal_fusion.py    baseline and hybrid decisions
│   ├── risk_manager.py     volatility / correlation / data filters
│   ├── backtester.py       trade simulator + sequential backtest
│   └── pipeline.py         train() and evaluate() orchestration
├── models/                 train.py, evaluate.py, saved_model/
├── dashboard/              charts.py, components.py
├── tests/                  31 unit tests
├── data/                   raw/ (cached bars + metadata), processed/
└── reports/                metrics.json, experiment_results.csv, …
```

## Limitations

- One pair and a few dozen test trades: every figure has wide uncertainty.
- Costs are a flat basis-point assumption. Short-borrow fees, dividends owed on short
  positions, and market impact are not modelled.
- Labels from consecutive days overlap, so training samples are not independent.
- The threshold was chosen from five values on a validation period that produced at most
  12 trades for any of them, so that choice is itself noisy.
- Yahoo Finance data is a research dataset, not an institutional feed.

## Troubleshooting

On Windows, if importing scikit-learn fails with *"An Application Control policy has blocked
this file"*, Smart App Control is refusing one of that release's binaries. Installing a
release it already trusts resolved it on the development machine:

```bash
pip install scikit-learn==1.8.0
```
