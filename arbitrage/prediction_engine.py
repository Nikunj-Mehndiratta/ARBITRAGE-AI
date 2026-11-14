"""ML prediction engine: p_t = P(hypothetical trade is profitable after costs | X_t).

The classifier is wrapped in a scikit-learn Pipeline so that every fitted transformation
(imputation, scaling) is learned from the training rows only and then merely applied to
validation and test rows.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_type: str, random_state: int = 42) -> Pipeline:
    """A small, heavily regularised model: the dataset is a few thousand tabular rows."""
    if model_type == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.03,
            max_iter=150,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
        steps = [("model", clf)]  # handles missing values natively
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=30,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        steps = [("impute", SimpleImputer(strategy="median")), ("model", clf)]
    elif model_type == "logistic_regression":
        clf = LogisticRegression(C=0.5, max_iter=2000)
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", clf),
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return Pipeline(steps)


def fit_model(x_train: pd.DataFrame, y_train: pd.Series, model_type: str, random_state: int = 42) -> Pipeline:
    if y_train.nunique() < 2:
        raise ValueError("Training labels contain a single class; cannot fit a classifier")
    model = build_model(model_type, random_state)
    model.fit(x_train, y_train.astype(int))
    return model


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Estimated probability that the hypothetical trade is profitable. Not a guarantee."""
    if len(x) == 0:
        return np.array([])
    return model.predict_proba(x)[:, 1]


def classification_metrics(y_true, prob, threshold: float) -> dict:
    """Classification quality, kept separate from trading results.

    Includes the no-skill references needed to read the numbers honestly: a model that
    always predicts the base rate has PR-AUC = base rate and Brier = base * (1 - base).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    n = int(len(y))
    if n == 0:
        return {"n_samples": 0}

    base = float(y.mean())
    pred = (p >= threshold).astype(int)
    both_classes = 0 < y.sum() < n
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "n_samples": n,
        "base_rate": base,
        "threshold": float(threshold),
        "predicted_positive": int(pred.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.sum() else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else None,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else None,
        "pr_auc_no_skill": base,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "brier_score": float(brier_score_loss(y, p)),
        "brier_no_skill": float(base * (1.0 - base)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_table(y_true, prob, bins: int = 5) -> list[dict]:
    """Mean predicted probability vs observed win rate, in equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.sum():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_win_rate": float(y[mask].mean()),
                }
            )
    return rows


def feature_importance(model: Pipeline, x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> list[dict]:
    """Permutation importance (drop in average precision) on held-out rows."""
    if len(x) < 20 or y.nunique() < 2:
        return []
    result = permutation_importance(
        model, x, y.astype(int), scoring="average_precision", n_repeats=20, random_state=random_state
    )
    rows = [
        {"feature": f, "importance": float(m), "std": float(s)}
        for f, m, s in zip(x.columns, result.importances_mean, result.importances_std)
    ]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def save_bundle(bundle: dict, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model_bundle.joblib"
    joblib.dump(bundle, path)
    return path


def load_bundle(model_dir: Path) -> dict:
    path = model_dir / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m models.train` first.")
    return joblib.load(path)
