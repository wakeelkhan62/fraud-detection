"""Metrics and business-cost threshold selection.

Accuracy is useless here: with ~3.5% fraud, a model that says "never fraud" is 96.5% accurate.
We report PR-AUC (the honest ranking metric for rare events), ROC-AUC (the Kaggle metric),
precision/recall at the chosen threshold, and money: the cost of mistakes vs. doing nothing.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def business_cost(y_true, flagged, amount, review_cost: float) -> float:
    y_true, flagged, amount = map(np.asarray, (y_true, flagged, amount))
    missed_fraud = ((y_true == 1) & (~flagged)) * amount
    false_alarms = ((y_true == 0) & flagged) * review_cost
    return float(missed_fraud.sum() + false_alarms.sum())


def choose_threshold(y_true, proba, amount, review_cost: float, n_grid: int = 400) -> float:
    """Pick the threshold that minimises total business cost on the VALIDATION set."""
    proba = np.asarray(proba)
    grid = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, n_grid)))
    costs = [business_cost(y_true, proba >= t, amount, review_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def recall_at_precision(y_true, proba, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def compute_metrics(y_true, proba, amount, threshold: float, review_cost: float) -> dict:
    y_true, proba, amount = map(np.asarray, (y_true, proba, amount))
    flagged = proba >= threshold
    tp = int(((y_true == 1) & flagged).sum())
    fp = int(((y_true == 0) & flagged).sum())
    fn = int(((y_true == 1) & ~flagged).sum())
    cost_model = business_cost(y_true, flagged, amount, review_cost)
    cost_do_nothing = float((y_true * amount).sum())  # every fraud slips through
    return {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "recall_at_90_precision": recall_at_precision(y_true, proba, 0.90),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "flag_rate": float(flagged.mean()),
        "cost_with_model": cost_model,
        "cost_do_nothing": cost_do_nothing,
        "savings_pct": 100 * (1 - cost_model / cost_do_nothing) if cost_do_nothing else 0.0,
    }
