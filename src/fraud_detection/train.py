"""Training pipeline: validate -> split by time -> features -> baseline vs LightGBM -> threshold -> report.

Run:  python -m fraud.train --config configs/config.yaml
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import load_config
from .data import AMOUNT_COL, TARGET, TIME_COL, load_raw, time_split
from .evaluate import choose_threshold, compute_metrics
from .features import FeatureBuilder
from .validation import assert_valid


def run(cfg: dict, min_rows: int = 100) -> dict:
    t0 = time.time()
    df = load_raw(cfg["data"]["raw_dir"], nrows=cfg["data"].get("nrows"))
    assert_valid(df, min_rows=min_rows)
    train, valid, test = time_split(df, cfg["split"]["valid_frac"], cfg["split"]["test_frac"])
    print(f"rows  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    fb = FeatureBuilder().fit(train)
    X_tr, X_va, X_te = fb.transform(train), fb.transform(valid), fb.transform(test)
    y_tr, y_va, y_te = train[TARGET].values, valid[TARGET].values, test[TARGET].values
    review_cost = cfg["cost"]["review_cost"]
    print(f"features: {len(fb.feature_names_)}")

    # --- Baseline: always beat something simple before trusting something complex.
    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=1000))
    baseline.fit(X_tr, y_tr)
    b_va, b_te = baseline.predict_proba(X_va)[:, 1], baseline.predict_proba(X_te)[:, 1]
    b_thr = choose_threshold(y_va, b_va, valid[AMOUNT_COL], review_cost)
    baseline_metrics = compute_metrics(y_te, b_te, test[AMOUNT_COL], b_thr, review_cost)

    # --- Main model. Early stopping uses VALID; TEST stays untouched until the final report.
    model = lgb.LGBMClassifier(**cfg["model"]["lgbm"])
    model.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,), eval_metric="average_precision",
              callbacks=[lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False)])
    p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
    threshold = choose_threshold(y_va, p_va, valid[AMOUNT_COL], review_cost)
    model_metrics = compute_metrics(y_te, p_te, test[AMOUNT_COL], threshold, review_cost)

    out = Path(cfg["artifacts_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model_version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    joblib.dump(model, out / "model.joblib")
    joblib.dump(fb, out / "features.joblib")
    report = {
        "model_version": model_version,
        "code_version": __version__,
        "threshold": threshold,
        "review_cost": review_cost,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "n_features": len(fb.feature_names_),
        "periods": {name: [float(d[TIME_COL].min()), float(d[TIME_COL].max())]
                    for name, d in [("train", train), ("valid", valid), ("test", test)]},
        "test_metrics": {"baseline_logreg": baseline_metrics, "lightgbm": model_metrics},
        "train_seconds": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(report, indent=2))
    _log_mlflow(cfg, report)
    _print_report(report)
    return report


def _log_mlflow(cfg: dict, report: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return  # optional: pip install mlflow to track experiments
    with mlflow.start_run(run_name=report["model_version"]):
        mlflow.log_params({f"lgbm_{k}": v for k, v in cfg["model"]["lgbm"].items()})
        mlflow.log_metrics({k: v for k, v in report["test_metrics"]["lightgbm"].items()
                            if isinstance(v, (int, float))})
        mlflow.log_artifacts(cfg["artifacts_dir"])


def _print_report(report: dict) -> None:
    rows = ["roc_auc", "pr_auc", "recall_at_90_precision", "precision", "recall", "flag_rate", "savings_pct"]
    m = report["test_metrics"]
    print(f"\nTEST SET (newest {m['lightgbm']['n']:,} transactions, never used for tuning)")
    print(f"{'metric':<24}{'baseline':>12}{'lightgbm':>12}")
    for r in rows:
        print(f"{r:<24}{m['baseline_logreg'][r]:>12.4f}{m['lightgbm'][r]:>12.4f}")
    print(f"\nthreshold={report['threshold']:.4f}  saved to artifacts/ as version {report['model_version']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
