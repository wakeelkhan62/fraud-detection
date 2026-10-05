"""Stream simulator: replays the held-out TEST period, in time order, against the live API.

IEEE-CIS is a static dataset. Replaying it chronologically is how we test the system the way it
would run in production: one transaction at a time, measuring latency and live performance.

Run (with the API running):  python -m fraud.replay --limit 2000
"""
import argparse
import csv
import math
import time
from pathlib import Path

import httpx
import numpy as np

from .config import load_config
from .data import AMOUNT_COL, TARGET, load_raw, time_split
from .evaluate import compute_metrics


def _clean(record: dict) -> dict:
    """JSON has no NaN; send missing values as null."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}


def replay(cfg: dict, url: str, limit: int | None, speedup: float | None, out_path: Path) -> dict:
    df = load_raw(cfg["data"]["raw_dir"], nrows=cfg["data"].get("nrows"))
    _, _, test = time_split(df, cfg["split"]["valid_frac"], cfg["split"]["test_frac"])
    if limit:
        test = test.head(limit)

    rows, latencies = [], []
    prev_dt = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=url, timeout=10) as client:
        for i, rec in enumerate(test.to_dict(orient="records")):
            if speedup and prev_dt is not None:  # optionally respect real gaps between transactions
                time.sleep(max(0.0, (rec["TransactionDT"] - prev_dt) / speedup))
            prev_dt = rec["TransactionDT"]
            label = rec.pop(TARGET)
            t0 = time.perf_counter()
            resp = client.post("/score", json={"transaction": _clean(rec)})
            resp.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
            body = resp.json()
            rows.append({"TransactionID": rec["TransactionID"], "TransactionDT": rec["TransactionDT"],
                         "amount": rec[AMOUNT_COL], "label": label,
                         "proba": body["fraud_probability"], "flagged": body["flagged"]})
            if (i + 1) % 500 == 0:
                print(f"{i + 1:>7} scored | p50 {np.percentile(latencies, 50):.1f} ms"
                      f" | p95 {np.percentile(latencies, 95):.1f} ms")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = np.array([r["label"] for r in rows])
    summary = {"scored": len(rows),
               "latency_p50_ms": float(np.percentile(latencies, 50)),
               "latency_p95_ms": float(np.percentile(latencies, 95)),
               "latency_p99_ms": float(np.percentile(latencies, 99))}
    if 0 < labels.sum() < len(labels):
        m = compute_metrics(labels, [r["proba"] for r in rows], [r["amount"] for r in rows],
                            threshold=_served_threshold(url), review_cost=cfg["cost"]["review_cost"])
        summary.update({k: m[k] for k in ("pr_auc", "precision", "recall", "savings_pct")})
    print("\nREPLAY SUMMARY")
    for k, v in summary.items():
        print(f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
    print(f"  log written to {out_path}")
    return summary


def _served_threshold(url: str) -> float:
    """Ask the API which threshold it uses, so the summary matches what was actually served."""
    with httpx.Client(base_url=url, timeout=10) as client:
        probe = client.post("/score", json={"transaction": {"TransactionAmt": 1.0, "TransactionDT": 0}})
        return probe.json()["threshold"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--speedup", type=float, default=None,
                   help="e.g. 3600 = replay one hour of real time per second; omit for max speed")
    p.add_argument("--out", default="logs/replay_log.csv")
    a = p.parse_args()
    replay(load_config(a.config), a.url, a.limit, a.speedup, Path(a.out))


if __name__ == "__main__":
    main()
