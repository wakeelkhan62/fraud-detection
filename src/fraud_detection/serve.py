"""Scoring API.

Run:  uvicorn fraud.serve:app --port 8000
POST /score  {"transaction": {...raw IEEE-CIS fields...}}
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STATE: dict[str, Any] = {}


def load_artifacts(artifacts_dir: str | Path) -> None:
    artifacts_dir = Path(artifacts_dir)
    STATE["model"] = joblib.load(artifacts_dir / "model.joblib")
    STATE["features"] = joblib.load(artifacts_dir / "features.joblib")
    STATE["meta"] = json.loads((artifacts_dir / "metadata.json").read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts(os.getenv("ARTIFACTS_DIR", "artifacts"))
    yield
    STATE.clear()


app = FastAPI(title="Fraud Scoring API", version="0.1.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    transaction: dict[str, Any] = Field(..., description="Raw transaction fields, e.g. TransactionAmt, card1")


class Reason(BaseModel):
    feature: str
    contribution: float  # positive pushes towards fraud (log-odds units)


class ScoreResponse(BaseModel):
    transaction_id: Any = None
    fraud_probability: float
    flagged: bool
    threshold: float
    top_reasons: list[Reason]
    model_version: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": STATE["meta"]["model_version"]}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    tx = req.transaction
    if "TransactionAmt" not in tx or "TransactionDT" not in tx:
        raise HTTPException(422, "transaction must include TransactionAmt and TransactionDT")

    X = STATE["features"].transform(pd.DataFrame([tx]))
    model = STATE["model"]
    proba = float(model.predict_proba(X)[0, 1])

    # Per-prediction explanation: LightGBM's built-in SHAP-style contributions (fast, no extra library).
    contrib = model.booster_.predict(X, pred_contrib=True)[0][:-1]  # last value is the bias term
    top = np.argsort(-np.abs(contrib))[:3]
    reasons = [Reason(feature=X.columns[i], contribution=round(float(contrib[i]), 4)) for i in top]

    threshold = STATE["meta"]["threshold"]
    return ScoreResponse(
        transaction_id=tx.get("TransactionID"),
        fraud_probability=round(proba, 6),
        flagged=proba >= threshold,
        threshold=threshold,
        top_reasons=reasons,
        model_version=STATE["meta"]["model_version"],
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
