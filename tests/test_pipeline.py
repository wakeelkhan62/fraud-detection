"""End-to-end: train on synthetic data, then score through the real API."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from fraud_detection import serve


def test_training_writes_artifacts_and_report(trained):
    cfg, report = trained
    out = Path(cfg["artifacts_dir"])
    for name in ("model.joblib", "features.joblib", "metadata.json"):
        assert (out / name).exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert 0 < meta["threshold"] < 1
    m = report["test_metrics"]["lightgbm"]
    assert m["roc_auc"] > 0.6  # the synthetic data has a planted signal the model must find


def _client(cfg, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", cfg["artifacts_dir"])
    return TestClient(serve.app)


def test_api_scores_a_transaction(trained, monkeypatch):
    cfg, _ = trained
    with _client(cfg, monkeypatch) as client:
        assert client.get("/health").json()["status"] == "ok"
        resp = client.post("/score", json={"transaction": {
            "TransactionID": 1, "TransactionDT": 100_000, "TransactionAmt": 250.0,
            "ProductCD": "W", "card1": 1020, "addr1": 204.0, "P_emaildomain": "gmail.com"}})
        assert resp.status_code == 200
        body = resp.json()
        assert 0 <= body["fraud_probability"] <= 1
        assert isinstance(body["flagged"], bool)
        assert len(body["top_reasons"]) == 3


def test_api_rejects_missing_required_fields(trained, monkeypatch):
    cfg, _ = trained
    with _client(cfg, monkeypatch) as client:
        resp = client.post("/score", json={"transaction": {"card1": 1020}})
        assert resp.status_code == 422
