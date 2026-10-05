# Real-Time Fraud Detection System

![CI](https://github.com/wakeelkhan62/fraud-detection/actions/workflows/ci.yml/badge.svg)

A production-style fraud detection service built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590,540 real e-commerce transactions, 3.5% fraud).

It goes beyond a notebook: data validation, leakage-safe features, time-based evaluation, a decision threshold chosen by **business cost**, a low-latency scoring API with per-prediction explanations, and a stream simulator that replays transactions one by one, the way a live system receives them.

## Architecture

```
 raw CSVs ──► load + join + memory optimisation ──► data validation (refuse bad data)
                                                          │
            time-based split: train (oldest 70%) │ valid (15%) │ test (newest 15%)
                                                          │
                    FeatureBuilder.fit(train only)  ──► transform(train / valid / test / live)
                                                          │
            baseline (logistic regression)  vs  LightGBM (early stopping on valid)
                                                          │
             threshold = argmin business cost on valid  ──► final report on test (touched once)
                                                          │
                     artifacts/ (model, features, metadata with version + metrics)
                                                          │
   replay.py (test period, in time order) ──HTTP──► FastAPI /score ──► probability, flag, top 3 reasons
```

## Results

Test set = the newest 15% of transactions (88,581), never used for training or tuning.

| Metric | Logistic regression baseline | LightGBM |
|---|---|---|
| ROC-AUC | 0.830 | **0.911** |
| PR-AUC | 0.177 | **0.557** |
| Recall at 90% precision | 0.000 | **0.247** |
| Precision / recall at chosen threshold | 0.104 / 0.755 | 0.164 / 0.817 |
| Flag rate | 25.4% | 17.3% |
| Cost saved vs. no model | 52.5% | **70.3%** |

**What these numbers mean**

- With 3.5% fraud, a random guess has PR-AUC of about 0.035. LightGBM reaches 0.557, roughly 16x better than random and 3x better than the baseline.
- At a strict 90% precision (useful for automatic blocking), the model still catches about a quarter of all fraud.
- **Known weakness:** the cost-based threshold flags 17% of transactions. That is far more than a fraud team can review manually. It happens because the cost model treats a review as very cheap ($5). The next step is a review-capacity constraint (see Roadmap).

## Key design decisions

| Decision | Why |
|---|---|
| **Time-based split, not random** | Fraud patterns drift. A random split lets the model learn from the future and inflates scores. |
| **Three sets: train / valid / test** | Valid is used for early stopping and the threshold; test is used once, for an honest report. |
| **Statistics learned from train only** | Category codes, frequencies and per-customer averages computed on all data would leak future information. |
| **PR-AUC and money, not accuracy** | With 3.5% fraud, "never fraud" is 96.5% accurate and catches nothing. |
| **Threshold by business cost** | A missed fraud costs the transaction amount; a false alarm costs a manual review (configurable in `configs/config.yaml`). |
| **`validate="one_to_one"` on the join** | Duplicate IDs would silently multiply rows and can leak across the split boundary. |
| **Memory optimisation** | float64 to float32 and text to category cut RAM use substantially, so training fits on a laptop. |
| **Per-prediction explanations** | LightGBM contribution values give the top 3 reasons for every flag, which fraud analysts and compliance need. |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
# 1. Install dependencies exactly as locked
uv sync

# 2. Get the data: join the Kaggle competition, download train_transaction.csv and
#    train_identity.csv into data/raw/  (data is not in this repo, per Kaggle's rules)

# 3. Run the tests (they use synthetic data, no download needed)
uv run pytest

# 4. Train: validates data, trains baseline + LightGBM, picks the threshold, writes artifacts/
uv run fraud-train

# 5. Serve the model, then open http://localhost:8000/docs to try /score in the browser
uv run uvicorn fraud_detection.serve:app --port 8000

# 6. In a second terminal, replay the test period against the live API
uv run fraud-replay --limit 2000
```

For a quick experiment on a subset, set `nrows: 100000` in `configs/config.yaml`.

### Example request

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
  -d '{"transaction": {"TransactionID": 1, "TransactionDT": 15000000, "TransactionAmt": 950.0,
       "ProductCD": "C", "card1": 13926, "addr1": 315.0, "P_emaildomain": "anonymous.com"}}'
```

Response: fraud probability, whether it is flagged, the threshold, the top 3 reasons, the model version and server-side latency.

## Project structure

```
configs/config.yaml          all settings (split sizes, model, business cost)
src/fraud_detection/
  data.py                    load, join, memory optimisation, time split
  validation.py              data quality checks before training
  features.py                leakage-safe feature engineering
  evaluate.py                metrics and cost-based threshold
  train.py                   training pipeline and report
  serve.py                   FastAPI scoring service
  replay.py                  stream simulator against the live API
tests/                       unit + end-to-end tests on synthetic data
.github/workflows/ci.yml     runs the tests on every push
```

## Limitations (honest)

- The dataset is static; "real-time" is simulated by replaying the newest period in time order.
- There is no true customer ID; `card1 + addr1` is a rough proxy.
- Most columns (V1–V339, id_*) are anonymised, so features cannot be interpreted in business terms.
- The cost model (missed fraud = amount, false alarm = fixed review cost) is a simplification.

## Roadmap

- [ ] Review-capacity constraint (e.g. flag at most 2% of transactions) with tiered actions: block / step-up verification / allow
- [ ] Experiment tracking with MLflow
- [ ] Drift monitoring on the replay log (feature and score distributions over time)
- [ ] Champion / challenger retraining
- [ ] Containerise with Docker and deploy the API to Azure Container Apps
- [ ] LLM analyst agent that turns each flag's top reasons into a readable case report