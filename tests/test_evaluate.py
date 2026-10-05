import numpy as np

from fraud_detection.evaluate import business_cost, choose_threshold, compute_metrics


def test_business_cost_counts_missed_fraud_and_false_alarms():
    y = np.array([1, 1, 0, 0])
    flagged = np.array([True, False, True, False])
    amount = np.array([100.0, 40.0, 10.0, 10.0])
    # one missed fraud ($40) + one false alarm (review cost $5)
    assert business_cost(y, flagged, amount, review_cost=5.0) == 45.0


def test_threshold_beats_doing_nothing():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    proba = np.clip(0.6 * y + rng.normal(0.2, 0.15, 2000), 0, 1)
    amount = rng.lognormal(4, 1, 2000)
    t = choose_threshold(y, proba, amount, review_cost=5.0)
    m = compute_metrics(y, proba, amount, t, review_cost=5.0)
    assert m["cost_with_model"] < m["cost_do_nothing"]
    assert 0 < m["recall"] <= 1
