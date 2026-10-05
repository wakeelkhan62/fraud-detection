import numpy as np
import pandas as pd

from fraud_detection.data import load_raw, time_split
from fraud_detection.features import FeatureBuilder, _as_key


def _fitted(raw_dir):
    train, valid, _ = time_split(load_raw(raw_dir), 0.15, 0.15)
    return FeatureBuilder().fit(train), train, valid


def test_same_columns_for_train_valid_and_single_request(raw_dir):
    fb, train, valid = _fitted(raw_dir)
    one_request = pd.DataFrame([{"TransactionAmt": 50.0, "TransactionDT": 90_000, "card1": 1010}])
    for X in (fb.transform(train), fb.transform(valid), fb.transform(one_request)):
        assert list(X.columns) == fb.feature_names_


def test_unseen_category_gets_minus_one(raw_dir):
    fb, _, _ = _fitted(raw_dir)
    req = pd.DataFrame([{"TransactionAmt": 10.0, "TransactionDT": 1, "ProductCD": "NEVER_SEEN"}])
    assert fb.transform(req)["ProductCD_code"].iloc[0] == -1


def test_numeric_keys_match_across_int_float_and_string():
    keys = _as_key(pd.Series([315, 315.0, "315", None], dtype=object))
    assert keys.iloc[0] == keys.iloc[1] == keys.iloc[2]
    assert keys.iloc[3] == "nan"


def test_statistics_come_from_train_only(raw_dir):
    """Leakage guard: a customer that only appears after the training period has no history."""
    fb, train, _ = _fitted(raw_dir)
    new_customer = pd.DataFrame([{"TransactionAmt": 99.0, "TransactionDT": 1,
                                  "card1": 999_999, "addr1": 1.0}])
    X = fb.transform(new_customer)
    assert X["uid_count"].iloc[0] == 0
    assert np.isnan(X["uid_amt_mean"].iloc[0])
