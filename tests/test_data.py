import pandas as pd
import pytest

from fraud_detection.data import TARGET, TIME_COL, load_raw, reduce_memory, time_split
from fraud_detection.validation import DataValidationError, assert_valid, validate_training_data


def test_load_raw_keeps_every_transaction(raw_dir):
    df = load_raw(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    assert len(df) == len(tx)          # left join must not add or drop rows
    assert "DeviceType" in df.columns  # identity columns were joined


def test_duplicate_identity_ids_are_rejected(raw_dir, tmp_path):
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    identity = pd.concat([identity, identity.head(1)])  # plant a duplicate TransactionID
    tx.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    with pytest.raises(pd.errors.MergeError):
        load_raw(tmp_path)


def test_reduce_memory_shrinks_and_keeps_values():
    df = pd.DataFrame({"f": [1.5, 2.25, None], "i": [1, 2, 3], "s": ["a", "b", "a"]})
    before = df.memory_usage(deep=True).sum()
    out = reduce_memory(df.copy())
    assert out["f"].dtype == "float32"
    assert out["i"].dtype == "int8"
    assert isinstance(out["s"].dtype, pd.CategoricalDtype)
    assert out.memory_usage(deep=True).sum() < before
    assert out["f"].iloc[1] == pytest.approx(2.25)


def test_time_split_has_no_overlap_and_no_lost_rows(raw_dir):
    df = load_raw(raw_dir)
    train, valid, test = time_split(df, 0.15, 0.15)
    assert len(train) + len(valid) + len(test) == len(df)
    assert train[TIME_COL].max() <= valid[TIME_COL].min()
    assert valid[TIME_COL].max() <= test[TIME_COL].min()


def test_validation_catches_bad_data(raw_dir):
    df = load_raw(raw_dir)
    assert validate_training_data(df) == []
    broken = df.copy()
    broken.loc[broken.index[0], "TransactionAmt"] = -10
    broken.loc[broken.index[1], TARGET] = 7
    issues = validate_training_data(broken)
    assert any("TransactionAmt" in i for i in issues)
    assert any(TARGET in i for i in issues)
    with pytest.raises(DataValidationError):
        assert_valid(broken)
