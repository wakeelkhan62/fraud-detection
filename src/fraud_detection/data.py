"""Loading the raw IEEE-CIS files and splitting them by time."""
from pathlib import Path

import pandas as pd

TARGET = "isFraud"           # label: 1 = fraud, 0 = legitimate
TIME_COL = "TransactionDT"   # seconds from a hidden reference point; real dates are masked
ID_COL = "TransactionID"     # key that joins the transaction and identity files
AMOUNT_COL = "TransactionAmt"


def reduce_memory(df: pd.DataFrame, max_categories: int = 1000) -> pd.DataFrame:
    """Shrink the DataFrame in RAM without changing what the data means.

    - float64 -> float32: halves the memory per number; enough precision for ML features.
    - int64 -> the smallest integer type that fits the values (int8/int16/int32).
    - low-cardinality text columns -> 'category' (each distinct string is stored only once).
    """
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            # High-cardinality text (e.g. DeviceInfo) gains little from category, so skip it.
            if s.nunique(dropna=True) < max_categories:
                df[col] = s.astype("category")
    return df


def load_raw(raw_dir: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load train_transaction.csv, left-join train_identity.csv, then reduce memory.

    nrows: read only the first N transactions; useful for quick experiments.
    """
    raw_dir = Path(raw_dir)
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Download the IEEE-CIS data from Kaggle into {raw_dir}/"
        )

    df = pd.read_csv(tx_path, nrows=nrows)

    if id_path.exists():
        identity = pd.read_csv(id_path)
        n_before = len(df)
        # Left join: every transaction is kept; rows without identity get NaN.
        # validate="one_to_one" raises an error if TransactionID is duplicated in either file.
        # Without it, duplicates silently multiply rows and the model trains on corrupted data.
        df = df.merge(identity, on=ID_COL, how="left", validate="one_to_one")
        assert len(df) == n_before, "Join changed the number of rows; check the key column."

    return reduce_memory(df)


def time_split(df: pd.DataFrame, valid_frac: float = 0.15, test_frac: float = 0.15):
    """Sort by time, then split oldest -> train, middle -> valid, newest -> test.

    A random split lets the model see examples from the 'future' and inflates the score.
    In production a model always learns from the past and predicts on newer data.
    """
    if not 0 < valid_frac + test_frac < 1:
        raise ValueError("valid_frac + test_frac must be between 0 and 1")

    # kind="stable" keeps rows with equal TransactionDT in their original order,
    # so the split is identical every time it runs (reproducibility).
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)

    n = len(df)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    return train, valid, test
