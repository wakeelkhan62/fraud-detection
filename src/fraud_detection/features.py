"""Leakage-safe feature engineering.

Rule: every statistic (category codes, frequencies, per-user averages) is learned ONLY from the
training period in `fit`, then applied unchanged to validation, test and live traffic in `transform`.
Computing them on the full dataset would leak future information into training.
"""
import numpy as np
import pandas as pd

from .data import AMOUNT_COL, ID_COL, TARGET, TIME_COL

# Columns used to build frequency features (how common is this card / address / email domain?).
FREQUENCY_COLS = ["card1", "card2", "addr1", "P_emaildomain"]
EXCLUDE = {ID_COL, TARGET, TIME_COL}


def _to_str(s: pd.Series) -> pd.Series:
    """String keys with missing values as the explicit key "nan" (same result on pandas 2 and 3)."""
    return s.astype(str).fillna("nan")


def _as_key(s: pd.Series) -> pd.Series:
    """Turn a column into stable string keys for lookups.

    Numbers are normalised first, so 315, 315.0 and "315" from a live JSON request all map to
    the same key as the value seen in training. Missing values become the key "nan".
    """
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() == s.notna().sum():  # every non-missing value is a number
        return _to_str(numeric.astype("float64"))
    return _to_str(s)


def _uid(df: pd.DataFrame) -> pd.Series:
    """Rough 'customer' key. IEEE-CIS has no customer ID; card1 + addr1 is a common proxy."""
    card1 = _as_key(df["card1"]) if "card1" in df else pd.Series("nan", index=df.index)
    addr1 = _as_key(df["addr1"]) if "addr1" in df else pd.Series("nan", index=df.index)
    return card1 + "_" + addr1


class FeatureBuilder:
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.raw_columns_ = [c for c in df.columns if c != TARGET]
        features = [c for c in self.raw_columns_ if c not in EXCLUDE]
        self.numeric_cols_ = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols_ = [c for c in features if c not in self.numeric_cols_]

        # Category -> integer code, ordered by frequency in TRAIN. Unseen values later get -1.
        self.cat_maps_ = {}
        for col in self.categorical_cols_:
            values = _to_str(df[col]).value_counts().index
            self.cat_maps_[col] = {v: i for i, v in enumerate(values)}

        self.freq_maps_ = {
            col: _as_key(df[col]).value_counts(normalize=True).to_dict()
            for col in FREQUENCY_COLS if col in df
        }

        # Per-'customer' spending profile, learned from TRAIN only.
        uid = _uid(df)
        amt = pd.to_numeric(df[AMOUNT_COL]).astype("float64")
        self.uid_amt_mean_ = amt.groupby(uid).mean().to_dict()
        self.uid_amt_std_ = amt.groupby(uid).std().fillna(0).to_dict()
        self.uid_count_ = uid.value_counts().to_dict()
        self.global_amt_mean_ = float(amt.mean())

        self.feature_names_ = list(self.transform(df.head(2)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Live requests may omit fields; add them as missing so every row has the same shape.
        df = df.reindex(columns=self.raw_columns_)
        cols: dict[str, pd.Series] = {}

        for col in self.numeric_cols_:
            cols[col] = pd.to_numeric(df[col], errors="coerce")

        # Time features. TransactionDT is in seconds, so these are relative hour and weekday.
        dt = pd.to_numeric(df[TIME_COL], errors="coerce")
        cols["hour"] = (dt // 3600) % 24
        cols["day_of_week"] = (dt // 86400) % 7

        # Amount features. Fraudsters often test cards with odd or very round amounts.
        amt = pd.to_numeric(df[AMOUNT_COL], errors="coerce").astype("float64")
        cols["amt_log"] = np.log1p(amt)
        cols["amt_cents"] = (amt - np.floor(amt)).round(3)

        for col, mapping in self.cat_maps_.items():
            cols[f"{col}_code"] = _to_str(df[col]).map(mapping).fillna(-1)  # unseen -> -1
        for col, mapping in self.freq_maps_.items():
            cols[f"{col}_freq"] = _as_key(df[col]).map(mapping).fillna(0.0)

        # How unusual is this amount for this 'customer'?
        uid = _uid(df)
        uid_mean = uid.map(self.uid_amt_mean_)
        cols["uid_count"] = uid.map(self.uid_count_).fillna(0)
        cols["uid_amt_mean"] = uid_mean
        cols["amt_to_uid_mean"] = amt / uid_mean.fillna(self.global_amt_mean_)
        cols["amt_zscore_uid"] = (amt - uid_mean) / uid.map(self.uid_amt_std_).replace(0, np.nan)

        out = pd.DataFrame(cols, index=df.index).astype("float32")
        if hasattr(self, "feature_names_"):
            out = out.reindex(columns=self.feature_names_)
        return out
