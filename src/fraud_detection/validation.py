"""Data validation: refuse to train on broken data.

In production, bad data is the most common cause of silent model failure.
These checks run before every training job.
"""
import pandas as pd

from .data import AMOUNT_COL, ID_COL, TARGET, TIME_COL

REQUIRED_COLUMNS = [ID_COL, TIME_COL, AMOUNT_COL, TARGET, "ProductCD", "card1"]


class DataValidationError(ValueError):
    pass


def validate_training_data(df: pd.DataFrame, min_rows: int = 100) -> list[str]:
    """Return a list of problems. Empty list means the data passed."""
    issues = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing required columns: {missing}"]

    if len(df) < min_rows:
        issues.append(f"only {len(df)} rows; expected at least {min_rows}")
    if df[ID_COL].duplicated().any():
        issues.append(f"{int(df[ID_COL].duplicated().sum())} duplicate {ID_COL} values")
    if df[TIME_COL].isna().any():
        issues.append(f"{int(df[TIME_COL].isna().sum())} rows with missing {TIME_COL}")
    if (df[AMOUNT_COL] <= 0).any() or df[AMOUNT_COL].isna().any():
        issues.append(f"{AMOUNT_COL} has missing, zero or negative values")
    bad_target = ~df[TARGET].isin([0, 1])
    if bad_target.any():
        issues.append(f"{int(bad_target.sum())} rows where {TARGET} is not 0 or 1")
    else:
        rate = df[TARGET].mean()
        # IEEE-CIS fraud rate is about 3.5%. A wildly different rate usually means a join or filter bug.
        if not 0.001 <= rate <= 0.30:
            issues.append(f"fraud rate {rate:.4f} is outside the plausible range [0.001, 0.30]")
    return issues


def assert_valid(df: pd.DataFrame, min_rows: int = 100) -> None:
    issues = validate_training_data(df, min_rows=min_rows)
    if issues:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(issues))
