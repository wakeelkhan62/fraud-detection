"""Small synthetic data that mimics the IEEE-CIS schema.

FOR TESTS ONLY. It exists so the pipeline can be tested without the real 1.3 GB dataset.
Never report results from this data; it is generated with a planted fraud pattern.
"""
import numpy as np
import pandas as pd


def make_synthetic(n: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    card1 = rng.integers(1000, 1100, n)
    amt = np.round(rng.lognormal(3.5, 1.0, n), 2) + 0.5
    product = rng.choice(list("WHCSR"), n)
    risky_card = card1 >= 1085
    logit = -4.5 + 2.5 * risky_card + 1.2 * (amt > 150) + 1.0 * (product == "C") + rng.normal(0, 0.3, n)
    is_fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    tx = pd.DataFrame({
        "TransactionID": np.arange(3_000_000, 3_000_000 + n),
        "isFraud": is_fraud,
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 180, n)),
        "TransactionAmt": amt,
        "ProductCD": product,
        "card1": card1,
        "card2": rng.choice([np.nan, 111.0, 321.0, 555.0], n),
        "card4": rng.choice(["visa", "mastercard", "discover"], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice([np.nan, 204.0, 325.0, 330.0], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "anonymous.com", np.nan], n),
        "C1": rng.poisson(2, n).astype(float),
        "D1": rng.choice([np.nan, 0.0, 14.0, 120.0], n),
        "M4": rng.choice(["M0", "M1", "M2", np.nan], n),
        "V258": rng.normal(1, 0.3, n),
    })
    ids = tx["TransactionID"].sample(frac=0.25, random_state=seed).values
    identity = pd.DataFrame({
        "TransactionID": ids,
        "DeviceType": rng.choice(["desktop", "mobile"], len(ids)),
        "id_02": rng.normal(100_000, 20_000, len(ids)),
    })
    return tx, identity
