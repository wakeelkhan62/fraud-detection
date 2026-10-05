import copy

import pytest

from fraud_detection.config import load_config
from fraud_detection.train import run

from .synthetic import make_synthetic


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    """Write synthetic IEEE-CIS-shaped CSVs to a temp folder (tests never need the real data)."""
    d = tmp_path_factory.mktemp("raw")
    tx, identity = make_synthetic(n=6000, seed=0)
    tx.to_csv(d / "train_transaction.csv", index=False)
    identity.to_csv(d / "train_identity.csv", index=False)
    return d


@pytest.fixture(scope="session")
def test_config(raw_dir, tmp_path_factory):
    cfg = copy.deepcopy(load_config("configs/config.yaml"))
    cfg["data"]["raw_dir"] = str(raw_dir)
    cfg["artifacts_dir"] = str(tmp_path_factory.mktemp("artifacts"))
    cfg["model"]["lgbm"]["n_estimators"] = 200  # keep the test fast
    cfg["model"]["early_stopping_rounds"] = 20
    return cfg


@pytest.fixture(scope="session")
def trained(test_config):
    """Run the full training pipeline once and share the result across tests."""
    report = run(test_config)
    return test_config, report
