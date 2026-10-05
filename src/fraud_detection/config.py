from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
