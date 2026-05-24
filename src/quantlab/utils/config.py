from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_universe(universe_path: str | Path, key: str) -> list[str]:
    cfg = load_config(universe_path)
    if key not in cfg:
        raise KeyError(f"Universe '{key}' not found in {universe_path}. Available: {list(cfg.keys())}")
    return cfg[key]
