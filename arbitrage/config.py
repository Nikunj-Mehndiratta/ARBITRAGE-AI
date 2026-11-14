"""Configuration loading and project-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML experiment configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str | Path) -> Path:
    """Resolve a path from config.yaml against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def all_tickers(cfg: dict[str, Any]) -> list[str]:
    """Every ticker the experiment needs: both legs of each candidate pair plus the market proxy."""
    tickers: list[str] = []
    for a, b in cfg["data"]["candidate_pairs"]:
        tickers += [a, b]
    force = cfg["pair_selection"].get("force_pair")
    if force:
        tickers += list(force)
    tickers.append(cfg["data"]["market_ticker"])
    return list(dict.fromkeys(tickers))  # de-duplicate, keep order
