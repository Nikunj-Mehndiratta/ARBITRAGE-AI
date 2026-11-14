"""Train ARBITRAGE AI without touching the final test period.

    python -m models.train                # use cached data if present
    python -m models.train --refresh-data # re-download from Yahoo Finance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbitrage.config import load_config  # noqa: E402
from arbitrage.pipeline import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ARBITRAGE AI model (train + validation only).")
    parser.add_argument("--config", default=None, help="Path to a config YAML (default: config.yaml)")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download market data")
    args = parser.parse_args()
    train(load_config(args.config), refresh_data=args.refresh_data)


if __name__ == "__main__":
    main()
