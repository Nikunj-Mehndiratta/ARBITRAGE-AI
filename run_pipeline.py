"""Run the whole experiment: download/validate data, train, then evaluate on the test period.

    python run_pipeline.py
    python run_pipeline.py --refresh-data
"""

from __future__ import annotations

import argparse

from arbitrage.pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ARBITRAGE AI experiment.")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download market data")
    args = parser.parse_args()
    run_all(refresh_data=args.refresh_data)


if __name__ == "__main__":
    main()
