"""Evaluate the frozen model on the final, unseen test period and write the reports.

    python -m models.evaluate

Run this after `python -m models.train`. Do not change config.yaml in response to what
this prints: that would be tuning on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbitrage.pipeline import evaluate  # noqa: E402


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
