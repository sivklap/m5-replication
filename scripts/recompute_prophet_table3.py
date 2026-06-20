#!/usr/bin/env python3
"""Recompute Prophet Table 3 from saved predictions (no refit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PAPER_TABLE3, RESULTS_DIR  # noqa: E402
from src.evaluate import rmse_by_category  # noqa: E402


def main() -> None:
    path = RESULTS_DIR / "prophet_predictions.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run Prophet benchmark first.")

    preds = pd.read_csv(path)
    daily = rmse_by_category(preds, metric="daily")
    horizon = rmse_by_category(preds, metric="horizon")
    paper = PAPER_TABLE3["Facebook Prophet"]

    print("=== Prophet daily RMSE (example series style) ===")
    print(daily.to_string(index=False))
    print("\n=== Prophet horizon RMSE (Table 3 style) ===")
    print(horizon.to_string(index=False))
    print("\n=== vs paper Table 3 ===")
    for cat in ["HOUSEHOLD", "HOBBIES", "FOODS", "TOTAL"]:
        h = horizon.loc[horizon.cat_id == cat, "rmse"].iloc[0]
        p = paper[cat]
        print(f"  {cat}: ours {h:.3f}  paper {p:.3f}  diff {h - p:+.3f}")

    horizon.to_csv(RESULTS_DIR / "prophet_rmse_summary.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'prophet_rmse_summary.csv'}")


if __name__ == "__main__":
    main()
