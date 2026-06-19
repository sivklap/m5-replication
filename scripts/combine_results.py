"""Merge per-model RMSE summaries and compare against paper Table 3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR

PAPER_TABLE_3 = {
    "ARIMA": {"HOUSEHOLD": 0.83701, "HOBBIES": 0.96462, "FOODS": 1.4941, "TOTAL": 1.098577},
    "Facebook Prophet": {
        "HOUSEHOLD": 11.2851,
        "HOBBIES": 5.8918,
        "FOODS": 3.7229,
        "TOTAL": 6.9666,
    },
    "LightGBM": {"HOUSEHOLD": 0.867, "HOBBIES": 0.972, "FOODS": 1.726, "TOTAL": 1.188333},
}

CAT_MAP = {"HOUSEHOLD": "HOUSEHOLD", "HOBBIES": "HOBBIES", "FOODS": "FOODS", "TOTAL": "TOTAL"}


def load_summary(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    method_files = {
        "ARIMA": "arima_rmse_summary.csv",
        "Facebook Prophet": "prophet_rmse_summary.csv",
        "LightGBM": "lightgbm_rmse_summary.csv",
    }

    rows: list[dict] = []
    for method, fname in method_files.items():
        df = load_summary(fname)
        if df is None:
            print(f"Missing {fname}, skipping {method}")
            continue
        for _, r in df.iterrows():
            cat = r["cat_id"]
            ours = float(r["rmse"])
            paper = PAPER_TABLE_3.get(method, {}).get(cat)
            rows.append(
                {
                    "method": method,
                    "cat_id": cat,
                    "our_rmse": ours,
                    "paper_rmse": paper,
                    "diff": ours - paper if paper is not None else None,
                }
            )

    if not rows:
        print("No result files found in outputs/results/")
        return

    comparison = pd.DataFrame(rows)
    pivot_ours = comparison.pivot(index="cat_id", columns="method", values="our_rmse")
    pivot_paper = pd.DataFrame(PAPER_TABLE_3).T
    pivot_paper.index.name = "method"
    pivot_paper = pivot_paper.reset_index().melt(id_vars="method", var_name="cat_id", value_name="paper_rmse")
    pivot_paper_wide = pivot_paper.pivot(index="cat_id", columns="method", values="paper_rmse")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "paper_comparison.csv", index=False)
    pivot_ours.to_csv(RESULTS_DIR / "our_rmse_pivot.csv")
    pivot_paper_wide.to_csv(RESULTS_DIR / "paper_rmse_pivot.csv")

    report = {
        "our_rmse": pivot_ours.to_dict(),
        "paper_rmse": PAPER_TABLE_3,
        "comparison_rows": rows,
    }
    (RESULTS_DIR / "paper_comparison.json").write_text(json.dumps(report, indent=2))

    print("=== Our results (RMSE) ===")
    print(pivot_ours.to_string())
    print("\n=== Paper Table 3 (RMSE) ===")
    print(pivot_paper_wide.to_string())
    print(f"\nSaved to {RESULTS_DIR}/paper_comparison.csv")


if __name__ == "__main__":
    main()
