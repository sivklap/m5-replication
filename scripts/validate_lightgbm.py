#!/usr/bin/env python3
"""Run LightGBM validation: both train scopes + full dataset vs paper targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, PAPER_TABLE3, RESULTS_DIR
from src.evaluate import rmse, rmse_by_category
from src.load_data import build_panel, sample_series_ids
from src.models.lightgbm_model import forecast_lightgbm

PAPER_LGB = PAPER_TABLE3["LightGBM"]


def _compare(summary, label: str) -> dict:
    rows = {}
    for _, r in summary.iterrows():
        cat = r["cat_id"]
        ours = float(r["rmse"])
        paper = PAPER_LGB.get(cat)
        rows[cat] = {
            "ours": ours,
            "paper": paper,
            "diff": ours - paper if paper is not None else None,
        }
    print(f"\n=== {label} vs paper Table 3 ===")
    for cat in ["HOUSEHOLD", "HOBBIES", "FOODS", "TOTAL"]:
        if cat in rows:
            d = rows[cat]
            diff = d["diff"]
            diff_s = f"{diff:+.3f}" if diff is not None else "n/a"
            print(f"  {cat}: {d['ours']:.3f} vs {d['paper']:.3f} ({diff_s})")
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series_ids = sample_series_ids(per_category=100, data_dir=DATA_DIR)
    subset_panel = build_panel(DATA_DIR, series_ids=series_ids, use_cache=False)

    report: dict = {"paper_table3_lightgbm": PAPER_LGB, "benchmarks": {}}

    for scope, panel in [("full", subset_panel), ("subset", subset_panel)]:
        print(f"\n--- Benchmark train_scope={scope} ---")
        preds, meta = forecast_lightgbm(
            panel,
            series_ids=series_ids,
            train_scope=scope,  # type: ignore[arg-type]
        )
        summary = rmse_by_category(preds)
        preds.to_csv(RESULTS_DIR / f"lightgbm_predictions_{scope}.csv", index=False)
        summary.to_csv(RESULTS_DIR / f"lightgbm_rmse_summary_{scope}.csv", index=False)
        report["benchmarks"][scope] = {
            "meta": meta,
            "comparison": _compare(summary, f"train_scope={scope}"),
        }

    print("\n--- Full dataset (30,490 series) ---")
    full_preds, full_meta = forecast_lightgbm(
        subset_panel.head(1),
        series_ids=None,
        train_scope="full",
    )
    full_score = rmse(full_preds["sales"], full_preds["prediction"])
    full_preds.to_csv(RESULTS_DIR / "lightgbm_full_predictions.csv", index=False)
    print(f"  Pooled RMSE: {full_score:.4f} (paper Section 6.2 target: 0.32)")
    report["full_dataset"] = {
        "pooled_rmse": full_score,
        "paper_target": 0.32,
        "diff": full_score - 0.32,
        "meta": full_meta,
    }

    out = RESULTS_DIR / "lightgbm_validation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
