#!/usr/bin/env python3
"""Replicate arXiv:2203.06848 on the M5 Walmart dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ARIMA_EXAMPLE_ID,
    BENCHMARK_SEED,
    BENCHMARK_SERIES_FILE,
    DATA_DIR,
    PAPER_ARIMA_ORDER,
    PAPER_TABLE3,
    PROPHET_EXAMPLE_ID,
    RESULTS_DIR,
)
from src.evaluate import per_series_rmse_table, rmse, rmse_by_category  # noqa: E402
from src.load_data import (  # noqa: E402
    build_holidays,
    build_panel,
    get_series,
    load_benchmark_series_ids,
    load_calendar,
)


def _save_json(obj: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"Saved {path}")


def _save_csv(df: pd.DataFrame, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def run_examples(panel: pd.DataFrame, data_dir: Path, methods: tuple[str, ...] = ("arima", "prophet", "lightgbm")) -> None:
    from src.models import arima_model, lightgbm_model, prophet_model

    example_ids = list({ARIMA_EXAMPLE_ID, PROPHET_EXAMPLE_ID})
    if not set(example_ids).issubset(set(panel["id"].unique())):
        panel = build_panel(data_dir, series_ids=example_ids, use_cache=False)

    calendar = load_calendar(data_dir)
    results: dict = {}

    if "arima" in methods:
        arima_series = get_series(panel, ARIMA_EXAMPLE_ID)
        if arima_series.empty:
            print(f"Warning: ARIMA example {ARIMA_EXAMPLE_ID} not found")
        else:
            print(f"\n=== ARIMA example: {arima_series['id'].iloc[0]} ===")
            results["arima"] = arima_model.evaluate_example(arima_series)
            print(json.dumps(results["arima"], indent=2, default=str))

    if "prophet" in methods or "lightgbm" in methods:
        prophet_series = get_series(panel, PROPHET_EXAMPLE_ID)
        if prophet_series.empty:
            print(f"Warning: Prophet example {PROPHET_EXAMPLE_ID} not found")
        else:
            sid = prophet_series["id"].iloc[0]
            if "prophet" in methods:
                print(f"\n=== Prophet example: {sid} ===")
                results["prophet"] = prophet_model.evaluate_example(prophet_series, calendar)
                print(json.dumps(results["prophet"], indent=2, default=str))

            if "lightgbm" in methods:
                print(f"\n=== LightGBM example: {sid} ===")
                full_panel = build_panel(data_dir, use_cache=True)
                lgb_out = lightgbm_model.evaluate_example(
                    full_panel,
                    sid,
                    train_panel=full_panel,
                )
                results["lightgbm"] = {
                    "rmse": lgb_out["rmse"],
                    "train_seconds": lgb_out["train_seconds"],
                    "n_train_series": lgb_out.get("n_train_series"),
                }
                print(json.dumps(results["lightgbm"], indent=2, default=str))

    _save_json(results, "example_rmse.json")


def run_benchmark(
    panel: pd.DataFrame | None,
    data_dir: Path,
    per_category: int = 100,
    methods: tuple[str, ...] = ("arima", "prophet", "lightgbm"),
    arima_order: tuple[int, int, int] | None = None,
    arima_jobs: int = 1,
) -> pd.DataFrame:
    series_ids = load_benchmark_series_ids(per_category=per_category, data_dir=data_dir)
    print(f"\nBenchmarking {len(series_ids)} series x 28-day horizon")

    if panel is None or not set(series_ids).issubset(set(panel["id"].unique())):
        print("Building subset panel for benchmark...")
        panel = build_panel(data_dir, series_ids=series_ids, use_cache=False)

    calendar = load_calendar(data_dir)
    holidays = build_holidays(calendar)
    table_rows: list[dict] = []

    if "arima" in methods:
        from src.models.arima_model import forecast_arima_batch

        print("\n--- ARIMA ---")
        arima_preds, arima_meta = forecast_arima_batch(
            panel, series_ids, order=arima_order, n_jobs=arima_jobs
        )
        summary = rmse_by_category(arima_preds, metric="daily")
        for _, row in summary.iterrows():
            table_rows.append(
                {"method": "ARIMA", "cat_id": row["cat_id"], "rmse": row["rmse"]}
            )
        _save_csv(arima_preds, "arima_predictions.csv")
        _save_csv(summary, "arima_rmse_summary.csv")
        _save_json({"meta": arima_meta.to_dict(orient="records") if not arima_meta.empty else []}, "arima_meta.json")

    if "prophet" in methods:
        from src.models.prophet_model import forecast_prophet_batch

        print("\n--- Prophet ---")
        prophet_preds = forecast_prophet_batch(panel, series_ids, holidays=holidays)
        summary = rmse_by_category(prophet_preds, metric="horizon")
        for _, row in summary.iterrows():
            table_rows.append(
                {"method": "Facebook Prophet", "cat_id": row["cat_id"], "rmse": row["rmse"]}
            )
        _save_csv(prophet_preds, "prophet_predictions.csv")
        _save_csv(summary, "prophet_rmse_summary.csv")

    if "lightgbm" in methods:
        from src.models.lightgbm_model import forecast_lightgbm

        print("\n--- LightGBM ---")
        print("Training global model on full M5 panel, evaluating benchmark subset...")
        full_panel = build_panel(data_dir, use_cache=True)
        lgb_preds, lgb_meta = forecast_lightgbm(
            full_panel,
            eval_series_ids=series_ids,
            train_panel=full_panel,
        )
        summary = rmse_by_category(lgb_preds, metric="daily")
        for _, row in summary.iterrows():
            table_rows.append(
                {"method": "LightGBM", "cat_id": row["cat_id"], "rmse": row["rmse"]}
            )
        per_series = per_series_rmse_table(lgb_preds)
        _save_csv(lgb_preds, "lightgbm_predictions.csv")
        _save_csv(summary, "lightgbm_rmse_summary.csv")
        _save_csv(per_series, "lightgbm_per_series_rmse.csv")
        foods_outliers = per_series.loc[per_series["cat_id"] == "FOODS"].head(10)
        _save_csv(foods_outliers, "lightgbm_foods_outliers.csv")
        if not foods_outliers.empty:
            print("\nTop FOODS outliers (per-series RMSE):")
            print(foods_outliers[["id", "rmse"]].to_string(index=False))
        _save_json(lgb_meta, "lightgbm_meta.json")

    table = pd.DataFrame(table_rows)
    pivot = table.pivot(index="cat_id", columns="method", values="rmse")
    print("\n=== Table 3 style RMSE ===")
    print("(Prophet: horizon RMSE; ARIMA/LightGBM: daily RMSE — see src/evaluate.py)")
    print(pivot.to_string())
    print("\n=== Paper Table 3 reference ===")
    for method, vals in PAPER_TABLE3.items():
        if method in pivot.columns:
            row = pivot[method]
            print(f"{method}: ours vs paper")
            for cat in ["HOUSEHOLD", "HOBBIES", "FOODS", "TOTAL"]:
                if cat in row.index and cat in vals:
                    print(f"  {cat}: {row[cat]:.3f} vs {vals[cat]:.3f}")
    _save_csv(table, "benchmark_rmse_table.csv")
    _save_csv(pivot.reset_index(), "benchmark_rmse_pivot.csv")
    _save_json({"paper_table3": PAPER_TABLE3, "ours": pivot.to_dict()}, "benchmark_comparison.json")
    return pivot


def run_full_lightgbm(panel: pd.DataFrame) -> None:
    from src.models.lightgbm_model import forecast_lightgbm

    print("\n--- LightGBM full dataset (30,490 series) ---")
    preds, meta = forecast_lightgbm(panel, eval_series_ids=None, train_panel=panel)
    summary = rmse_by_category(preds, metric="daily")
    pooled_rmse = rmse(preds["sales"], preds["prediction"])
    print(f"Full-dataset pooled daily RMSE: {pooled_rmse:.4f}")
    print(f"Full-dataset mean per-series daily RMSE: {summary.loc[summary['cat_id'] == 'TOTAL', 'rmse'].iloc[0]:.4f}")
    print(f"Training time: {meta['train_seconds']:.1f}s")
    meta["full_rmse_pooled"] = pooled_rmse
    meta["full_rmse_per_series_mean"] = float(
        summary.loc[summary["cat_id"] == "TOTAL", "rmse"].iloc[0]
    )
    meta["paper_full_rmse_reference"] = 0.32
    _save_csv(preds, "lightgbm_full_predictions.csv")
    _save_csv(summary, "lightgbm_full_rmse_summary.csv")
    _save_json(meta, "lightgbm_full_meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing M5 CSV files",
    )
    parser.add_argument(
        "--stage",
        choices=["eda", "examples", "benchmark", "lightgbm-full", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=100,
        help="Series sampled per category for benchmark (paper: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast benchmark with 5 series per category",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["arima", "prophet", "lightgbm"],
        default=["arima", "prophet", "lightgbm"],
        help="Models to include in benchmark",
    )
    parser.add_argument(
        "--arima-order",
        type=int,
        nargs=3,
        metavar=("P", "D", "Q"),
        default=(1, 1, 1),
        help="Fixed ARIMA order for benchmark (paper example: 1 1 1)",
    )
    parser.add_argument(
        "--arima-jobs",
        type=int,
        default=1,
        help="Parallel workers for ARIMA benchmark (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_category = 5 if args.quick else args.per_category
    arima_order = tuple(args.arima_order) if args.arima_order else PAPER_ARIMA_ORDER

    if args.stage == "eda":
        from src.eda import run_eda

        run_eda(args.data_dir)
        return

    print("M5 replication pipeline")
    panel: pd.DataFrame | None = None

    if args.stage in ("examples", "all"):
        example_ids = list({ARIMA_EXAMPLE_ID, PROPHET_EXAMPLE_ID})
        panel = build_panel(args.data_dir, series_ids=example_ids, use_cache=False)
        print(f"Example panel: {panel['id'].nunique()} series")
        run_examples(panel, args.data_dir, methods=tuple(args.methods))

    if args.stage in ("benchmark", "all"):
        run_benchmark(
            panel,
            data_dir=args.data_dir,
            per_category=per_category,
            methods=tuple(args.methods),
            arima_order=arima_order,
            arima_jobs=args.arima_jobs,
        )

    if args.stage in ("lightgbm-full", "all"):
        if args.quick:
            print("Skipping full LightGBM in --quick mode")
        else:
            print("Building full panel (first run caches to data/m5/panel.parquet)...")
            panel = build_panel(args.data_dir, use_cache=True)
            print(f"Panel: {panel.shape[0]:,} rows, {panel['id'].nunique():,} series")
            run_full_lightgbm(panel)

    if args.stage in ("eda", "all") and not args.quick:
        from src.eda import run_eda

        if panel is None or panel["id"].nunique() < 1000:
            panel = build_panel(args.data_dir, use_cache=True)
        run_eda(args.data_dir)


if __name__ == "__main__":
    main()
