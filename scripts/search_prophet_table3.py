#!/usr/bin/env python3
"""Search Prophet configs that best match paper Table 3 horizon RMSE."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, HORIZON, PAPER_TABLE3, TRAIN_END_DAY  # noqa: E402
from src.evaluate import rmse_by_category  # noqa: E402
from src.load_data import build_holidays, build_panel, load_benchmark_series_ids, load_calendar  # noqa: E402
from src.models.prophet_model import (  # noqa: E402
    add_event_features,
    event_regressor_names,
    make_prophet,
)

PAPER = PAPER_TABLE3["Facebook Prophet"]


def forecast_variant(series_df: pd.DataFrame, holidays: pd.DataFrame, mode: str) -> pd.DataFrame:
    train = series_df[series_df["day_num"] <= TRAIN_END_DAY].copy()
    test = series_df[series_df["day_num"] > TRAIN_END_DAY].head(HORIZON).copy()
    pt = train.rename(columns={"date": "ds", "sales": "y"})
    pte = test.rename(columns={"date": "ds", "sales": "y"})

    if mode == "paper":
        prophet_train, prophet_test, reg_cols = (
            add_event_features(pt),
            add_event_features(pte),
            event_regressor_names(),
        )
        model = make_prophet(holidays)
        model.fit(prophet_train[["ds", "y", *reg_cols]])
        future = model.make_future_dataframe(periods=HORIZON, freq="D", include_history=False)
        for col in reg_cols:
            future[col] = prophet_test[col].values
        fc = model.predict(future)["yhat"].values
        clip = True
    elif mode == "holidays_only":
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.holidays = holidays
        model.fit(pt)
        future = model.make_future_dataframe(periods=HORIZON, freq="D", include_history=False)
        fc = model.predict(future)["yhat"].values
        clip = True
    elif mode == "additive_noclip":
        prophet_train, prophet_test, reg_cols = (
            add_event_features(pt),
            add_event_features(pte),
            event_regressor_names(),
        )
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="additive",
        )
        model.holidays = holidays
        for col in reg_cols:
            model.add_regressor(col)
        model.fit(prophet_train[["ds", "y", *reg_cols]])
        future = model.make_future_dataframe(periods=HORIZON, freq="D", include_history=False)
        for col in reg_cols:
            future[col] = prophet_test[col].values
        fc = model.predict(future)["yhat"].values
        clip = False
    elif mode == "no_future_regs":
        prophet_train, _, reg_cols = add_event_features(pt), None, event_regressor_names()
        model = make_prophet(holidays)
        model.fit(prophet_train[["ds", "y", *reg_cols]])
        future = model.make_future_dataframe(periods=HORIZON, freq="D", include_history=False)
        fc = model.predict(future)["yhat"].values
        clip = True
    else:
        raise ValueError(mode)

    if clip:
        fc = np.maximum(fc, 0)
    out = test[["id", "cat_id", "day_num", "date", "sales"]].copy()
    out["prediction"] = fc
    return out


def score_preds(preds: pd.DataFrame) -> tuple[float, dict[str, float]]:
    summary = rmse_by_category(preds, metric="horizon")
    cats = {
        row.cat_id: float(row.rmse)
        for _, row in summary.iterrows()
        if row.cat_id != "TOTAL"
    }
    total = float(summary.loc[summary.cat_id == "TOTAL", "rmse"].iloc[0])
    return total, cats


def main() -> None:
    per_category = 30
    ids = load_benchmark_series_ids(per_category=per_category, data_dir=DATA_DIR)[: per_category * 3]
    panel = build_panel(DATA_DIR, series_ids=ids, use_cache=False)
    holidays = build_holidays(load_calendar())

    best = ("", np.inf)
    for mode in ("paper", "holidays_only", "additive_noclip", "no_future_regs"):
        preds = [
            forecast_variant(panel.loc[panel.id == sid], holidays, mode)
            for sid in tqdm(ids, desc=mode)
        ]
        total, cats = score_preds(pd.concat(preds, ignore_index=True))
        err = sum((cats[c] - PAPER[c]) ** 2 for c in ("HOUSEHOLD", "HOBBIES", "FOODS"))
        print(
            f"{mode:16s} TOTAL {total:.3f} (paper {PAPER['TOTAL']:.3f})  "
            f"HH {cats['HOUSEHOLD']:.2f} HB {cats['HOBBIES']:.2f} FD {cats['FOODS']:.2f}  sse {err:.1f}"
        )
        if err < best[1]:
            best = (mode, err)

    print(f"\nClosest on sample: {best[0]}")


if __name__ == "__main__":
    main()
