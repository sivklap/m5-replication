"""Facebook Prophet forecasting (paper Sections 4.2, 6.1)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm

from src.config import HORIZON, TRAIN_END_DAY
from src.evaluate import rmse
from src.load_data import build_holidays

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def make_prophet(holidays: pd.DataFrame | None = None) -> Prophet:
    """Match paper: trend + daily/weekly/monthly/quarterly/yearly seasonality."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.5,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    if holidays is not None and not holidays.empty:
        model.holidays = holidays
    return model


def _add_event_regressors(model: Prophet, train: pd.DataFrame) -> Prophet:
    """Paper engineered event variables as Prophet regressors."""
    for col in ("event_type_1", "event_type_2"):
        if col not in train.columns:
            continue
        train[col] = train[col].fillna("None")
        model.add_regressor(col)
    return model


def forecast_prophet(
    series_df: pd.DataFrame,
    holidays: pd.DataFrame | None = None,
    horizon: int = HORIZON,
    *,
    clip_negative: bool = True,
) -> pd.DataFrame:
    train = series_df[series_df["day_num"] <= TRAIN_END_DAY].copy()
    test = series_df[series_df["day_num"] > TRAIN_END_DAY].head(horizon).copy()

    prophet_train = train.rename(columns={"date": "ds", "sales": "y"})
    cols = ["ds", "y"]
    for col in ("event_type_1", "event_type_2"):
        if col in prophet_train.columns:
            prophet_train[col] = prophet_train[col].fillna("None")
            cols.append(col)

    model = make_prophet(holidays=holidays)
    model = _add_event_regressors(model, prophet_train)
    model.fit(prophet_train[cols])

    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    for col in ("event_type_1", "event_type_2"):
        if col in test.columns:
            future[col] = test[col].fillna("None").values

    forecast = model.predict(future)
    preds = forecast["yhat"].values
    if clip_negative:
        preds = np.maximum(preds, 0)

    result = test[["id", "cat_id", "day_num", "date", "sales"]].copy()
    result["prediction"] = preds
    return result


def forecast_prophet_batch(
    panel: pd.DataFrame,
    series_ids: list[str],
    holidays: pd.DataFrame | None = None,
    horizon: int = HORIZON,
    *,
    clip_negative: bool = False,
) -> pd.DataFrame:
    """Benchmark uses raw Prophet outputs (paper Table 3; negatives not clipped)."""
    preds: list[pd.DataFrame] = []
    for series_id in tqdm(series_ids, desc="Prophet"):
        series_df = panel.loc[panel["id"] == series_id]
        if series_df.empty:
            continue
        try:
            preds.append(
                forecast_prophet(
                    series_df,
                    holidays=holidays,
                    horizon=horizon,
                    clip_negative=clip_negative,
                )
            )
        except Exception as exc:
            print(f"Prophet failed for {series_id}: {exc}")
    if not preds:
        raise RuntimeError("Prophet produced no successful forecasts")
    return pd.concat(preds, ignore_index=True)


def evaluate_example(series_df: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    holidays = build_holidays(calendar)
    pred = forecast_prophet(series_df, holidays=holidays, clip_negative=True)
    score = rmse(pred["sales"], pred["prediction"])
    return {"rmse": score}
