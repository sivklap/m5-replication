"""Facebook Prophet forecasting (paper Sections 4.2, 5.1, 6.1)."""

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

# M5 calendar event types (paper Section 5.1: engineered event features)
EVENT_TYPES: dict[str, tuple[str, ...]] = {
    "event_type_1": ("Cultural", "National", "Religious", "Sporting"),
    "event_type_2": ("Cultural", "Religious"),
}


def event_regressor_names() -> list[str]:
    """One-hot column names for Prophet extra regressors."""
    names: list[str] = []
    for src, values in EVENT_TYPES.items():
        for value in values:
            names.append(f"evt_{src}_{value}")
    return names


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode event_type_1/2 as numeric 0/1 regressors for Prophet."""
    out = df.copy()
    for src, values in EVENT_TYPES.items():
        if src not in out.columns:
            continue
        for value in values:
            out[f"evt_{src}_{value}"] = (out[src] == value).astype(float)
    return out


def make_prophet(holidays: pd.DataFrame | None = None) -> Prophet:
    """
    Match paper Section 4.2 / 5.1:
    multiplicative trend + daily/weekly/biweekly/monthly/quarterly/yearly seasonality.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.5,
    )
    model.add_seasonality(name="biweekly", period=14, fourier_order=5)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    if holidays is not None and not holidays.empty:
        model.holidays = holidays
    for reg in event_regressor_names():
        model.add_regressor(reg)
    return model


def _prepare_prophet_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    prophet_train = add_event_features(train.rename(columns={"date": "ds", "sales": "y"}))
    prophet_test = add_event_features(test.rename(columns={"date": "ds", "sales": "y"}))
    reg_cols = event_regressor_names()
    cols = ["ds", "y", *reg_cols]
    return prophet_train[cols], prophet_test, reg_cols


def forecast_prophet(
    series_df: pd.DataFrame,
    holidays: pd.DataFrame | None = None,
    horizon: int = HORIZON,
    *,
    clip_negative: bool = True,
) -> pd.DataFrame:
    train = series_df[series_df["day_num"] <= TRAIN_END_DAY].copy()
    test = series_df[series_df["day_num"] > TRAIN_END_DAY].head(horizon).copy()

    prophet_train, prophet_test, reg_cols = _prepare_prophet_frames(train, test)

    model = make_prophet(holidays=holidays)
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    for col in reg_cols:
        future[col] = prophet_test[col].values

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
    clip_negative: bool = True,
) -> pd.DataFrame:
    """Table 3 benchmark: per-series Prophet, paper-style negative clipping."""
    preds: list[pd.DataFrame] = []
    failures: list[str] = []
    for series_id in tqdm(series_ids, desc="Prophet"):
        series_df = panel.loc[panel["id"] == series_id]
        if series_df.empty:
            failures.append(series_id)
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
            failures.append(series_id)
            print(f"Prophet failed for {series_id}: {exc}")
    if failures:
        print(f"Prophet failures: {len(failures)} / {len(series_ids)}")
    if not preds:
        raise RuntimeError("Prophet produced no successful forecasts")
    return pd.concat(preds, ignore_index=True)


def evaluate_example(series_df: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    """Paper Section 6.1 example: HOBBIES_1_001_CA_1, target RMSE ~1.71."""
    holidays = build_holidays(calendar)
    pred = forecast_prophet(series_df, holidays=holidays, clip_negative=True)
    score = rmse(pred["sales"], pred["prediction"])
    return {"rmse": score, "clip_negative": True}
