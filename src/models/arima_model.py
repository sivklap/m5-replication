"""ARIMA forecasting (paper Sections 4.1, 6)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from src.config import HORIZON, PAPER_ARIMA_EXAMPLE_ORDER, TRAIN_END_DAY
from src.evaluate import rmse


def select_arima_order(
    y: pd.Series,
    p_range: range = range(3),
    d_range: range = range(3),
    q_range: range = range(3),
) -> tuple[int, int, int, float]:
    best_aic = np.inf
    best_order = (1, 1, 1)
    y = y.dropna().astype(float)
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(y, order=(p, d, q))
            fit = model.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return (*best_order, best_aic)


def fit_arima(y: pd.Series, order: tuple[int, int, int] | None = None):
    y = y.dropna().astype(float)
    if order is None:
        p, d, q, _ = select_arima_order(y)
        order = (p, d, q)
    model = ARIMA(y, order=order)
    return model.fit(), order


def forecast_arima(
    series_df: pd.DataFrame,
    order: tuple[int, int, int] | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    train = series_df[series_df["day_num"] <= TRAIN_END_DAY].copy()
    test = series_df[series_df["day_num"] > TRAIN_END_DAY].head(horizon).copy()
    y_train = train["sales"]

    fit, used_order = fit_arima(y_train, order=order)
    forecast = fit.forecast(steps=horizon)
    forecast = np.maximum(forecast, 0)

    result = test[["id", "cat_id", "day_num", "date", "sales"]].copy()
    result["prediction"] = forecast.values
    meta = {"order": used_order, "aic": float(fit.aic)}
    return result, meta


def _forecast_arima_one(
    series_df: pd.DataFrame,
    order: tuple[int, int, int] | None,
    horizon: int,
) -> tuple[pd.DataFrame, dict]:
    pred, meta = forecast_arima(series_df, order=order, horizon=horizon)
    meta["id"] = series_df["id"].iloc[0]
    return pred, meta


def forecast_arima_batch(
    panel: pd.DataFrame,
    series_ids: list[str],
    order: tuple[int, int, int] | None = None,
    horizon: int = HORIZON,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    if n_jobs <= 1:
        for series_id in tqdm(series_ids, desc="ARIMA"):
            series_df = panel.loc[panel["id"] == series_id]
            if series_df.empty:
                continue
            pred, meta = forecast_arima(series_df, order=order, horizon=horizon)
            meta["id"] = series_id
            preds.append(pred)
            meta_rows.append(meta)
    else:
        tasks = {
            sid: panel.loc[panel["id"] == sid]
            for sid in series_ids
            if not panel.loc[panel["id"] == sid].empty
        }
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_forecast_arima_one, tasks[sid], order, horizon): sid
                for sid in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ARIMA"):
                pred, meta = fut.result()
                preds.append(pred)
                meta_rows.append(meta)

    return pd.concat(preds, ignore_index=True), pd.DataFrame(meta_rows)


def evaluate_example(series_df: pd.DataFrame) -> dict:
    pred, meta = forecast_arima(series_df, order=PAPER_ARIMA_EXAMPLE_ORDER)
    score = rmse(pred["sales"], pred["prediction"])
    return {"rmse": score, **meta}
