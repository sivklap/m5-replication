"""LightGBM forecasting (paper Sections 4.3, 5.2, 6.2)."""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import HORIZON, LIGHTGBM_PARAMS, TRAIN_END_DAY
from src.evaluate import rmse
from src.features import (
    CATEGORICAL_FEATURES,
    LIGHTGBM_FEATURES,
    build_lightgbm_frame,
    prepare_lightgbm_matrix,
    train_test_feature_frames,
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in LIGHTGBM_FEATURES if c in df.columns]


def train_lightgbm(x_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in x_train.columns]
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    return lgb.train(LIGHTGBM_PARAMS, train_set)


def forecast_lightgbm_direct(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    """
    Paper-style LightGBM evaluation: one-step-ahead forecasts on the test
    window using actual past sales for lag features (teacher forcing).
    """
    if series_ids is not None:
        panel = panel[panel["id"].isin(series_ids)].copy()

    train_df, test_df = train_test_feature_frames(panel)
    x_train, y_train = prepare_lightgbm_matrix(train_df)
    x_test, _ = prepare_lightgbm_matrix(test_df)

    t0 = time.perf_counter()
    model = train_lightgbm(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    raw_preds = model.predict(x_test[_feature_cols(x_test)])
    predictions = test_df[
        ["id", "cat_id", "day_num", "date", "sales"]
    ].copy()
    predictions["prediction"] = np.maximum(raw_preds, 0)

    predictions = (
        predictions.sort_values(["id", "day_num"])
        .groupby("id", observed=True)
        .head(horizon)
        .reset_index(drop=True)
    )

    meta = {
        "train_seconds": train_seconds,
        "n_series": predictions["id"].nunique(),
        "mode": "direct_lags",
        "feature_importance": dict(
            zip(_feature_cols(x_train), model.feature_importance().tolist())
        ),
    }
    return predictions, meta


def forecast_lightgbm(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    return forecast_lightgbm_direct(panel, series_ids=series_ids, horizon=horizon)


def evaluate_example(panel: pd.DataFrame, series_id: str) -> dict:
    preds, meta = forecast_lightgbm(panel, series_ids=[series_id])
    score = rmse(preds["sales"], preds["prediction"])
    return {"rmse": score, **meta}
