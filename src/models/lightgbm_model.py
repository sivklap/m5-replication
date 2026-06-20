"""LightGBM forecasting (paper Sections 4.3, 5.2, 6.2)."""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import HORIZON, LIGHTGBM_PARAMS, TEST_END_DAY, TEST_START_DAY, TRAIN_END_DAY
from src.evaluate import rmse
from src.features import (
    CATEGORICAL_FEATURES,
    LIGHTGBM_FEATURES,
    build_lightgbm_frame,
    prepare_lightgbm_matrix,
    prepare_train_matrix_chunked,
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


def _prepare_train_matrix(train_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    if train_source["id"].nunique() > 1000:
        x_train, y_train = prepare_train_matrix_chunked(train_source)
        n_train_series = int(train_source["id"].nunique())
        return x_train, y_train, n_train_series

    train_full = build_lightgbm_frame(train_source)
    train_df = train_full.loc[train_full["day_num"] <= TRAIN_END_DAY]
    x_train, y_train = prepare_lightgbm_matrix(train_df)
    return x_train, y_train, int(train_df["id"].nunique())


def forecast_lightgbm_direct(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    eval_series_ids: list[str] | None = None,
    train_panel: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    """
    Paper-style LightGBM evaluation: one-step-ahead forecasts on the test
    window using actual past sales for lag features (teacher forcing).

    For Table 3 benchmarking, train on ``train_panel`` (ideally all 30,490
    series) but evaluate only ``eval_series_ids``. ``series_ids`` is kept as
    an alias for ``eval_series_ids`` for backwards compatibility.
    """
    if eval_series_ids is None:
        eval_series_ids = series_ids

    train_source = train_panel if train_panel is not None else panel
    eval_source = (
        panel[panel["id"].isin(eval_series_ids)].copy()
        if eval_series_ids is not None
        else panel
    )

    x_train, y_train, n_train_series = _prepare_train_matrix(train_source)
    eval_full = build_lightgbm_frame(eval_source)
    test_df = eval_full[
        (eval_full["day_num"] >= TEST_START_DAY)
        & (eval_full["day_num"] <= TEST_END_DAY)
    ].copy()
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
        "n_train_series": n_train_series,
        "mode": "direct_lags",
        "feature_importance": dict(
            zip(_feature_cols(x_train), model.feature_importance().tolist())
        ),
    }
    return predictions, meta


def forecast_lightgbm(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    eval_series_ids: list[str] | None = None,
    train_panel: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    return forecast_lightgbm_direct(
        panel,
        series_ids=series_ids,
        eval_series_ids=eval_series_ids,
        train_panel=train_panel,
        horizon=horizon,
    )


def evaluate_example(
    panel: pd.DataFrame,
    series_id: str,
    train_panel: pd.DataFrame | None = None,
) -> dict:
    preds, meta = forecast_lightgbm(
        panel,
        eval_series_ids=[series_id],
        train_panel=train_panel,
    )
    score = rmse(preds["sales"], preds["prediction"])
    return {"rmse": score, **meta}
