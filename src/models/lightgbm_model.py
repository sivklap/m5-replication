"""LightGBM forecasting (paper Sections 4.3, 5.2, 6.2)."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import HORIZON, LIGHTGBM_PARAMS
from src.evaluate import rmse
from src.features import (
    CATEGORICAL_FEATURES,
    LIGHTGBM_FEATURES,
    PAPER_STRICT_FEATURES,
    build_test_feature_matrix,
    build_train_feature_matrix,
    prepare_lightgbm_matrix,
    train_test_feature_frames,
)
from src.load_data import PANEL_CHUNK_SIZE, iter_panel_chunks, load_sales_wide

TrainScope = Literal["full", "subset"]

_BATCHED_TRAIN_THRESHOLD = 1000


def _feature_cols(df: pd.DataFrame, *, paper_strict: bool = False) -> list[str]:
    allowed = (
        [c for c in LIGHTGBM_FEATURES if c not in ("event_type_1", "event_type_2", "sell_price")]
        if paper_strict
        else LIGHTGBM_FEATURES
    )
    return [c for c in allowed if c in df.columns]


def _full_dataset_chunk_count() -> int:
    return (len(load_sales_wide()) + PANEL_CHUNK_SIZE - 1) // PANEL_CHUNK_SIZE


def train_lightgbm(x_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in x_train.columns]
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    return lgb.train(LIGHTGBM_PARAMS, train_set)


def train_lightgbm_incremental(
    panel_chunks: Iterator[pd.DataFrame],
    n_chunks: int,
    *,
    paper_strict: bool = False,
) -> lgb.Booster:
    """Train across panel chunks when the full matrix does not fit in memory."""
    total_iterations = int(LIGHTGBM_PARAMS["num_iterations"])
    base_iters = total_iterations // n_chunks
    remainder = total_iterations % n_chunks

    booster: lgb.Booster | None = None

    for chunk_idx, panel_chunk in enumerate(panel_chunks):
        x_chunk, y_chunk = build_train_feature_matrix(panel_chunk)
        x_chunk = x_chunk[_feature_cols(x_chunk, paper_strict=paper_strict)]
        n_iter = base_iters + (1 if chunk_idx < remainder else 0)
        if n_iter <= 0:
            continue

        chunk_cats = [c for c in CATEGORICAL_FEATURES if c in x_chunk.columns]
        train_set = lgb.Dataset(
            x_chunk,
            label=y_chunk,
            categorical_feature=chunk_cats,
            free_raw_data=False,
        )
        params = {**LIGHTGBM_PARAMS, "num_iterations": n_iter}
        booster = lgb.train(
            params,
            train_set,
            init_model=booster,
            keep_training_booster=True,
        )
        del panel_chunk, x_chunk, y_chunk, train_set
        gc.collect()

    if booster is None:
        raise RuntimeError("No training data found for incremental LightGBM fit")
    return booster


def _predict_test_chunks(
    model: lgb.Booster,
    panel_chunks: Iterator[pd.DataFrame],
    series_ids: list[str] | None,
    horizon: int,
    *,
    paper_strict: bool = False,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for panel_chunk in panel_chunks:
        test_df = build_test_feature_matrix(panel_chunk, series_ids=series_ids)
        if test_df.empty:
            continue
        x_test, _ = prepare_lightgbm_matrix(test_df, paper_strict=paper_strict)
        raw_preds = model.predict(x_test[_feature_cols(x_test, paper_strict=paper_strict)])
        preds = test_df[["id", "cat_id", "day_num", "date", "sales"]].copy()
        preds["prediction"] = np.maximum(raw_preds, 0)
        parts.append(preds)
        del panel_chunk, test_df, x_test, preds
        gc.collect()

    if not parts:
        raise RuntimeError("No test rows found for LightGBM prediction")

    predictions = pd.concat(parts, ignore_index=True)
    return (
        predictions.sort_values(["id", "day_num"])
        .groupby("id", observed=True)
        .head(horizon)
        .reset_index(drop=True)
    )


def _finalize_meta(
    model: lgb.Booster,
    x_ref: pd.DataFrame,
    train_seconds: float,
    predictions: pd.DataFrame,
    *,
    train_scope: TrainScope,
    n_train_series: int,
    batched_train: bool,
    incremental_train: bool,
    paper_strict: bool = False,
) -> dict:
    return {
        "train_seconds": train_seconds,
        "n_series": predictions["id"].nunique(),
        "mode": "direct_lags",
        "train_scope": train_scope,
        "n_train_series": n_train_series,
        "batched_train": batched_train,
        "incremental_train": incremental_train,
        "paper_strict": paper_strict,
        "feature_importance": dict(
            zip(_feature_cols(x_ref, paper_strict=paper_strict), model.feature_importance().tolist())
        ),
    }


def forecast_lightgbm_direct(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    horizon: int = HORIZON,
    *,
    train_scope: TrainScope = "full",
    paper_strict: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Paper-style LightGBM evaluation: one-step-ahead forecasts on the test
    window using actual past sales for lag features (teacher forcing).

    train_scope:
      - full: train on all series in the dataset; evaluate on series_ids if given
      - subset: train and evaluate only on series_ids
    """
    eval_ids = series_ids
    use_incremental = train_scope == "full" and (
        series_ids is None or panel["id"].nunique() <= len(eval_ids or [])
    )

    if use_incremental:
        n_chunks = _full_dataset_chunk_count()
        n_train_series = len(load_sales_wide())

        t0 = time.perf_counter()
        model = train_lightgbm_incremental(
            iter_panel_chunks(), n_chunks, paper_strict=paper_strict
        )
        train_seconds = time.perf_counter() - t0

        predictions = _predict_test_chunks(
            model,
            iter_panel_chunks(),
            series_ids=eval_ids,
            horizon=horizon,
            paper_strict=paper_strict,
        )

        x_ref = None
        for panel_chunk in iter_panel_chunks():
            if eval_ids is not None:
                chunk_eval = [sid for sid in eval_ids if sid in panel_chunk["id"].values]
                if not chunk_eval:
                    continue
                ref_ids = [chunk_eval[0]]
            else:
                ref_ids = None
            ref_test = build_test_feature_matrix(panel_chunk, series_ids=ref_ids)
            if not ref_test.empty:
                x_ref, _ = prepare_lightgbm_matrix(ref_test, paper_strict=paper_strict)
                break
        if x_ref is None:
            x_ref = pd.DataFrame(columns=PAPER_STRICT_FEATURES if paper_strict else LIGHTGBM_FEATURES)

        meta = _finalize_meta(
            model,
            x_ref,
            train_seconds,
            predictions,
            train_scope=train_scope,
            n_train_series=n_train_series,
            batched_train=True,
            incremental_train=True,
            paper_strict=paper_strict,
        )
        return predictions, meta

    if train_scope == "subset" and series_ids is not None:
        train_panel = panel[panel["id"].isin(series_ids)].copy()
    else:
        train_panel = panel

    use_batched = train_panel["id"].nunique() > _BATCHED_TRAIN_THRESHOLD
    if use_batched:
        x_train, y_train = build_train_feature_matrix(train_panel)
        x_train = x_train[_feature_cols(x_train, paper_strict=paper_strict)]
        test_df = build_test_feature_matrix(train_panel, series_ids=eval_ids)
        x_test, _ = prepare_lightgbm_matrix(test_df, paper_strict=paper_strict)
        gc.collect()
    else:
        train_df, test_df = train_test_feature_frames(
            train_panel,
            eval_series_ids=eval_ids,
        )
        x_train, y_train = prepare_lightgbm_matrix(train_df, paper_strict=paper_strict)
        x_test, _ = prepare_lightgbm_matrix(test_df, paper_strict=paper_strict)

    t0 = time.perf_counter()
    model = train_lightgbm(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    raw_preds = model.predict(x_test[_feature_cols(x_test, paper_strict=paper_strict)])
    predictions = test_df[["id", "cat_id", "day_num", "date", "sales"]].copy()
    predictions["prediction"] = np.maximum(raw_preds, 0)
    predictions = (
        predictions.sort_values(["id", "day_num"])
        .groupby("id", observed=True)
        .head(horizon)
        .reset_index(drop=True)
    )

    meta = _finalize_meta(
        model,
        x_train,
        train_seconds,
        predictions,
        train_scope=train_scope,
        n_train_series=train_panel["id"].nunique(),
        batched_train=use_batched,
        incremental_train=False,
        paper_strict=paper_strict,
    )
    return predictions, meta


def forecast_lightgbm(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
    horizon: int = HORIZON,
    *,
    train_scope: TrainScope = "full",
    paper_strict: bool = False,
) -> tuple[pd.DataFrame, dict]:
    return forecast_lightgbm_direct(
        panel,
        series_ids=series_ids,
        horizon=horizon,
        train_scope=train_scope,
        paper_strict=paper_strict,
    )


def evaluate_example(panel: pd.DataFrame, series_id: str) -> dict:
    preds, meta = forecast_lightgbm(
        panel,
        series_ids=[series_id],
        train_scope="subset",
    )
    score = rmse(preds["sales"], preds["prediction"])
    return {"rmse": score, **meta}
