"""Feature engineering for LightGBM (paper Section 5.2)."""

from __future__ import annotations

import gc
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import TEST_END_DAY, TEST_START_DAY, TRAIN_END_DAY

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

LIGHTGBM_FEATURES = [
    "lag_7",
    "lag_28",
    "rmean_7_7",
    "rmean_28_7",
    "rmean_7_28",
    "rmean_28_28",
    "week",
    "quarter",
    "mday",
    "event_type_1",
    "event_type_2",
    "sell_price",
    *CATEGORICAL_FEATURES,
]

# Table 1 snapshot + categorical IDs (strict paper replication mode)
PAPER_STRICT_FEATURES = [
    "lag_7",
    "lag_28",
    "rmean_7_7",
    "rmean_28_7",
    "rmean_7_28",
    "rmean_28_28",
    "week",
    "quarter",
    "mday",
    *CATEGORICAL_FEATURES,
]

PANEL_FEATURE_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "day_num",
    "date",
    "sales",
    "sell_price",
    "event_type_1",
    "event_type_2",
]

TRAIN_BATCH_SIZE = 2500


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["mday"] = out["date"].dt.day.astype(int)
    out["event_type_1"] = out["event_type_1"].fillna("None").astype("category")
    out["event_type_2"] = out["event_type_2"].fillna("None").astype("category")
    return out


def add_lag_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean features within one time series (actual sales)."""
    out = series_df.sort_values("day_num").copy()
    sales = out["sales"].astype(np.float32)

    out["lag_7"] = sales.shift(7).astype(np.float32)
    out["lag_28"] = sales.shift(28).astype(np.float32)
    # Paper naming: rmean_{window}_{lag} = rolling mean of lag feature over window days
    out["rmean_7_7"] = out["lag_7"].rolling(7, min_periods=1).mean().astype(np.float32)
    out["rmean_28_7"] = out["lag_7"].rolling(28, min_periods=1).mean().astype(np.float32)
    out["rmean_7_28"] = out["lag_28"].rolling(7, min_periods=1).mean().astype(np.float32)
    out["rmean_28_28"] = out["lag_28"].rolling(28, min_periods=1).mean().astype(np.float32)
    return out


def impute_sell_price(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["id", "day_num"]).copy()
    out["sell_price"] = out.groupby("id", observed=True)["sell_price"].ffill()
    out["sell_price"] = out.groupby("id", observed=True)["sell_price"].bfill()
    return out


def compact_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for LightGBM features."""
    cols = [c for c in PANEL_FEATURE_COLS if c in panel.columns]
    out = panel[cols].copy()
    out["sales"] = out["sales"].astype(np.float32)
    if "sell_price" in out.columns:
        out["sell_price"] = out["sell_price"].astype(np.float32)
    return out


def build_lightgbm_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for all series using actual sales for lags."""
    panel = compact_panel(panel)
    panel = impute_sell_price(panel)
    panel = add_calendar_features(panel)
    parts = []
    for _, group in panel.groupby("id", sort=False):
        parts.append(add_lag_features(group))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _iter_series_batches(series_ids: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(series_ids), batch_size):
        yield series_ids[start : start + batch_size]


def build_train_feature_matrix(
    panel: pd.DataFrame,
    batch_size: int = TRAIN_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training matrix in series batches to limit peak memory."""
    panel = compact_panel(panel)
    panel = impute_sell_price(panel)
    panel = add_calendar_features(panel)

    series_ids = panel["id"].unique()
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for batch_ids in _iter_series_batches(series_ids, batch_size):
        batch = panel[panel["id"].isin(batch_ids) & (panel["day_num"] <= TRAIN_END_DAY)]
        parts = []
        for _, group in batch.groupby("id", sort=False):
            parts.append(add_lag_features(group))
        framed = pd.concat(parts, ignore_index=True)
        x_batch, y_batch = prepare_lightgbm_matrix(framed)
        x_parts.append(x_batch)
        y_parts.append(y_batch)
        del batch, framed, x_batch, y_batch
        gc.collect()

    return pd.concat(x_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)


def build_test_feature_matrix(
    panel: pd.DataFrame,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build test-window feature rows (needs history through TEST_END_DAY)."""
    if panel.empty:
        return panel.copy()
    panel = compact_panel(panel)
    if series_ids is not None:
        panel = panel[panel["id"].isin(series_ids)]
        if panel.empty:
            return panel.copy()
    panel = panel[panel["day_num"] <= TEST_END_DAY]
    framed = build_lightgbm_frame(panel)
    if framed.empty:
        return framed
    return framed[
        (framed["day_num"] >= TEST_START_DAY) & (framed["day_num"] <= TEST_END_DAY)
    ].copy()


def prepare_lightgbm_matrix(
    df: pd.DataFrame,
    *,
    paper_strict: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y for rows with valid lags."""
    allowed = PAPER_STRICT_FEATURES if paper_strict else LIGHTGBM_FEATURES
    feature_cols = [c for c in allowed if c in df.columns]
    valid = df.dropna(subset=["lag_7", "lag_28"]).copy()
    x = valid[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in x.columns:
            x[col] = x[col].astype("category")
    y = valid["sales"].astype(np.float32)
    return x, y


def train_test_feature_frames(
    panel: pd.DataFrame,
    *,
    eval_series_ids: list[str] | None = None,
    use_batched_train: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature-ready frames into train and test windows."""
    if use_batched_train:
        x_train, y_train = build_train_feature_matrix(panel)
        train = x_train.copy()
        train["sales"] = y_train.values
        test = build_test_feature_matrix(panel, series_ids=eval_series_ids)
        return train, test

    full = build_lightgbm_frame(panel)
    train = full[full["day_num"] <= TRAIN_END_DAY].copy()
    test = full[
        (full["day_num"] >= TEST_START_DAY) & (full["day_num"] <= TEST_END_DAY)
    ].copy()
    if eval_series_ids is not None:
        test = test[test["id"].isin(eval_series_ids)].copy()
    return train, test
