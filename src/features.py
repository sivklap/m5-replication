"""Feature engineering for LightGBM (paper Section 5.2)."""

from __future__ import annotations

import gc

import pandas as pd

from src.config import CATEGORIES, TEST_END_DAY, TEST_START_DAY, TRAIN_END_DAY

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
]

# Paper Table 1: engineered features plus M5 calendar / price fields.
EVENT_FEATURES = ["has_event_1", "has_event_2"]

ORIGINAL_FEATURES = [
    "wday",
    "month",
    "year",
    "wm_yr_wk",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
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
    *EVENT_FEATURES,
    *ORIGINAL_FEATURES,
    *CATEGORICAL_FEATURES,
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["mday"] = out["date"].dt.day.astype(int)
    # Paper: two binary event features for calendar holidays / special days.
    out["has_event_1"] = out["event_name_1"].notna().astype(int)
    out["has_event_2"] = out["event_name_2"].notna().astype(int)
    if "weekday" in out.columns:
        out["weekday"] = out["weekday"].astype("category")
    for col in ("wday", "month", "year", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def add_lag_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean features within one time series (actual sales)."""
    out = series_df.sort_values("day_num").copy()
    sales = out["sales"].astype(float)

    out["lag_7"] = sales.shift(7)
    out["lag_28"] = sales.shift(28)
    out["rmean_7_7"] = out["lag_7"].rolling(7, min_periods=1).mean()
    out["rmean_28_7"] = out["lag_28"].rolling(7, min_periods=1).mean()
    out["rmean_7_28"] = out["lag_7"].rolling(28, min_periods=1).mean()
    out["rmean_28_28"] = out["lag_28"].rolling(28, min_periods=1).mean()
    return out


def impute_sell_price(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["id", "day_num"]).copy()
    out["sell_price"] = out.groupby("id", observed=True)["sell_price"].ffill()
    out["sell_price"] = out.groupby("id", observed=True)["sell_price"].bfill()
    return out


def build_lightgbm_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for all series using actual sales for lags."""
    panel = impute_sell_price(panel)
    panel = add_calendar_features(panel)
    parts = []
    for _, group in panel.groupby("id", sort=False):
        parts.append(add_lag_features(group))
    return pd.concat(parts, ignore_index=True)


def prepare_lightgbm_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y for rows with valid lags."""
    feature_cols = [c for c in LIGHTGBM_FEATURES if c in df.columns]
    valid = df.dropna(subset=["lag_7", "lag_28"]).copy()
    x = valid[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in x.columns:
            x[col] = x[col].astype("category")
    y = valid["sales"].astype(float)
    return x, y


def train_test_feature_frames(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature-ready frames into train and test windows."""
    full = build_lightgbm_frame(panel)
    train = full[full["day_num"] <= TRAIN_END_DAY].copy()
    test = full[
        (full["day_num"] >= TEST_START_DAY) & (full["day_num"] <= TEST_END_DAY)
    ].copy()
    return train, test


def prepare_train_matrix_chunked(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the training matrix category-by-category to limit peak memory."""
    x_chunks: list[pd.DataFrame] = []
    y_chunks: list[pd.Series] = []
    for cat in CATEGORIES:
        cat_panel = panel.loc[panel["cat_id"] == cat]
        if cat_panel.empty:
            continue
        frame = build_lightgbm_frame(cat_panel)
        train_df = frame.loc[frame["day_num"] <= TRAIN_END_DAY]
        x_chunk, y_chunk = prepare_lightgbm_matrix(train_df)
        x_chunks.append(x_chunk)
        y_chunks.append(y_chunk)
        del frame, train_df, cat_panel, x_chunk, y_chunk
        gc.collect()
    return pd.concat(x_chunks, ignore_index=True), pd.concat(y_chunks, ignore_index=True)
