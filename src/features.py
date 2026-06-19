"""Feature engineering for LightGBM (paper Section 5.2)."""

from __future__ import annotations

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
