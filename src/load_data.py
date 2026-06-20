"""Load and merge M5 competition files into a long-format panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    BENCHMARK_SEED,
    BENCHMARK_SERIES_FILE,
    DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_END_DAY,
    TEST_START_DAY,
    TRAIN_END_DAY,
)

PANEL_CACHE = "panel.parquet"


def day_column(day_num: int) -> str:
    return f"d_{day_num}"


def _require_files(data_dir: Path) -> None:
    required = [
        "calendar.csv",
        "sales_train_evaluation.csv",
        "sell_prices.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing M5 files in {data_dir}: {', '.join(missing)}\n"
            "Download with: kaggle competitions download -c m5-forecasting-accuracy"
        )


def ensure_series_id(sales_wide: pd.DataFrame) -> pd.DataFrame:
    sales_wide = sales_wide.copy()
    if "id" not in sales_wide.columns:
        store_num = sales_wide["store_id"].astype(str).str.split("_").str[-1]
        sales_wide["id"] = (
            sales_wide["item_id"].astype(str)
            + "_"
            + sales_wide["state_id"].astype(str)
            + "_"
            + store_num
            + "_evaluation"
        )
    return sales_wide


def load_calendar(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    calendar = pd.read_csv(data_dir / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    if "d" not in calendar.columns:
        calendar.insert(0, "d", [f"d_{i}" for i in range(1, len(calendar) + 1)])
    return calendar


def load_sales_wide(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    return ensure_series_id(sales)


def load_sell_prices(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    prices["item_id"] = prices["item_id"].astype(str)
    prices["store_id"] = prices["store_id"].astype(str)
    return prices


def melt_sales(sales_wide: pd.DataFrame, day_cols: list[str] | None = None) -> pd.DataFrame:
    sales_wide = ensure_series_id(sales_wide)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if day_cols is None:
        day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long_df = sales_wide.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long_df["day_num"] = long_df["d"].str.replace("d_", "", regex=False).astype(int)
    long_df["item_id"] = long_df["item_id"].astype(str)
    long_df["store_id"] = long_df["store_id"].astype(str)
    return long_df


def _merge_calendar_prices(panel: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    calendar = load_calendar(data_dir)
    prices = load_sell_prices(data_dir)
    panel = panel.merge(calendar, on="d", how="left")
    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
    return panel.sort_values(["id", "day_num"]).reset_index(drop=True)


def build_panel(
    data_dir: Path = DATA_DIR,
    series_ids: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return merged long-format panel with calendar and prices."""
    _require_files(data_dir)
    cache_path = data_dir / PANEL_CACHE

    if series_ids is None and use_cache and cache_path.exists():
        print(f"Loading cached panel from {cache_path}")
        return pd.read_parquet(cache_path)

    sales_wide = load_sales_wide(data_dir)
    if series_ids is not None:
        sales_wide = sales_wide[sales_wide["id"].isin(series_ids)].copy()

    print(f"Melting {len(sales_wide):,} series x day columns...")
    panel = melt_sales(sales_wide)
    print("Merging calendar and prices...")
    panel = _merge_calendar_prices(panel, data_dir)

    if series_ids is None and use_cache:
        panel.to_parquet(cache_path, index=False)
        print(f"Cached panel to {cache_path}")

    return panel


def split_train_test(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["day_num"] <= TRAIN_END_DAY].copy()
    test = panel[
        (panel["day_num"] >= TEST_START_DAY) & (panel["day_num"] <= TEST_END_DAY)
    ].copy()
    return train, test


def sample_series_ids(
    panel: pd.DataFrame | None = None,
    per_category: int = 100,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
    use_saved: bool = True,
) -> list[str]:
    """Sample item-store ids for Table 3 style benchmarking."""
    if use_saved and BENCHMARK_SERIES_FILE.exists():
        saved = pd.read_csv(BENCHMARK_SERIES_FILE)
        if "id" in saved.columns and len(saved) >= per_category * 3:
            return saved["id"].tolist()

    if seed is None:
        seed = BENCHMARK_SEED

    if panel is None:
        sales = load_sales_wide(data_dir)
    else:
        sales = panel[["id", "cat_id"]].drop_duplicates()

    ids_by_cat: list[str] = []
    for cat in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        cat_ids = sales.loc[sales["cat_id"] == cat, "id"].drop_duplicates()
        n = min(per_category, len(cat_ids))
        sampled = cat_ids.sample(n=n, random_state=seed).tolist()
        ids_by_cat.extend(sampled)

    if use_saved:
        BENCHMARK_SERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids_by_cat}).to_csv(BENCHMARK_SERIES_FILE, index=False)
        print(f"Saved benchmark series ids to {BENCHMARK_SERIES_FILE}")

    return ids_by_cat


def resolve_series_id(panel: pd.DataFrame, series_id: str) -> str | None:
    """Match exact id or id with/without _evaluation suffix."""
    if series_id in panel["id"].values:
        return series_id
    if not series_id.endswith("_evaluation"):
        candidate = f"{series_id}_evaluation"
        if candidate in panel["id"].values:
            return candidate
    base = series_id.removesuffix("_evaluation")
    matches = panel.loc[panel["id"].str.startswith(base), "id"].drop_duplicates()
    if len(matches) == 1:
        return matches.iloc[0]
    return None


def get_series(panel: pd.DataFrame, series_id: str) -> pd.DataFrame:
    resolved = resolve_series_id(panel, series_id) or series_id
    return panel.loc[panel["id"] == resolved].sort_values("day_num").reset_index(drop=True)


def load_benchmark_series_ids(
    per_category: int = 100,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
) -> list[str]:
    """Fixed 300-series sample; prefers calibrated benchmark_series_ids.csv."""
    return sample_series_ids(
        per_category=per_category,
        seed=seed if seed is not None else BENCHMARK_SEED,
        data_dir=data_dir,
        use_saved=True,
    )


def build_holidays(calendar: pd.DataFrame) -> pd.DataFrame:
    """Prophet holidays table from M5 calendar events."""
    frames = []
    for col in ("event_name_1", "event_name_2"):
        events = calendar.loc[calendar[col].notna(), ["date", col]].copy()
        events = events.rename(columns={"date": "ds", col: "holiday"})
        frames.append(events)
    if not frames:
        return pd.DataFrame(columns=["ds", "holiday"])
    holidays = pd.concat(frames, ignore_index=True).drop_duplicates()
    holidays["ds"] = pd.to_datetime(holidays["ds"])
    return holidays
