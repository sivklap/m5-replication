"""Exploratory analysis mirroring paper Section 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import ARIMA_EXAMPLE_ID, FIGURES_DIR
from src.load_data import (
    build_panel,
    get_series,
    load_calendar,
    load_sales_wide,
    load_sell_prices,
    melt_sales,
)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_category_hierarchy(sales_wide: pd.DataFrame) -> None:
    counts = (
        sales_wide[["cat_id", "dept_id", "item_id"]]
        .drop_duplicates()
        .groupby(["cat_id", "dept_id"])
        .size()
        .reset_index(name="n_items")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=counts, x="dept_id", y="n_items", hue="cat_id", ax=ax)
    ax.set_title("M5 product hierarchy (items per department)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "01_category_hierarchy.png")


def plot_decomposition(panel: pd.DataFrame, series_id: str | None = None) -> None:
    if series_id is None:
        series_id = panel["id"].iloc[0]
    series = get_series(panel, series_id)
    daily = series.set_index("date")["sales"].asfreq("D", fill_value=0)
    # Multiplicative decomposition requires strictly positive values.
    daily = daily.clip(lower=0.01)
    decomp = seasonal_decompose(daily, model="multiplicative", period=7)
    fig = decomp.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"Multiplicative decomposition: {series_id}", y=1.02)
    _save(fig, "02_decomposition.png")


def plot_weekday_sales(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> None:
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    weekdays = (
        calendar.set_index("d")
        .reindex(day_cols)["date"]
        .dt.day_name()
        .to_numpy()
    )
    sales_values = sales_wide[day_cols].to_numpy()
    weekday_labels = np.repeat(weekdays, sales_values.shape[0])
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    agg = (
        pd.DataFrame({"weekday": weekday_labels, "sales": sales_values.ravel()})
        .groupby("weekday", observed=True)["sales"]
        .mean()
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average sales by weekday")
    ax.set_ylabel("Mean unit sales")
    _save(fig, "03_weekday_sales.png")


def plot_category_sales(sales_wide: pd.DataFrame) -> None:
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    totals = sales_wide.groupby("cat_id")[day_cols].sum().sum(axis=1)
    agg = totals.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    agg.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Total sales by category")
    ax.set_ylabel("Unit sales")
    _save(fig, "04_category_sales.png")


def plot_event_price_effect(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
) -> None:
    """Sample series to keep price/event merge within memory."""
    sample_ids = sales_wide["id"].drop_duplicates().sample(
        n=min(sample_size, sales_wide["id"].nunique()),
        random_state=seed,
    )
    panel = melt_sales(sales_wide[sales_wide["id"].isin(sample_ids)])
    panel = panel.merge(calendar, on="d", how="left")
    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
    panel["is_event"] = panel["event_name_1"].notna() | panel["event_name_2"].notna()
    agg = panel.groupby("is_event")["sell_price"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    agg.plot(kind="bar", ax=ax, color=["gray", "green"])
    ax.set_xticklabels(["Normal day", "Event day"], rotation=0)
    ax.set_title("Average sell price: event vs normal days")
    _save(fig, "05_event_price_effect.png")


def run_eda(data_dir: Path | None = None) -> pd.DataFrame:
    from src.config import DATA_DIR

    data_dir = data_dir or DATA_DIR
    sales_wide = load_sales_wide(data_dir)
    calendar = load_calendar(data_dir)
    prices = load_sell_prices(data_dir)
    print(
        f"Sales wide: {len(sales_wide):,} series, "
        f"{sum(c.startswith('d_') for c in sales_wide.columns):,} days"
    )

    plot_category_hierarchy(sales_wide)
    plot_weekday_sales(sales_wide, calendar)
    plot_category_sales(sales_wide)
    plot_event_price_effect(sales_wide, calendar, prices)

    example_id = ARIMA_EXAMPLE_ID
    example_panel = build_panel(data_dir, series_ids=[example_id], use_cache=False)
    plot_decomposition(example_panel, series_id=example_id)
    return example_panel
