"""Exploratory analysis mirroring paper Section 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import FIGURES_DIR
from src.load_data import build_panel, get_series


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_category_hierarchy(panel: pd.DataFrame) -> None:
    counts = (
        panel[["cat_id", "dept_id", "item_id"]]
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
    decomp = seasonal_decompose(daily, model="multiplicative", period=7)
    fig = decomp.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"Multiplicative decomposition: {series_id}", y=1.02)
    _save(fig, "02_decomposition.png")


def plot_weekday_sales(panel: pd.DataFrame) -> None:
    panel = panel.copy()
    panel["weekday"] = panel["date"].dt.day_name()
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    agg = panel.groupby("weekday", observed=True)["sales"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average sales by weekday")
    ax.set_ylabel("Mean unit sales")
    _save(fig, "03_weekday_sales.png")


def plot_category_sales(panel: pd.DataFrame) -> None:
    agg = panel.groupby("cat_id")["sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    agg.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Total sales by category")
    ax.set_ylabel("Unit sales")
    _save(fig, "04_category_sales.png")


def plot_event_price_effect(panel: pd.DataFrame) -> None:
    panel = panel.copy()
    panel["is_event"] = (
        panel["event_name_1"].notna() | panel["event_name_2"].notna()
    )
    agg = panel.groupby("is_event")["sell_price"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    agg.plot(kind="bar", ax=ax, color=["gray", "green"])
    ax.set_xticklabels(["Normal day", "Event day"], rotation=0)
    ax.set_title("Average sell price: event vs normal days")
    _save(fig, "05_event_price_effect.png")


def run_eda(data_dir: Path | None = None) -> pd.DataFrame:
    from src.config import DATA_DIR

    panel = build_panel(data_dir or DATA_DIR)
    print(f"Panel shape: {panel.shape}, series: {panel['id'].nunique()}")
    plot_category_hierarchy(panel)
    plot_decomposition(panel)
    plot_weekday_sales(panel)
    plot_category_sales(panel)
    plot_event_price_effect(panel)
    return panel
