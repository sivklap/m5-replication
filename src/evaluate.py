"""Evaluation metrics aligned with the paper (Table 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Per-day RMSE: sqrt(mean((y - yhat)^2))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def horizon_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Paper Table 3 Prophet metric: sqrt(sum((y - yhat)^2)) over the 28-day horizon.

    arXiv:2203.06848 Table 3 Prophet TOTAL (~6.97) matches this aggregation on our
  forecasts (daily RMSE ~1.28 -> horizon ~6.76). ARIMA/LightGBM rows use daily RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.sum((y_true - y_pred) ** 2)))


def rmse_per_series(
    results: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "prediction",
    id_col: str = "id",
    *,
    metric: str = "daily",
) -> pd.Series:
    """RMSE computed independently for each series."""
    scorer = horizon_rmse if metric == "horizon" else rmse
    return results.groupby(id_col, observed=True).apply(
        lambda g: scorer(g[true_col], g[pred_col]),
        include_groups=False,
    )


def rmse_by_category(
    results: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "prediction",
    cat_col: str = "cat_id",
    id_col: str = "id",
    *,
    metric: str = "daily",
) -> pd.DataFrame:
    """
    Summarize RMSE by category for Table 3.

    metric='daily': mean of per-series daily RMSE (ARIMA / LightGBM).
    metric='horizon': mean of per-series horizon RMSE (Prophet Table 3).
    TOTAL: mean of the three category means (paper Table 3 footer).
    """
    rows: list[dict] = []
    series_scores = rmse_per_series(results, true_col, pred_col, id_col, metric=metric)

    for cat, group in results.groupby(cat_col, observed=True):
        ids = group[id_col].unique()
        score = float(series_scores.loc[series_scores.index.isin(ids)].mean())
        rows.append({"cat_id": cat, "rmse": score, "n_series": len(ids)})

    cat_means = [row["rmse"] for row in rows]
    total = float(np.mean(cat_means))
    n_series = results[id_col].nunique()
    rows.append({"cat_id": "TOTAL", "rmse": total, "n_series": n_series})
    return pd.DataFrame(rows)


def per_series_rmse_table(
    results: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "prediction",
    id_col: str = "id",
    cat_col: str = "cat_id",
    *,
    metric: str = "daily",
) -> pd.DataFrame:
    """Per-series RMSE with category labels for outlier inspection."""
    meta = results[[id_col, cat_col]].drop_duplicates()
    scores = rmse_per_series(results, true_col, pred_col, id_col, metric=metric).rename(
        "rmse"
    )
    table = meta.set_index(id_col).join(scores).reset_index()
    return table.sort_values("rmse", ascending=False).reset_index(drop=True)


def table3_total(category_summary: pd.DataFrame) -> float:
    """Paper Table 3 TOTAL = average of HOUSEHOLD, HOBBIES, FOODS means."""
    cats = category_summary.loc[category_summary["cat_id"] != "TOTAL", "rmse"]
    return float(cats.mean())
