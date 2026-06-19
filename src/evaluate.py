"""Evaluation metrics aligned with the paper (Table 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_per_series(
    results: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "prediction",
    id_col: str = "id",
) -> pd.Series:
    """RMSE computed independently for each series."""
    return results.groupby(id_col, observed=True).apply(
        lambda g: rmse(g[true_col], g[pred_col]),
        include_groups=False,
    )


def rmse_by_category(
    results: pd.DataFrame,
    true_col: str = "sales",
    pred_col: str = "prediction",
    cat_col: str = "cat_id",
    id_col: str = "id",
    *,
    paper_style: bool = True,
) -> pd.DataFrame:
    """
    Summarize RMSE by category.

    paper_style=True (default): mean of per-series RMSE within each category,
    matching Table 3 in arXiv:2203.06848.
    paper_style=False: pooled RMSE over all rows in the category.
    """
    rows: list[dict] = []
    series_scores = rmse_per_series(results, true_col, pred_col, id_col)

    for cat, group in results.groupby(cat_col, observed=True):
        ids = group[id_col].unique()
        if paper_style:
            score = float(series_scores.loc[series_scores.index.isin(ids)].mean())
        else:
            score = rmse(group[true_col], group[pred_col])
        rows.append({"cat_id": cat, "rmse": score, "n_series": len(ids)})

    if paper_style:
        total = float(series_scores.mean())
        n_series = results[id_col].nunique()
    else:
        total = rmse(results[true_col], results[pred_col])
        n_series = results[id_col].nunique()

    rows.append({"cat_id": "TOTAL", "rmse": total, "n_series": n_series})
    return pd.DataFrame(rows)
