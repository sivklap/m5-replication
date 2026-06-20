"""Tests for LightGBM feature engineering (paper Section 5.2)."""

from __future__ import annotations

import pandas as pd

from src.features import add_lag_features


def _make_series(sales: list[float]) -> pd.DataFrame:
    n = len(sales)
    return pd.DataFrame(
        {
            "day_num": range(1, n + 1),
            "sales": sales,
        }
    )


def test_rmean_naming_convention_window_on_lag_source() -> None:
    """rmean_{window}_{lag} uses rolling(window) on the named lag column."""
    sales = [float(i + 1) for i in range(40)]
    df = add_lag_features(_make_series(sales))

    pd.testing.assert_series_equal(
        df["rmean_7_7"],
        df["lag_7"].rolling(7, min_periods=1).mean(),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        df["rmean_28_7"],
        df["lag_7"].rolling(28, min_periods=1).mean(),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        df["rmean_7_28"],
        df["lag_28"].rolling(7, min_periods=1).mean(),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        df["rmean_28_28"],
        df["lag_28"].rolling(28, min_periods=1).mean(),
        check_names=False,
        check_dtype=False,
    )

    # Guard against the previous swapped implementation.
    wrong_28_7 = df["lag_28"].rolling(7, min_periods=1).mean()
    wrong_7_28 = df["lag_7"].rolling(28, min_periods=1).mean()
    assert not df["rmean_28_7"].equals(wrong_28_7)
    assert not df["rmean_7_28"].equals(wrong_7_28)
