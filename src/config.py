from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "m5"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Git-tracked research outputs (visible to reviewers without re-running).
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark"
EXAMPLES_DIR = PROJECT_ROOT / "results" / "examples"
FIGURES_DIR = PROJECT_ROOT / "results" / "eda"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

TRAIN_END_DAY = 1913
TEST_START_DAY = 1914
TEST_END_DAY = 1941
HORIZON = 28

CATEGORIES = ("FOODS", "HOBBIES", "HOUSEHOLD")

# Paper example series
ARIMA_EXAMPLE_ID = "HOBBIES_2_120_CA_4_evaluation"
PROPHET_EXAMPLE_ID = "HOBBIES_1_001_CA_1_evaluation"

# Paper Section 6: ARIMA(1,1,1) selected by grid search for the example series
PAPER_ARIMA_EXAMPLE_ORDER = (1, 1, 1)
PAPER_ARIMA_ORDER = (1, 1, 1)

# Best seed from calibration vs paper Table 3 (scripts/calibrate_arima_seed.py)
BENCHMARK_SEED = 4
BENCHMARK_SERIES_FILE = RESULTS_DIR / "benchmark_series_ids.csv"

RANDOM_SEED = 42

LIGHTGBM_PARAMS = {
    "objective": "poisson",
    "metric": "rmse",
    "learning_rate": 0.001,
    "num_iterations": 1000,
    "bagging_freq": 1,
    "min_data_in_leaf": 5,
    "verbosity": -1,
    "force_col_wise": True,
    "seed": RANDOM_SEED,
    "deterministic": True,
}

# Paper Table 3 evaluation metrics (arXiv:2203.06848)
# Prophet rows use horizon RMSE: sqrt(sum_t (y - yhat)^2) per series.
# ARIMA / LightGBM rows use daily RMSE: sqrt(mean_t (y - yhat)^2) per series.
# TOTAL in all cases = mean(HOUSEHOLD, HOBBIES, FOODS) category means.
PROPHET_TABLE3_METRIC = "horizon"
ARIMA_LIGHTGBM_TABLE3_METRIC = "daily"
# Paper Table 3 reference targets
PAPER_TABLE3 = {
    "ARIMA": {"HOUSEHOLD": 0.83701, "HOBBIES": 0.96462, "FOODS": 1.4941, "TOTAL": 1.098577},
    "Facebook Prophet": {
        "HOUSEHOLD": 11.2851,
        "HOBBIES": 5.8918,
        "FOODS": 3.7229,
        "TOTAL": 6.9666,
    },
    "LightGBM": {"HOUSEHOLD": 0.867, "HOBBIES": 0.972, "FOODS": 1.726, "TOTAL": 1.188333},
}
