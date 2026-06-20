from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "m5"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

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
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
}

# Paper Table 3 targets for reference
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
