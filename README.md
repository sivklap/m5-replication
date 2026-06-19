# M5 Forecasting Replication

Replication of [arXiv:2203.06848](https://arxiv.org/abs/2203.06848) on the [M5 Walmart forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

The pipeline trains and evaluates three forecasting methods — ARIMA, Facebook Prophet, and LightGBM — on a 28-day horizon and compares per-category RMSE against the paper's Table 3.

## Models

| Method | Description |
|--------|-------------|
| **ARIMA** | Per-series ARIMA with fixed order `(1, 1, 1)` for the benchmark |
| **Prophet** | Per-series Prophet with calendar holidays |
| **LightGBM** | Global gradient-boosted model with Poisson objective and lag/price features |

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/sivklap/m5-replication.git
cd m5-replication
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the M5 dataset

Place your [Kaggle API token](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`, then run:

```bash
bash scripts/download_data.sh
```

This downloads the competition files into `data/m5/`. The dataset is not included in the repository (~763 MB).

## Usage

Run the full pipeline (examples, benchmark, full LightGBM, and EDA):

```bash
python run_replication.py
```

### Pipeline stages

| Stage | Description |
|-------|-------------|
| `eda` | Exploratory plots saved to `outputs/figures/` |
| `examples` | Single-series examples from the paper |
| `benchmark` | 100 series per category (300 total), Table 3 comparison |
| `lightgbm-full` | LightGBM trained on all 30,490 series |
| `all` | Run everything (default) |

### Common options

```bash
# Quick smoke test (5 series per category, skips full LightGBM)
python run_replication.py --quick

# Run only the benchmark with selected models
python run_replication.py --stage benchmark --methods arima prophet

# Parallel ARIMA fitting
python run_replication.py --stage benchmark --methods arima --arima-jobs 4
```

### Run stages separately

For long runs, each model can be executed independently and results merged afterward:

```bash
bash scripts/run_all_separate.sh
python scripts/combine_results.py
```

## Project structure

```
m5-replication/
├── run_replication.py      # Main entry point
├── src/
│   ├── config.py           # Paths, horizons, paper reference values
│   ├── load_data.py        # M5 CSV loading and panel construction
│   ├── features.py         # LightGBM feature engineering
│   ├── evaluate.py         # RMSE metrics
│   ├── eda.py              # Exploratory analysis plots
│   └── models/
│       ├── arima_model.py
│       ├── prophet_model.py
│       └── lightgbm_model.py
├── scripts/
│   ├── download_data.sh
│   ├── run_all_separate.sh
│   └── combine_results.py
├── data/m5/                # M5 dataset (gitignored)
└── outputs/                # Results and figures (gitignored)
```

## Outputs

Results are written to `outputs/results/`:

- `benchmark_rmse_pivot.csv` — per-category RMSE by method
- `benchmark_comparison.json` — side-by-side comparison with paper Table 3
- `*_predictions.csv` — per-model forecast files
- `run_*.log` — logs when using `run_all_separate.sh`

Figures from EDA are saved to `outputs/figures/`.

## Reference

Paper targets (Table 3, mean per-series RMSE):

| Method | HOUSEHOLD | HOBBIES | FOODS | TOTAL |
|--------|-----------|---------|-------|-------|
| ARIMA | 0.837 | 0.965 | 1.494 | 1.099 |
| Facebook Prophet | 11.285 | 5.892 | 3.723 | 6.967 |
| LightGBM | 0.867 | 0.972 | 1.726 | 1.188 |
