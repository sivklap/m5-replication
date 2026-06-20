# M5 Forecasting Replication

Replication of [arXiv:2203.06848](https://arxiv.org/abs/2203.06848) on the [M5 Walmart forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

The pipeline trains and evaluates three forecasting methods — ARIMA, Facebook Prophet, and LightGBM — on a 28-day horizon and compares per-category RMSE against the paper's Table 3.

## For reviewers

**Code** lives under `src/` and `run_replication.py`.

**Outputs** are committed under [`results/`](results/) (plots, RMSE tables, predictions, logs). See [`results/README.md`](results/README.md) for a file guide. You do not need to download data or re-run anything to inspect the results.

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

Alternative (no Kaggle account):

```bash
mkdir -p data/m5 && curl -L -o data/m5/m5.zip \
  https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip
unzip -o data/m5/m5.zip -d data/m5
```

## Usage

Run the full pipeline (examples, benchmark, full LightGBM, and EDA):

```bash
python run_replication.py
```

### Pipeline stages

| Stage | Description |
|-------|-------------|
| `eda` | Exploratory plots saved to `results/eda/` |
| `examples` | Single-series examples from the paper → `results/examples/` |
| `benchmark` | 100 series per category (300 total), Table 3 comparison → `results/benchmark/` |
| `lightgbm-full` | LightGBM trained on all 30,490 series (needs ≥32 GB RAM) |
| `all` | Run everything (default) |

### Common options

```bash
# Quick smoke test (5 series per category, skips full LightGBM)
python run_replication.py --quick

# Run only the benchmark with selected models
python run_replication.py --stage benchmark --methods arima prophet

# LightGBM on benchmark subset (default, fits 16 GB RAM)
python run_replication.py --stage benchmark --methods lightgbm

# Paper-style LightGBM trained on all 30,490 series (high memory)
python run_replication.py --stage benchmark --methods lightgbm --lgb-train-scope full

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
├── results/                # Committed outputs (plots, RMSE, predictions)
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
└── outputs/                # Scratch cache (gitignored)
```

## Outputs

Runtime results are written directly to `results/` (tracked in git):

- `results/benchmark/benchmark_rmse_pivot.csv` — per-category RMSE by method
- `results/benchmark/benchmark_comparison.json` — side-by-side comparison with paper Table 3
- `results/benchmark/*_predictions.csv` — per-model forecast files
- `results/eda/*.png` — exploratory plots
- `results/logs/run_*.log` — logs when using `run_all_separate.sh`

## Reference

Paper targets (Table 3, mean per-series RMSE):

| Method | HOUSEHOLD | HOBBIES | FOODS | TOTAL |
|--------|-----------|---------|-------|-------|
| ARIMA | 0.837 | 0.965 | 1.494 | 1.099 |
| Facebook Prophet | 11.285 | 5.892 | 3.723 | 6.967 |
| LightGBM | 0.867 | 0.972 | 1.726 | 1.188 |
