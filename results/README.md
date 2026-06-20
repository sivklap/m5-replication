# Research outputs

These files are **committed to git** so reviewers can inspect results without re-running the pipeline or downloading the M5 dataset (~763 MB).

## Layout

| Folder | Contents |
|--------|----------|
| [`eda/`](eda/) | Exploratory analysis plots (paper Section 3) |
| [`benchmark/`](benchmark/) | Table 3 style benchmark: RMSE summaries, predictions, comparison vs paper |
| [`examples/`](examples/) | Single-series example metrics from the paper |
| [`logs/`](logs/) | Run logs |

## Key files

- **`benchmark/benchmark_rmse_pivot.csv`** — per-category RMSE by method (ours)
- **`benchmark/benchmark_comparison.json`** — side-by-side comparison with paper Table 3
- **`benchmark/benchmark_series_ids.csv`** — fixed 300-series sample (100 per category, seed=4)
- **`examples/example_rmse.json`** — ARIMA / Prophet / LightGBM example-series RMSE

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # or use Nixtla mirror — see README
python run_replication.py --stage eda
python run_replication.py --stage examples
python run_replication.py --stage benchmark --lgb-train-scope subset
```

LightGBM defaults to training on the **benchmark subset** (fits in 16 GB RAM). For paper-style global training on all 30,490 series, use `--lgb-train-scope full` on a machine with ≥32 GB RAM.

## Paper reference

[arXiv:2203.06848](https://arxiv.org/abs/2203.06848) — Table 3 targets are listed in the root `README.md`.
