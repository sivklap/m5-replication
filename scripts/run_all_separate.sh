#!/usr/bin/env bash
# Run each replication stage separately and save logs + results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="$ROOT/.matplotlib"
export PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin/python"
RESULTS="$ROOT/outputs/results"
mkdir -p "$RESULTS" "$MPLCONFIGDIR"

run_stage() {
  local name="$1"
  shift
  echo "========== $name =========="
  "$PY" run_replication.py "$@" 2>&1 | tee "$RESULTS/run_${name}.log"
}

# Fixed 300-series benchmark sample (seed=42)
"$PY" -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.load_data import sample_series_ids
ids = sample_series_ids(per_category=100)
pd.DataFrame({'id': ids}).to_csv('$RESULTS/benchmark_series_ids.csv', index=False)
print('Benchmark series:', len(ids))
"

run_stage examples --stage examples
run_stage arima --stage benchmark --methods arima --per-category 100 --arima-jobs 4
run_stage prophet --stage benchmark --methods prophet --per-category 100
run_stage lightgbm --stage benchmark --methods lightgbm --per-category 100 --lgb-train-scope full

"$PY" scripts/combine_results.py 2>&1 | tee "$RESULTS/run_combine.log"
echo "Done. Results in $RESULTS"
