#!/usr/bin/env bash
# Run each replication stage separately and save logs + results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="$ROOT/.matplotlib"
export PYTHONUNBUFFERED=1
PY="${PY:-python3}"
mkdir -p "$ROOT/results/benchmark" "$ROOT/results/logs" "$MPLCONFIGDIR"

run_stage() {
  local name="$1"
  shift
  echo "========== $name =========="
  "$PY" run_replication.py "$@" 2>&1 | tee "$ROOT/results/logs/run_${name}.log"
}

run_stage eda --stage eda
run_stage examples --stage examples
run_stage arima --stage benchmark --methods arima --per-category 100 --arima-jobs 4
run_stage prophet --stage benchmark --methods prophet --per-category 100
run_stage lightgbm --stage benchmark --methods lightgbm --per-category 100 --lgb-train-scope subset

"$PY" scripts/combine_results.py 2>&1 | tee "$ROOT/results/logs/run_combine.log"
echo "Done. Results in $ROOT/results/"
