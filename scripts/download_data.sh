#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/m5"
mkdir -p "$DATA"
if ! command -v kaggle &>/dev/null; then
  echo "Install Kaggle CLI: pip install kaggle"
  echo "Then place API token at ~/.kaggle/kaggle.json"
  exit 1
fi
kaggle competitions download -c m5-forecasting-accuracy -p "$DATA"
unzip -o "$DATA/m5-forecasting-accuracy.zip" -d "$DATA"
echo "M5 data ready in $DATA"
