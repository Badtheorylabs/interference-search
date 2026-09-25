#!/usr/bin/env bash
# Reproduce every Countdown result without a language model: CPU only, any OS PyTorch runs on.
# Fresh outputs go to repro/, then get compared against results/countdown.
#   bash scripts/reproduce_countdown.sh           about 10 minutes on a laptop
#   RETRAIN=1 bash scripts/reproduce_countdown.sh also retrains the judge (adds roughly an hour on CPU)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$PWD/repro"
mkdir -p "$OUT"

python -m pytest -q

cd experiments/countdown
python compression.py | tee "$OUT/compression.txt"
python benchmark.py --out "$OUT/countdown_benchmark.json"
python frontier_vs_linear.py --ablation --out "$OUT/frontier_vs_linear.json"
if [ "${RETRAIN:-0}" = 1 ]; then
  python train_judge.py --out "$OUT/countdown_judge_retrained.pt" | tee "$OUT/train_judge.txt"
fi
cd ../..

python scripts/compare_results.py "$OUT"
