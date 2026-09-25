#!/bin/bash
# One command for a CUDA machine (Colab T4, rented 3090/4090, ...).
#   pip install torch   (already present on Colab)
#   bash run_gpu.sh            -> trains 4 arms x 3 seeds in parallel, then evaluates
# Results: runs_gpu/*.log, runs_gpu/final_eval.json, printed table at the end.
set -e
cd "$(dirname "$0")"
mkdir -p runs_gpu
ITERS=${ITERS:-1500}
for seed in 0 1 2; do
  for arm in indep visible unsigned signed; do
    python3 -u train.py --arm $arm --iters $ITERS --bc-iters 0 --refute-p 0.25 --seed $seed \
      --device cuda --out runs_gpu/${arm}_s${seed}.json > runs_gpu/${arm}_s${seed}.log 2>&1 &
  done
done
wait
DEVICE=cuda python3 final_eval.py runs_gpu
