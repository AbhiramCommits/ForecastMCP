#!/usr/bin/env bash
# Reproduce the committed results from a clean checkout with pinned seeds.
# Optional: SKIP_FINETUNE=1 to reuse the committed adapter.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/forecastmcp ]]; then
  FC=".venv/bin/forecastmcp"
else
  FC="forecastmcp"
fi

export FORECASTMCP_SEED=42
export PYTHONHASHSEED=0

echo "[reproduce] seeds: FORECASTMCP_SEED=$FORECASTMCP_SEED PYTHONHASHSEED=$PYTHONHASHSEED"

"$FC" data prepare --dataset synthetic_supply
"$FC" data prepare --dataset m4_daily

if [[ "${SKIP_FINETUNE:-0}" != "1" ]]; then
  "$FC" finetune --dataset synthetic_supply --tag v1 --config configs/finetune.yaml
fi

"$FC" bench --run-id benchmark --series 50 --windows 6 --resamples 1000
"$FC" finetune compare --tag v1 --series 20 --windows 6
"$FC" eval agents --trials 3 --prefix agenteval --benchmark-run benchmark
"$FC" eval plots
"$FC" eval summary

echo "[reproduce] done: results/summary.json, results/summary.md, results/NEGATIVE_RESULTS.md"
