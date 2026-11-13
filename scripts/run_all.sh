#!/usr/bin/env bash
# Full pipeline: data -> benchmark -> fine-tune compare -> agent eval -> plots -> summary.
# Usage: scripts/run_all.sh [--fast]
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/forecastmcp ]]; then
  FC=".venv/bin/forecastmcp"
else
  FC="forecastmcp"
fi

FAST=0
if [[ "${1:-}" == "--fast" ]]; then
  FAST=1
fi
OFFLINE="${FORECASTMCP_OFFLINE:-0}"
RUN_ID="${FORECASTMCP_BENCH_RUN_ID:-benchmark}"

echo "[run_all] preparing data (offline=${OFFLINE})"
"$FC" data prepare --dataset synthetic_supply
if [[ "$OFFLINE" != "1" ]]; then
  "$FC" data prepare --dataset m4_daily || echo "[run_all] m4_daily unavailable; continuing"
fi

if [[ "$FAST" == "1" ]]; then
  echo "[run_all] fast benchmark"
  "$FC" bench --run-id "$RUN_ID" --datasets synthetic_supply \
    --models seasonal_naive,autoets --series 20 --windows 6 --resamples 200 --fast
else
  echo "[run_all] full benchmark"
  "$FC" bench --run-id "$RUN_ID" --series 50 --windows 6 --resamples 1000
fi

if [[ "$OFFLINE" != "1" && -f artifacts/finetune/v1/model_card.json && "$FAST" != "1" ]]; then
  echo "[run_all] zero-shot vs LoRA compare"
  "$FC" finetune compare --tag v1 --series 20 --windows 6
fi

if [[ "$FAST" == "1" ]]; then
  echo "[run_all] agent harness evaluation (fast)"
  "$FC" eval agents --trials 1 --limit 3 --prefix agenteval-fast --benchmark-run "$RUN_ID"
else
  echo "[run_all] agent harness evaluation"
  "$FC" eval agents --trials 3 --prefix agenteval --benchmark-run "$RUN_ID"
fi

echo "[run_all] plots"
"$FC" eval plots
echo "[run_all] summary"
"$FC" eval summary

echo "[run_all] done"
echo "  results/summary.json"
echo "  results/summary.md"
echo "  results/NEGATIVE_RESULTS.md"
echo "  results/figures/"
