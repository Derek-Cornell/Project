#!/usr/bin/env bash
# Run all 8 experiments (PatchTST + DLinear, T in {96, 192, 336, 720}) sequentially.
# Assumes you are at the repo root and electricity.csv is in place.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

CONFIGS=(
  code/configs/electricity_patchtst_96.yaml
  code/configs/electricity_patchtst_192.yaml
  code/configs/electricity_patchtst_336.yaml
  code/configs/electricity_patchtst_720.yaml
  code/configs/electricity_dlinear_96.yaml
  code/configs/electricity_dlinear_192.yaml
  code/configs/electricity_dlinear_336.yaml
  code/configs/electricity_dlinear_720.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================"
  echo " Running $cfg"
  echo "================================================================"
  python code/main.py --config "$cfg"
done

python code/scripts/summarize.py
