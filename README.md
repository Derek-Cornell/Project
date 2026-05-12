# Patches &gt; Linears? Reproducing PatchTST on the ILI Benchmark

Cornell CS 4782 (Spring 2026) final project · Cade Jin · Chenkai Shen · Derek Xu

## 1. Introduction

This repo reproduces **PatchTST** (Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023) and pits it against the **DLinear** baseline (Zeng et al., AAAI 2023). PatchTST's two ideas — segmenting each series into short overlapping **patches** as tokens, and treating each channel **independently** so one Transformer is shared across all channels — were claimed to flip the script after Zeng et al. showed a single linear layer matched Transformer baselines. We test whether that claim survives an independent re-implementation on the failure-prone CDC influenza-like-illness (ILI) dataset.

## 2. Chosen Result

The ILI rows of **Table 3** in Nie et al.: supervised PatchTST/42 vs.\ DLinear at horizons T ∈ {24, 36, 48, 60} weeks with look-back L = 104. This is the smallest dataset in the benchmark (966 weekly rows × 7 channels) and the one where DLinear's claim is most relevant — a tiny model on tiny data is exactly where a Transformer would be most expected to over-fit.

## 3. GitHub Contents

```
code/        Model + training code (see CLAUDE.md for architecture notes)
  configs/   Per-experiment YAML configs (illness_*, electricity_*)
  models/    PatchTST and DLinear implementations
  scripts/   smoke_test, test_patchtst_ar, summarize, plot_history, run_all.sh
data/        Place datasets here — not committed (see §5)
notebooks/   illness_runner.ipynb (headline track), colab_runner.ipynb (Electricity)
results/cs4782_illness_results/   Committed multi-seed + AR-vs-direct results
poster/poster.pdf                  Class poster
report/                            2-page report (LaTeX source; compile with pdflatex/Overleaf)
```

## 4. Re-implementation Details

- **PatchTST** ([code/models/patchtst.py](code/models/patchtst.py)) — channel-independence (M channels folded into the batch dim, one shared backbone), patching with P = 24 and S = 2, 3-layer Transformer encoder using **BatchNorm** (paper footnote 1, citing Zerveas et al. 2021), residual attention, d_model = 16, H = 4 heads, d_ff = 128, dropout 0.3, and RevIN (Kim et al. 2022) with `affine=False` wrapping the model in normalize/denormalize calls.
- **DLinear** ([code/models/dlinear.py](code/models/dlinear.py)) — moving-average trend/seasonal decomposition with a per-channel `Linear(L → T)` for each component, summed.
- **Training** — Adam, MSE, batch 16, lr 2.5e-3 (constant), 100 epochs, val-MSE early-stop checkpoint. Full ILI sweep + ablation completes in &lt; 30 min on a single T4 GPU.
- **Beyond the paper** — (i) PatchTST run on 3 seeds {1, 42, 2021} and reported as mean ± std; (ii) an autoregressive forecasting head (`forecasting_mode=autoregressive`) that trains a single T = 24 model and rolls it forward for longer horizons. The AR path is regression-tested by [code/scripts/test_patchtst_ar.py](code/scripts/test_patchtst_ar.py).

## 5. Reproduction Steps

**Data.** Both datasets ship in the standard Autoformer Google Drive bundle:
https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

- ILI (required for the headline track): place `national_illness.csv` at `data/illness/national_illness.csv` — 966 weekly rows × 7 numeric columns + a `date` column.
- Electricity (optional): place `electricity.csv` at `data/electricity/electricity.csv` — 26 304 hourly rows × 321 numeric columns + a `date` column.

**Run.** From the repo root:

```bash
pip install -r code/requirements.txt
python code/scripts/smoke_test.py                # sanity check
python code/scripts/test_patchtst_ar.py          # AR-head regression test

# One ILI run
python code/main.py --config code/configs/illness_patchtst_24.yaml

# Multi-seed PatchTST sweep + AR ablation
jupyter notebook notebooks/illness_runner.ipynb

# Full Electricity sweep (exploratory)
bash code/scripts/run_all.sh

# Aggregate results/*.json into results/summary.csv vs.\ paper Table 3
python code/scripts/summarize.py
```

Override any `Config` field from the CLI, e.g. `--override pred_len=36 seed=42 forecasting_mode=autoregressive`.

**Compute.** Single T4 GPU or a modern CPU is sufficient for the ILI track. Electricity is heavier but still fits on a single T4.

