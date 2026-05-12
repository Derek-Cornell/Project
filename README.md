# Reproducing PatchTST on the ILI Benchmark

Cade Jin · Chenkai Shen · Derek Xu

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

We recommend running everything on **Google Colab with a GPU runtime**. A free T4 GPU is sufficient for the entire ILI track.

**Step 1 — Open a Colab notebook with a GPU**

Go to [colab.research.google.com](https://colab.research.google.com), create a new notebook, then `Runtime → Change runtime type → Hardware accelerator: T4 GPU` (free) or `A100 GPU` (Pro). Confirm with:

```python
import torch
print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
```

**Step 2 — Get the dataset onto Colab**

The ILI dataset (`national_illness.csv`) ships in the standard Autoformer Google Drive bundle: [https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Download `national_illness.csv` to your own Google Drive, then mount it in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data/illness
!cp '/content/drive/MyDrive/national_illness.csv' data/illness/national_illness.csv
```

**Step 3 — Get the code onto Colab.** Pick one of:

- **Option A: clone via Personal Access Token (PAT)** — if you have a PAT for this repo, paste:

  ```python
  REPO_URL = 'https://<USERNAME>:<YOUR_PAT>@github.com/<OWNER>/<REPO>.git'
  !git clone $REPO_URL /content/Project
  %cd /content/Project
  !pip install -q -r code/requirements.txt
  ```

  Generate a PAT at [github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope.

- **Option B: paste the notebook directly** — open `notebooks/illness_runner.ipynb` in this repo, copy each cell into your Colab notebook, and run top-to-bottom. The notebook is self-contained: it installs dependencies, sanity-checks the data loader, and runs the full multi-seed + AR ablation + per-step decomposition with one click.

**Step 4 — Run the experiments**

If you cloned the repo (Option A), launch `notebooks/illness_runner.ipynb` in Colab. If you pasted the notebook (Option B), just run all cells. Either way you'll get:

- All 4 horizons × 2 modes × 3 seeds (24 runs) for PatchTST direct and AR.
- DLinear baseline for all 4 horizons.
- Aggregate test MSE table vs. paper.
- Per-step MSE decomposition plot with rollout-boundary annotations.

**Override any `Config` field from the CLI**, e.g. `--override pred_len=36 seed=42 forecasting_mode=autoregressive`. The notebook does this internally to sweep horizons, seeds, and modes.

**Local fallback.** If you'd rather run locally, the same notebook runs unchanged on Jupyter as long as you have a CUDA-capable GPU. From the repo root:

```bash
pip install -r code/requirements.txt
python code/scripts/smoke_test.py            # sanity check
python code/scripts/test_patchtst_ar.py      # AR-head regression test
python code/main.py --config code/configs/illness_patchtst_24.yaml   # single run
jupyter notebook notebooks/illness_runner.ipynb                       # full sweep
python code/scripts/summarize.py             # aggregate results/*.json → results/summary.csv
```

