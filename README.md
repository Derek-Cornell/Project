# Data

We use the **Electricity** (ECL) dataset, also known as Electricity Load Diagrams 2011–2014 from the
UCI Machine Learning Repository. The standard pre-processed CSV used by every Transformer
time-series forecasting paper (Informer, Autoformer, FEDformer, DLinear, PatchTST) lives in the
Autoformer release bundle.

## Download

Place `electricity.csv` at `data/electricity/electricity.csv`. There are two convenient sources:

### Option A — Autoformer Google Drive bundle (recommended, matches paper)

```bash
mkdir -p data/electricity
# Manual: visit
#   https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
# and download "electricity.csv" into data/electricity/
```

This is the same file used by the original PatchTST authors, so train/val/test splits match
exactly.

### Option B — UCI raw ZIP (will require pre-processing yourself)

```bash
mkdir -p data/electricity
wget -O data/electricity/raw.zip \
  https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip data/electricity/raw.zip -d data/electricity/
# You then need to (a) resample 15-min cadence to hourly,
# (b) drop the first ~1 year of mostly-zero readings, and
# (c) write out a CSV with a `date` column and 321 numeric columns.
```

The Colab notebook (`notebooks/colab_runner.ipynb`) handles Option A automatically using the
public mirror that ships with the Autoformer repo.

## Expected file format

| date                | MT_001 | MT_002 | ... | MT_320 | OT     |
|---------------------|--------|--------|-----|--------|--------|
| 2016-07-01 00:00:00 | 14.0   | 69.0   | ... | 69.0   | 0.7   |
| 2016-07-01 01:00:00 | 18.0   | 92.0   | ... | 79.0   | 1.0   |

- 26,304 hourly rows.
- 321 numeric columns (the data loader uses **all** of them as features in the multivariate
  setting; the `OT` column is only special for univariate forecasting, which we are not running).
- A `date` column that the loader drops.
