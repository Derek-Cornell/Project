"""Aggregate per-run JSONs in results/ into a single CSV side-by-side with the paper's numbers."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, Tuple

# Paper Table 3 reference values, keyed by (model, dataset, horizon).
# - Electricity: PatchTST/42 with seq_len=336 (Nie et al. 2023).
# - Illness: PatchTST/64 with seq_len=104 (Nie et al. 2023, "ILI" dataset).
PAPER: Dict[Tuple[str, str, int], Dict[str, float]] = {
    # Electricity
    ("patchtst", "electricity", 96):  {"mse": 0.130, "mae": 0.222},
    ("patchtst", "electricity", 192): {"mse": 0.148, "mae": 0.240},
    ("patchtst", "electricity", 336): {"mse": 0.167, "mae": 0.261},
    ("patchtst", "electricity", 720): {"mse": 0.202, "mae": 0.291},
    ("dlinear",  "electricity", 96):  {"mse": 0.140, "mae": 0.237},
    ("dlinear",  "electricity", 192): {"mse": 0.153, "mae": 0.249},
    ("dlinear",  "electricity", 336): {"mse": 0.169, "mae": 0.267},
    ("dlinear",  "electricity", 720): {"mse": 0.203, "mae": 0.301},
    # Illness (ILI)
    ("patchtst", "illness", 24): {"mse": 1.319, "mae": 0.754},
    ("patchtst", "illness", 36): {"mse": 1.430, "mae": 0.834},
    ("patchtst", "illness", 48): {"mse": 1.553, "mae": 0.815},
    ("patchtst", "illness", 60): {"mse": 1.470, "mae": 0.788},
    ("dlinear",  "illness", 24): {"mse": 2.215, "mae": 1.081},
    ("dlinear",  "illness", 36): {"mse": 1.963, "mae": 0.963},
    ("dlinear",  "illness", 48): {"mse": 2.130, "mae": 1.024},
    ("dlinear",  "illness", 60): {"mse": 2.368, "mae": 1.096},
}


def _infer_dataset(cfg: dict) -> str:
    """Best-effort dataset name from data_path or run_name."""
    data_path = cfg.get("data_path", "") or ""
    run_name = cfg.get("run_name", "") or ""
    blob = (data_path + " " + run_name).lower()
    for tag in ("electricity", "illness", "ili", "weather", "traffic", "etth1", "etth2", "ettm1", "ettm2"):
        if tag in blob:
            return "illness" if tag == "ili" else tag
    return "unknown"


def main() -> None:
    rows = []
    for path in sorted(glob.glob("results/*.json")):
        with open(path) as f:
            data = json.load(f)
        cfg = data["config"]
        model = cfg["model"]
        horizon = cfg["pred_len"]
        dataset = _infer_dataset(cfg)
        ours = data["test"]
        ref = PAPER.get((model, dataset, horizon), {"mse": float("nan"), "mae": float("nan")})
        ours_mse = ours.get("mse", float("nan"))
        ours_mae = ours.get("mae", float("nan"))
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "run_name": cfg.get("run_name", ""),
                "ours_mse": round(ours_mse, 4) if ours_mse == ours_mse else float("nan"),
                "paper_mse": ref["mse"],
                "delta_mse": round(ours_mse - ref["mse"], 4) if (ours_mse == ours_mse and ref["mse"] == ref["mse"]) else float("nan"),
                "ours_mae": round(ours_mae, 4) if ours_mae == ours_mae else float("nan"),
                "paper_mae": ref["mae"],
                "delta_mae": round(ours_mae - ref["mae"], 4) if (ours_mae == ours_mae and ref["mae"] == ref["mae"]) else float("nan"),
                "wall_clock_min": round(data.get("wall_clock_sec", 0) / 60, 1),
            }
        )

    if not rows:
        print("No JSON results found in results/. Run an experiment first.")
        return

    rows.sort(key=lambda r: (r["dataset"], r["model"], r["horizon"], r["run_name"]))
    out_path = os.path.join("results", "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Pretty print
    print(f"Wrote {out_path}\n")
    header = list(rows[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in header]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(w) for h, w in zip(header, widths)))


if __name__ == "__main__":
    main()
