"""Aggregate per-run JSONs in results/ into a single CSV side-by-side with the paper's numbers."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, Tuple

# Paper Table 3 (Electricity, supervised PatchTST/42 with seq_len=336, and DLinear).
PAPER: Dict[Tuple[str, int], Dict[str, float]] = {
    ("patchtst", 96):  {"mse": 0.130, "mae": 0.222},
    ("patchtst", 192): {"mse": 0.148, "mae": 0.240},
    ("patchtst", 336): {"mse": 0.167, "mae": 0.261},
    ("patchtst", 720): {"mse": 0.202, "mae": 0.291},
    ("dlinear", 96):   {"mse": 0.140, "mae": 0.237},
    ("dlinear", 192):  {"mse": 0.153, "mae": 0.249},
    ("dlinear", 336):  {"mse": 0.169, "mae": 0.267},
    ("dlinear", 720):  {"mse": 0.203, "mae": 0.301},
}


def main() -> None:
    rows = []
    for path in sorted(glob.glob("results/*.json")):
        with open(path) as f:
            data = json.load(f)
        cfg = data["config"]
        model = cfg["model"]
        horizon = cfg["pred_len"]
        ours = data["test"]
        ref = PAPER.get((model, horizon), {"mse": float("nan"), "mae": float("nan")})
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "ours_mse": round(ours["mse"], 4),
                "paper_mse": ref["mse"],
                "delta_mse": round(ours["mse"] - ref["mse"], 4),
                "ours_mae": round(ours["mae"], 4),
                "paper_mae": ref["mae"],
                "delta_mae": round(ours["mae"] - ref["mae"], 4),
                "wall_clock_min": round(data["wall_clock_sec"] / 60, 1),
            }
        )

    if not rows:
        print("No JSON results found in results/. Run an experiment first.")
        return

    rows.sort(key=lambda r: (r["model"], r["horizon"]))
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
