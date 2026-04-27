"""Plot per-run training/validation curves from results/*.json."""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt


def main() -> None:
    paths = sorted(glob.glob("results/*.json"))
    if not paths:
        print("No JSON results in results/.")
        return

    os.makedirs("results/plots", exist_ok=True)
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        history = data.get("history", [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        train = [h["train_loss"] for h in history]
        val_mse = [h["val_mse"] for h in history]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train, label="train MSE")
        ax.plot(epochs, val_mse, label="val MSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_title(data["config"]["run_name"])
        ax.legend()
        ax.grid(alpha=0.3)
        out = os.path.join("results/plots", f"{data['config']['run_name']}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
