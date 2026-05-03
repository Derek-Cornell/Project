"""Training and evaluation entry point.

Run a single (model, horizon) experiment. Use ``main.py`` to drive this with a YAML config, or
invoke ``train(...)`` directly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider import build_dataloaders
from models import DLinear, PatchTST
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, set_seed


@dataclass
class Config:
    # data
    data_path: str = "data/electricity/electricity.csv"
    seq_len: int = 336
    pred_len: int = 96

    # model
    model: str = "patchtst"            # "patchtst" or "dlinear"
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    head_dropout: float = 0.0
    revin: bool = True
    forecasting_mode: str = "direct"   # PatchTST only: "direct" | "autoregressive"
    kernel_size: int = 25              # DLinear only
    individual: bool = True            # DLinear only

    # training
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 100
    patience: int = 10
    lr_schedule: str = "type1"
    num_workers: int = 4
    seed: int = 2021

    # bookkeeping
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    run_name: str = "patchtst_96"
    extra: Dict[str, Any] = field(default_factory=dict)


def _build_model(cfg: Config, c_in: int) -> nn.Module:
    if cfg.model == "patchtst":
        return PatchTST(
            c_in=c_in,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
            revin=cfg.revin,
            forecasting_mode=cfg.forecasting_mode,
        )
    if cfg.model == "dlinear":
        return DLinear(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            c_in=c_in,
            kernel_size=cfg.kernel_size,
            individual=cfg.individual,
        )
    raise ValueError(f"Unknown model {cfg.model!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Dict[str, float]:
    model.eval()
    losses = []
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss.item())
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    out = metric(preds, trues)
    out["loss"] = float(np.mean(losses))
    return out


def train(cfg: Config) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] run={cfg.run_name} device={device}")

    loaders, c_in = build_dataloaders(
        csv_path=cfg.data_path,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"[train] features={c_in} train_steps={len(loaders['train'])}")

    model = _build_model(cfg, c_in).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model={cfg.model} trainable_params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}.pt")
    stopper = EarlyStopping(patience=cfg.patience)

    history = []
    epoch_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tic = time.time()
        train_losses = []
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch:03d}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_metrics = evaluate(model, loaders["val"], device, criterion)
        lr_now = adjust_learning_rate(optimizer, epoch + 1, cfg.lr, schedule=cfg.lr_schedule)
        elapsed = time.time() - tic
        print(
            f"[train] epoch {epoch:03d} | "
            f"train_mse {np.mean(train_losses):.4f} | "
            f"val_mse {val_metrics['mse']:.4f} | val_mae {val_metrics['mae']:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "lr": lr_now,
            }
        )

        stopper(val_metrics["mse"], model, ckpt_path)
        if stopper.early_stop:
            print(f"[train] early stopping at epoch {epoch}")
            break

    # Reload best checkpoint and evaluate on test
    print(f"[train] loading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loaders["test"], device, criterion)
    total_time = time.time() - epoch_start
    print(
        f"[train] TEST | mse {test_metrics['mse']:.4f} | mae {test_metrics['mae']:.4f} | "
        f"total {total_time / 60:.1f} min"
    )

    os.makedirs(cfg.results_dir, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "test": test_metrics,
        "best_val_mse": stopper.best_score,
        "history": history,
        "trainable_params": n_params,
        "wall_clock_sec": total_time,
    }
    out_path = os.path.join(cfg.results_dir, f"{cfg.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[train] wrote {out_path}")
    return payload
