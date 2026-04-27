"""Misc training utilities: seeding, early stopping, learning-rate adjustment."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    counter: int = 0
    best_score: float = float("inf")
    early_stop: bool = False

    def __call__(self, val_loss: float, model: torch.nn.Module, ckpt_path: str) -> None:
        if val_loss + self.min_delta < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def adjust_learning_rate(optimizer, epoch: int, initial_lr: float, schedule: str = "type1") -> float:
    """Match the LR schedules used by the official PatchTST codebase.

    type1  — halve LR every epoch after epoch 2 (used for the long-horizon supervised runs)
    type3  — keep LR constant for 3 epochs, then decay 0.9x per epoch (DLinear-style)
    constant — no change
    """
    if schedule == "type1":
        lr = initial_lr * (0.5 ** ((epoch - 1) // 1)) if epoch > 2 else initial_lr
    elif schedule == "type3":
        lr = initial_lr if epoch < 3 else initial_lr * (0.9 ** ((epoch - 3) // 1))
    else:
        lr = initial_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr
