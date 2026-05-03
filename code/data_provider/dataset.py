"""Electricity dataset loader.

Matches the Informer/Autoformer/PatchTST 70/10/20 chronological split. The first 70 % of the
time series is used for training (and for fitting the per-feature StandardScaler); the next 10 %
is the validation set and the last 20 % is the test set. Validation and test windows include a
look-back-length overlap into the previous segment so the first sample in each set is well-formed.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


_FLAG_TO_INDEX = {"train": 0, "val": 1, "test": 2}


class ElectricityDataset(Dataset):
    """Sliding-window multivariate dataset over a single CSV.

    Each item is a pair ``(x, y)`` of shapes ``(seq_len, M)`` and ``(pred_len, M)``.
    """

    def __init__(
        self,
        csv_path: str,
        flag: str,
        seq_len: int,
        pred_len: int,
        scale: bool = True,
    ):
        if flag not in _FLAG_TO_INDEX:
            raise ValueError(f"flag must be one of {list(_FLAG_TO_INDEX)}, got {flag!r}")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Could not find {csv_path}. See data/README.md for download instructions."
            )

        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.lower() != "date"]
        data = df[cols].values.astype(np.float32)  # (T_total, M)

        n = len(data)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test

        # Use the standard overlap convention so each split's first sample is valid.
        borders1 = [0, n_train - seq_len, n - n_test - seq_len]
        borders2 = [n_train, n_train + n_val, n]

        self.scaler = StandardScaler()
        if scale:
            self.scaler.fit(data[borders1[0] : borders2[0]])
            data = self.scaler.transform(data).astype(np.float32)

        idx = _FLAG_TO_INDEX[flag]
        self.data = data[borders1[idx] : borders2[idx]]
        self.num_features = self.data.shape[-1]

        if len(self.data) < seq_len + pred_len:
            raise ValueError(
                f"Split {flag!r} has {len(self.data)} steps, need at least "
                f"{seq_len + pred_len} for one window."
            )

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(arr)


def build_dataloaders(
    csv_path: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[Dict[str, DataLoader], int]:
    """Build train/val/test DataLoaders. Returns (loaders, num_features)."""
    sets = {
        flag: ElectricityDataset(csv_path, flag, seq_len, pred_len)
        for flag in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(
            sets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        ),
        "val": DataLoader(
            sets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,  # matches reference data_provider/data_factory.py
            pin_memory=True,
        ),
    }
    return loaders, sets["train"].num_features
