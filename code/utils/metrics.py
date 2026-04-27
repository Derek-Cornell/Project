"""Forecasting metrics. Operate on numpy arrays of arbitrary shape; mean is taken over all elements."""

import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def metric(pred: np.ndarray, true: np.ndarray) -> dict:
    return {"mse": mse(pred, true), "mae": mae(pred, true)}
