"""CLI entry point — load a YAML config, optionally override fields, run training."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import yaml

# Make ``code/`` importable when running ``python code/main.py``
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Config, train  # noqa: E402  (after sys.path tweak)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_overrides(items: list[str]) -> Dict[str, Any]:
    """Parse ``key=value`` overrides, casting to the type already on Config."""
    out: Dict[str, Any] = {}
    defaults = Config()
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        if hasattr(defaults, k):
            current = getattr(defaults, k)
            if isinstance(current, bool):
                out[k] = v.lower() in ("1", "true", "yes", "y")
            elif isinstance(current, int):
                out[k] = int(v)
            elif isinstance(current, float):
                out[k] = float(v)
            else:
                out[k] = v
        else:
            out[k] = v  # forwarded but unused
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PatchTST or DLinear model.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config fields, e.g. --override pred_len=192 lr=5e-4",
    )
    args = parser.parse_args()

    cfg_dict = _load_yaml(args.config)
    cfg_dict.update(_parse_overrides(args.override))
    cfg = Config(**cfg_dict)
    train(cfg)


if __name__ == "__main__":
    main()
