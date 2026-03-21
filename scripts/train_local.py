#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import run_training_from_config_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Hugging Face Mafia training.")
    parser.add_argument("--config", default=str(REPO_ROOT / "train.yaml"), help="Path to train.yaml")
    parser.add_argument(
        "--checkpoint-root",
        default=str(REPO_ROOT / "checkpoints" / "local"),
        help="Directory where checkpoints will be written",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_training_from_config_path(
        config_path=args.config,
        checkpoint_root=args.checkpoint_root,
        seed=args.seed,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
