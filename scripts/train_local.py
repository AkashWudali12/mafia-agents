#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import load_dotenv, load_train_config, run_training_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Hugging Face Mafia training.")
    parser.add_argument("--config", default=str(REPO_ROOT / "train.yaml"), help="Path to train.yaml")
    parser.add_argument(
        "--checkpoint-root",
        default=str(REPO_ROOT / "checkpoints" / "local"),
        help="Directory where checkpoints will be written",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the local browser-based game viewer",
    )
    parser.add_argument("--ui-host", default=None, help="Host for the local viewer server")
    parser.add_argument("--ui-port", type=int, default=None, help="Port for the local viewer server")
    parser.add_argument(
        "--tts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable ElevenLabs TTS (API + voice ids in .env). Can be used with or without --ui.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    config = load_train_config(args.config)
    if args.ui is not None or args.ui_host is not None or args.ui_port is not None or args.tts is not None:
        viewer_config = config.viewer.model_copy(
            update={
                "enabled": config.viewer.enabled if args.ui is None else args.ui,
                "host": config.viewer.host if args.ui_host is None else args.ui_host,
                "port": config.viewer.port if args.ui_port is None else args.ui_port,
                "tts_enabled": config.viewer.tts_enabled if args.tts is None else args.tts,
            }
        )
        config = config.model_copy(update={"viewer": viewer_config})
    summary = run_training_from_config(
        config=config,
        checkpoint_root=args.checkpoint_root,
        seed=args.seed,
        viewer_started_callback=_open_viewer_browser if config.viewer.enabled else None,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _open_viewer_browser(url: str) -> None:
    webbrowser.open(url, new=2, autoraise=True)


if __name__ == "__main__":
    raise SystemExit(main())
