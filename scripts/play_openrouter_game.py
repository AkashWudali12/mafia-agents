#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import webbrowser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from game import new_game  # noqa: E402
from policies import OpenRouterPolicy, PydanticAiOpenRouterClient  # noqa: E402
from train import LiveTrainingViewer, load_dotenv, load_train_config, run_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Mafia game with all seats controlled by OpenRouter models.")
    parser.add_argument("--config", default=str(REPO_ROOT / "train.yaml"), help="Path to train.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="OpenRouter model to include in the seat assignment pool. Repeat to provide multiple models.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the local browser-based game viewer",
    )
    parser.add_argument("--ui-host", default="127.0.0.1", help="Host for the local viewer server")
    parser.add_argument("--ui-port", type=int, default=8765, help="Port for the local viewer server")
    parser.add_argument(
        "--linger-seconds",
        type=int,
        default=120,
        help="How long to keep the UI server alive after the game finishes. Use 0 to exit immediately.",
    )
    parser.add_argument(
        "--featured-seat",
        type=int,
        default=0,
        help="Seat index to highlight in the viewer and metadata summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required to run an OpenRouter-only local game")

    config = load_train_config(args.config)
    seed = args.seed if args.seed is not None else config.project.seed
    state = new_game(config=config.environment, seed=seed)

    if args.featured_seat < 0 or args.featured_seat >= state.config.num_players:
        raise ValueError(f"featured seat must be between 0 and {state.config.num_players - 1}")

    model_pool = tuple(args.models) if args.models else config.opponents.model_names
    if not model_pool:
        raise ValueError("provide at least one OpenRouter model via --model or train.yaml opponents.model_names")

    rng = random.Random(seed)
    client = PydanticAiOpenRouterClient()
    assigned_models = tuple(rng.choice(model_pool) for _ in range(state.config.num_players))
    seat_policies = {
        seat: OpenRouterPolicy(
            client=client,
            model=model_name,
            temperature=config.model.temperature,
            max_tokens=config.model.max_new_tokens,
        )
        for seat, model_name in enumerate(assigned_models)
    }

    viewer = None
    viewer_url = None
    if args.ui:
        viewer = LiveTrainingViewer(host=args.ui_host, port=args.ui_port, max_cached_events=config.viewer.max_cached_events)
        viewer_url = viewer.start()
        webbrowser.open(viewer_url, new=2, autoraise=True)

    try:
        rollout = run_episode(
            trainable_seat=args.featured_seat,
            seat_policies=seat_policies,
            metadata_update={
                "checkpoint_id": "openrouter_local_game",
                "opponent_pool_id": "local_openrouter_game",
                "opponent_model_names": assigned_models,
                "opponent_prompt_version": config.opponents.prompt_version,
                "opponent_cache_behavior": config.opponents.cache_behavior,
            },
            update_index=0,
            seed=seed,
            initial_state=state,
            viewer=viewer,
        )

        summary = {
            "viewer_url": viewer_url,
            "episode_id": rollout.metadata.episode_id,
            "seed": rollout.metadata.seed,
            "featured_seat": rollout.metadata.trainable_seat,
            "featured_role": rollout.metadata.trainable_role.value,
            "winner": rollout.outcome.winner.value if rollout.outcome.winner is not None else None,
            "days_reached": rollout.outcome.num_days_reached,
            "assigned_models": assigned_models,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

        if viewer is not None and args.linger_seconds > 0:
            _linger_with_viewer(args.linger_seconds)
        return 0
    finally:
        if viewer is not None:
            viewer.stop()


def _linger_with_viewer(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.25, remaining))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    raise SystemExit(main())
