from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from train.grpo import GroupedEpisodeBatch
from train.logging import log_debug_event
from train.trajectory import EpisodeRollout


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class CheckpointState(FrozenModel):
    checkpoint_id: str
    update_index: int
    step_counters: dict[str, int] = Field(default_factory=dict)
    model_reference: str | None = None
    optimizer_state: dict[str, Any] = Field(default_factory=dict)
    reward_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    train_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


def save_checkpoint(
    checkpoint_root: str | Path,
    *,
    state: CheckpointState,
    grouped_batch: GroupedEpisodeBatch | None = None,
    episodes: tuple[EpisodeRollout, ...] = (),
    logger: logging.Logger | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_root) / state.checkpoint_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "state.json").write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    if grouped_batch is not None:
        (checkpoint_dir / "group_batch.json").write_text(
            json.dumps(grouped_batch.batch.model_dump(mode="json"), indent=2, sort_keys=True)
        )
    if episodes:
        _write_episode_artifacts(checkpoint_dir, episodes)
    log_debug_event(
        logger,
        "checkpoint_saved",
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_id=state.checkpoint_id,
        update_index=state.update_index,
        num_episodes=len(episodes),
    )
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str | Path) -> CheckpointState:
    state_path = Path(checkpoint_dir) / "state.json"
    return CheckpointState.model_validate(json.loads(state_path.read_text()))


def _write_episode_artifacts(checkpoint_dir: Path, episodes: tuple[EpisodeRollout, ...]) -> None:
    ranked = [episode for episode in episodes if episode.outcome.final_reward is not None]
    ranked.sort(key=lambda episode: episode.outcome.final_reward, reverse=True)
    best = ranked[:2]
    worst = ranked[-2:] if ranked else []
    invalid = [episode for episode in episodes if any(step.validation_errors for step in episode.steps)][:2]
    samples = {
        "best_episodes": [episode.model_dump(mode="json") for episode in best],
        "worst_episodes": [episode.model_dump(mode="json") for episode in worst],
        "invalid_action_episodes": [episode.model_dump(mode="json") for episode in invalid],
    }
    (checkpoint_dir / "sample_artifacts.json").write_text(json.dumps(samples, indent=2, sort_keys=True))
