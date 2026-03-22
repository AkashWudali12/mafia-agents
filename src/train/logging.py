from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from contracts import Role

if TYPE_CHECKING:
    from train.grpo import GroupedEpisodeBatch
    from train.trajectory import EpisodeRollout


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class EpisodeLogSummary(FrozenModel):
    episode_id: str
    trainable_role: Role
    winner: str | None
    final_reward: float | None
    invalid_action_count: int
    transcript_length: int


class RunLogSummary(FrozenModel):
    total_episodes: int
    total_updates: int
    mean_reward: float
    win_rate: float
    invalid_action_rate: float
    reward_by_role: dict[str, float]
    grouped_reward_mean: float | None = None


def configure_training_logger(
    *,
    log_dir: str | Path,
    run_name: str,
    level: str = "DEBUG",
) -> logging.Logger:
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = f"train.{run_name}"
    logger = logging.getLogger(logger_name)
    truthfulqa_scores_path = resolved_log_dir / "truthfulqa_scores.jsonl"
    setattr(logger, "_truthfulqa_scores_path", truthfulqa_scores_path)
    truthfulqa_scores_path.touch(exist_ok=True)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    text_handler = logging.FileHandler(resolved_log_dir / "training.log")
    text_handler.setLevel(logger.level)
    text_handler.setFormatter(formatter)
    logger.addHandler(text_handler)

    event_handler = logging.FileHandler(resolved_log_dir / "events.jsonl")
    event_handler.setLevel(logger.level)
    event_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(event_handler)
    return logger


def log_debug_event(logger: logging.Logger | None, event: str, **payload: Any) -> None:
    if logger is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **_serialize_payload(payload),
    }
    logger.debug("%s", json.dumps(record, sort_keys=True, default=str))


def log_error_event(logger: logging.Logger | None, event: str, **payload: Any) -> None:
    if logger is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **_serialize_payload(payload),
    }
    logger.error("%s", json.dumps(record, sort_keys=True, default=str))


def close_training_logger(logger: logging.Logger | None) -> None:
    if logger is None:
        return
    handlers = list(logger.handlers)
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def log_truthfulqa_result(
    logger: logging.Logger | None,
    *,
    checkpoint_id: str,
    checkpoint_dir: str,
    update_index: int,
    subset: str,
    total_examples: int,
    score: float,
    mc1_score: float,
    mc2_score: float,
) -> None:
    if logger is None:
        return
    scores_path = getattr(logger, "_truthfulqa_scores_path", None)
    if scores_path is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_dir": checkpoint_dir,
        "update_index": update_index,
        "subset": subset,
        "total_examples": total_examples,
        "score": score,
        "mc1_score": mc1_score,
        "mc2_score": mc2_score,
    }
    Path(scores_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(scores_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def summarize_episode(episode: EpisodeRollout) -> EpisodeLogSummary:
    invalid_action_count = sum(1 for step in episode.steps if step.validation_errors)
    return EpisodeLogSummary(
        episode_id=episode.metadata.episode_id,
        trainable_role=episode.metadata.trainable_role,
        winner=episode.outcome.winner.value if episode.outcome.winner is not None else None,
        final_reward=episode.outcome.final_reward,
        invalid_action_count=invalid_action_count,
        transcript_length=len(episode.final_state.transcript),
    )


def summarize_run(
    episodes: Sequence[EpisodeRollout],
    *,
    total_updates: int,
    grouped_batch: GroupedEpisodeBatch | None = None,
) -> RunLogSummary:
    rewards = [episode.outcome.final_reward for episode in episodes if episode.outcome.final_reward is not None]
    wins = [
        episode
        for episode in episodes
        if episode.outcome.winner is not None and episode.metadata.trainable_alignment.value == episode.outcome.winner.value
    ]
    total_steps = sum(1 for episode in episodes for step in episode.steps if step.actor is not None)
    invalid_steps = sum(1 for episode in episodes for step in episode.steps if step.validation_errors)
    role_rewards: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        if episode.outcome.final_reward is not None:
            role_rewards[episode.metadata.trainable_role.value].append(episode.outcome.final_reward)
    reward_by_role = {
        role: sum(values) / len(values)
        for role, values in role_rewards.items()
    }
    grouped_reward_mean = None
    if grouped_batch is not None:
        present = [score for score in grouped_batch.batch.normalized_group_scores if score is not None]
        grouped_reward_mean = sum(present) / len(present) if present else None
    return RunLogSummary(
        total_episodes=len(episodes),
        total_updates=total_updates,
        mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        win_rate=len(wins) / len(episodes) if episodes else 0.0,
        invalid_action_rate=invalid_steps / total_steps if total_steps else 0.0,
        reward_by_role=reward_by_role,
        grouped_reward_mean=grouped_reward_mean,
    )


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, BaseModel):
            serialized[key] = value.model_dump(mode="json")
        elif isinstance(value, tuple):
            serialized[key] = list(value)
        elif isinstance(value, list):
            serialized[key] = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        elif isinstance(value, dict):
            serialized[key] = _serialize_payload(value)
        else:
            serialized[key] = value
    return serialized
