from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from math import sqrt

from pydantic import BaseModel, ConfigDict

from train.logging import log_debug_event
from train.trajectory import EpisodeRollout, GroupBatchRecord, RolloutStep


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class GroupedEpisodeBatch(FrozenModel):
    batch: GroupBatchRecord
    episodes: tuple[EpisodeRollout, ...]


def normalize_group_rewards(raw_rewards: Sequence[float | None]) -> tuple[float | None, ...]:
    present_rewards = [reward for reward in raw_rewards if reward is not None]
    if not present_rewards:
        return tuple(None for _ in raw_rewards)
    if len(present_rewards) == 1:
        baseline = present_rewards[0]
        return tuple(0.0 if reward is not None else None for reward in raw_rewards)

    mean_reward = sum(present_rewards) / len(present_rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in present_rewards) / len(present_rewards)
    if variance == 0.0:
        return tuple(0.0 if reward is not None else None for reward in raw_rewards)
    stddev = sqrt(variance)
    return tuple(((reward - mean_reward) / stddev) if reward is not None else None for reward in raw_rewards)


def build_grouped_episode_batch(
    episodes: Sequence[EpisodeRollout],
    *,
    group_id: str | None = None,
    logger: logging.Logger | None = None,
) -> GroupedEpisodeBatch:
    if not episodes:
        raise ValueError("at least one episode is required to build a GRPO batch")

    _validate_group_compatibility(episodes)
    resolved_group_id = group_id or _derive_group_id(episodes)
    raw_rewards = tuple(episode.outcome.final_reward for episode in episodes)
    normalized_scores = normalize_group_rewards(raw_rewards)
    episode_ids = tuple(episode.metadata.episode_id for episode in episodes)
    present = [(episode_id, reward) for episode_id, reward in zip(episode_ids, raw_rewards, strict=True) if reward is not None]
    best_episode_id = max(present, key=lambda item: item[1])[0] if present else None
    worst_episode_id = min(present, key=lambda item: item[1])[0] if present else None

    updated_episodes = tuple(
        _attach_group_score(episode, group_id=resolved_group_id, normalized_score=score)
        for episode, score in zip(episodes, normalized_scores, strict=True)
    )
    grouped = GroupedEpisodeBatch(
        batch=GroupBatchRecord(
            group_id=resolved_group_id,
            episode_ids=episode_ids,
            raw_rewards=raw_rewards,
            normalized_group_scores=normalized_scores,
            best_episode_id=best_episode_id,
            worst_episode_id=worst_episode_id,
        ),
        episodes=updated_episodes,
    )
    log_debug_event(
        logger,
        "grpo_batch_built",
        group_id=resolved_group_id,
        raw_rewards=raw_rewards,
        normalized_scores=normalized_scores,
        best_episode_id=best_episode_id,
        worst_episode_id=worst_episode_id,
    )
    return grouped


def _attach_group_score(
    episode: EpisodeRollout,
    *,
    group_id: str,
    normalized_score: float | None,
) -> EpisodeRollout:
    updated_steps: tuple[RolloutStep, ...] = tuple(
        step.model_copy(
            update={
                "group_normalized_score": normalized_score if step.is_trainable_actor else step.group_normalized_score,
            }
        )
        for step in episode.steps
    )
    updated_metadata = episode.metadata.model_copy(update={"group_id": group_id})
    return episode.model_copy(update={"metadata": updated_metadata, "steps": updated_steps})


def _derive_group_id(episodes: Sequence[EpisodeRollout]) -> str:
    payload = ":".join(episode.metadata.episode_id for episode in episodes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_group_compatibility(episodes: Sequence[EpisodeRollout]) -> None:
    first = episodes[0]
    baseline = (
        first.metadata.checkpoint_id,
        first.metadata.environment_config_hash,
    )
    for episode in episodes[1:]:
        current = (
            episode.metadata.checkpoint_id,
            episode.metadata.environment_config_hash,
        )
        if current != baseline:
            raise ValueError("episodes in the same GRPO batch must share checkpoint and environment config")
