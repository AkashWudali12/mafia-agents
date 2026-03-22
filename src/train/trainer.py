from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from contracts import EnvironmentConfig, Role
from game import new_game
from policies import OpenRouterPolicy, Policy, PydanticAiOpenRouterClient
from train.checkpoints import CheckpointState, save_checkpoint
from train.config import TrainConfig
from train.grpo import GroupedEpisodeBatch, build_grouped_episode_batch
from train.hf_policy import HuggingFaceGroupOptimizer, HuggingFaceTrainablePolicy
from train.logging import RunLogSummary, log_debug_event, summarize_episode, summarize_run
from train.rollout import run_episode
from train.trajectory import EpisodeRollout
from train.viewer import ViewerEventSink


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class GroupOptimizer(Protocol):
    def update(self, grouped_batch: GroupedEpisodeBatch) -> dict[str, float]:
        """Run one optimizer step and return scalar metrics."""


class NoOpGroupOptimizer:
    def update(self, grouped_batch: GroupedEpisodeBatch) -> dict[str, float]:
        trainable_steps = sum(
            1
            for episode in grouped_batch.episodes
            for step in episode.steps
            if step.is_trainable_actor
        )
        return {
            "loss": 0.0,
            "num_group_episodes": float(len(grouped_batch.episodes)),
            "num_trainable_steps": float(trainable_steps),
        }


class TrainingIterationResult(FrozenModel):
    grouped_batch: GroupedEpisodeBatch
    optimizer_metrics: dict[str, float]
    run_summary: RunLogSummary
    checkpoint_dir: str | None = None


class DebugTrainer:
    def __init__(
        self,
        *,
        config: TrainConfig,
        trainable_policy: Policy,
        optimizer: GroupOptimizer | None = None,
        checkpoint_root: str | Path | None = None,
        logger: logging.Logger | None = None,
        opponent_client: object | None = None,
        viewer: ViewerEventSink | None = None,
    ) -> None:
        self._config = config
        self._trainable_policy = trainable_policy
        self._optimizer = optimizer or NoOpGroupOptimizer()
        self._checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self._update_index = 0
        self._logger = logger
        self._opponent_client = opponent_client
        self._viewer = viewer

    def run_iteration(self, *, seed: int | None = None) -> TrainingIterationResult:
        log_debug_event(
            self._logger,
            "training_iteration_start",
            update_index=self._update_index,
            seed=seed if seed is not None else self._config.project.seed,
            group_size=self._config.training.grpo_group_size,
        )
        rng = random.Random(self._config.project.seed if seed is None else seed)
        episodes = tuple(self._run_group(rng))
        grouped_batch = build_grouped_episode_batch(
            episodes,
            group_id=f"group_{self._update_index:05d}",
            logger=self._logger,
        )
        optimizer_metrics = self._optimizer.update(grouped_batch)
        self._update_index += 1
        run_summary = summarize_run(
            grouped_batch.episodes,
            total_updates=self._update_index,
            grouped_batch=grouped_batch,
        )
        log_debug_event(
            self._logger,
            "optimizer_update_complete",
            update_index=self._update_index,
            optimizer_metrics=optimizer_metrics,
            run_summary=run_summary,
        )
        checkpoint_dir = None
        if self._checkpoint_root is not None and self._update_index % self._config.training.checkpoint_interval == 0:
            checkpoint = CheckpointState(
                checkpoint_id=f"ckpt_{self._update_index:05d}",
                update_index=self._update_index,
                step_counters={"updates": self._update_index, "episodes": len(grouped_batch.episodes)},
                model_reference=self._config.model.trainable_model_name,
                reward_config_snapshot=self._config.rewards.model_dump(mode="json"),
                train_config_snapshot=self._config.model_dump(mode="json"),
                metrics={
                    "mean_reward": run_summary.mean_reward,
                    "win_rate": run_summary.win_rate,
                    **optimizer_metrics,
                },
            )
            checkpoint_path = save_checkpoint(
                self._checkpoint_root,
                state=checkpoint,
                grouped_batch=grouped_batch,
                episodes=grouped_batch.episodes,
                logger=self._logger,
            )
            save_pretrained = getattr(self._trainable_policy, "save_pretrained", None)
            if callable(save_pretrained):
                save_pretrained(checkpoint_path / "model")
                log_debug_event(
                    self._logger,
                    "model_artifacts_saved",
                    checkpoint_dir=str(checkpoint_path),
                    model_dir=str(checkpoint_path / "model"),
                )
            checkpoint_dir = str(checkpoint_path)
        log_debug_event(
            self._logger,
            "training_iteration_complete",
            update_index=self._update_index,
            checkpoint_dir=checkpoint_dir,
            mean_reward=run_summary.mean_reward,
            win_rate=run_summary.win_rate,
        )
        return TrainingIterationResult(
            grouped_batch=grouped_batch,
            optimizer_metrics=optimizer_metrics,
            run_summary=run_summary,
            checkpoint_dir=checkpoint_dir,
        )

    def _run_group(self, rng: random.Random) -> list[EpisodeRollout]:
        episodes: list[EpisodeRollout] = []
        for _ in range(self._config.training.grpo_group_size):
            role = sample_trainable_role(rng, self._config.environment)
            episode_seed = rng.randint(0, 2**31 - 1)
            state = new_game(config=self._config.environment, seed=episode_seed)
            trainable_seat = sample_seat_for_role(rng, state.roles, role)
            log_debug_event(
                self._logger,
                "episode_assignment",
                update_index=self._update_index,
                role=role,
                trainable_seat=trainable_seat,
                episode_seed=episode_seed,
            )
            seat_policies, opponent_models = self._build_seat_policy_map(
                state=state,
                rng=rng,
                trainable_seat=trainable_seat,
            )
            seat_policies[trainable_seat] = self._trainable_policy
            rollout = run_episode(
                trainable_seat=trainable_seat,
                seat_policies=seat_policies,
                metadata_update={
                    "checkpoint_id": f"debug_update_{self._update_index:05d}",
                    "opponent_pool_id": self._config.opponents.opponent_pool_id,
                    "opponent_prompt_version": self._config.opponents.prompt_version,
                    "opponent_cache_behavior": self._config.opponents.cache_behavior,
                    "opponent_model_names": opponent_models,
                },
                update_index=self._update_index,
                seed=episode_seed,
                initial_state=state,
                logger=self._logger,
                viewer=self._viewer,
            )
            log_debug_event(
                self._logger,
                "episode_logged",
                update_index=self._update_index,
                episode_summary=summarize_episode(rollout),
            )
            episodes.append(rollout)
        return episodes

    def _build_seat_policy_map(
        self,
        *,
        state,
        rng: random.Random,
        trainable_seat: int,
    ) -> tuple[dict[int, Policy], tuple[str, ...]]:
        provider = self._config.opponents.opponent_provider
        if provider != "openrouter":
            raise ValueError("training runs require opponent_provider=openrouter")
        if not self._config.opponents.model_names:
            raise ValueError("training runs require a non-empty OpenRouter opponent model pool")

        client = self._opponent_client or PydanticAiOpenRouterClient()
        seat_policies: dict[int, Policy] = {}
        opponent_models: list[str] = []
        for seat, _role in enumerate(state.roles):
            if seat == trainable_seat:
                continue
            model_name = sample_opponent_model(rng, self._config.opponents.model_names)
            seat_policies[seat] = OpenRouterPolicy(client=client, model=model_name)
            opponent_models.append(model_name)
            log_debug_event(
                self._logger,
                "opponent_model_assigned",
                update_index=self._update_index,
                seat=seat,
                model_name=model_name,
                provider=provider,
            )
        return seat_policies, tuple(opponent_models)


def build_local_huggingface_trainer(
    *,
    config: TrainConfig,
    checkpoint_root: str | Path | None = None,
    logger: logging.Logger | None = None,
    opponent_client: object | None = None,
    viewer: ViewerEventSink | None = None,
) -> DebugTrainer:
    policy = HuggingFaceTrainablePolicy(
        model_name=config.model.trainable_model_name,
        tokenizer_name=config.model.tokenizer_name,
        device=config.model.device,
        cache_dir=config.model.cache_dir,
        max_new_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        logger=logger,
    )
    optimizer = HuggingFaceGroupOptimizer(
        policy=policy,
        learning_rate=config.training.learning_rate,
    )
    return DebugTrainer(
        config=config,
        trainable_policy=policy,
        optimizer=optimizer,
        checkpoint_root=checkpoint_root,
        logger=logger,
        opponent_client=opponent_client,
        viewer=viewer,
    )


def sample_trainable_role(rng: random.Random, environment_config: EnvironmentConfig) -> Role:
    categories = sorted(set(environment_config.roles), key=lambda role: role.value)
    return rng.choice(categories)


def sample_seat_for_role(rng: random.Random, roles: tuple[Role, ...], role: Role) -> int:
    compatible = [seat for seat, seat_role in enumerate(roles) if seat_role == role]
    if not compatible:
        raise ValueError(f"no seat available for role {role}")
    return rng.choice(compatible)


def sample_opponent_model(rng: random.Random, model_pool: tuple[str, ...]) -> str:
    if not model_pool:
        raise ValueError("opponent model pool must not be empty")
    return rng.choice(model_pool)
