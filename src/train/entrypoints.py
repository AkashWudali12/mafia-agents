from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from train.config import TrainConfig, load_train_config
from train.logging import close_training_logger, configure_training_logger, log_debug_event
from train.trainer import DebugTrainer, TrainingIterationResult, build_local_huggingface_trainer
from train.viewer import LiveTrainingViewer


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class TrainingRunSummary(FrozenModel):
    total_updates: int
    checkpoint_dirs: tuple[str, ...] = ()
    mean_rewards: tuple[float, ...] = ()
    win_rates: tuple[float, ...] = ()
    final_checkpoint_dir: str | None = None
    viewer_url: str | None = None


def run_training_updates(
    *,
    trainer: DebugTrainer,
    total_updates: int,
    seed: int | None = None,
) -> TrainingRunSummary:
    results: list[TrainingIterationResult] = []
    for update_index in range(total_updates):
        iteration_seed = None if seed is None else seed + update_index
        results.append(trainer.run_iteration(seed=iteration_seed))
    checkpoint_dirs = tuple(
        result.checkpoint_dir
        for result in results
        if result.checkpoint_dir is not None
    )
    mean_rewards = tuple(result.run_summary.mean_reward for result in results)
    win_rates = tuple(result.run_summary.win_rate for result in results)
    return TrainingRunSummary(
        total_updates=total_updates,
        checkpoint_dirs=checkpoint_dirs,
        mean_rewards=mean_rewards,
        win_rates=win_rates,
        final_checkpoint_dir=checkpoint_dirs[-1] if checkpoint_dirs else None,
    )


def run_training_from_config(
    *,
    config: TrainConfig,
    checkpoint_root: str | Path,
    seed: int | None = None,
    viewer_started_callback: Callable[[str], None] | None = None,
) -> TrainingRunSummary:
    checkpoint_root_path = Path(checkpoint_root)
    log_dir = checkpoint_root_path / config.logging.log_dir_name
    logger = configure_training_logger(
        log_dir=log_dir,
        run_name=config.project.experiment_name,
        level=config.logging.level,
    )
    log_debug_event(
        logger,
        "training_run_start",
        experiment_name=config.project.experiment_name,
        checkpoint_root=str(checkpoint_root_path),
        config=config,
    )
    viewer = None
    viewer_url = None
    try:
        if config.viewer.enabled:
            viewer = LiveTrainingViewer(
                host=config.viewer.host,
                port=config.viewer.port,
                max_cached_events=config.viewer.max_cached_events,
            )
            viewer_url = viewer.start()
            log_debug_event(logger, "viewer_started", viewer_url=viewer_url)
            if viewer_started_callback is not None:
                viewer_started_callback(viewer_url)
        trainer = build_local_huggingface_trainer(
            config=config,
            checkpoint_root=checkpoint_root,
            logger=logger,
            viewer=viewer,
        )
        summary = run_training_updates(
            trainer=trainer,
            total_updates=config.training.total_updates,
            seed=seed if seed is not None else config.project.seed,
        )
        summary = summary.model_copy(update={"viewer_url": viewer_url})
        log_debug_event(logger, "training_run_complete", summary=summary)
        return summary
    finally:
        if viewer is not None:
            viewer.stop()
        close_training_logger(logger)


def run_training_from_config_path(
    *,
    config_path: str | Path,
    checkpoint_root: str | Path,
    seed: int | None = None,
    viewer_started_callback: Callable[[str], None] | None = None,
) -> TrainingRunSummary:
    config = load_train_config(config_path)
    return run_training_from_config(
        config=config,
        checkpoint_root=checkpoint_root,
        seed=seed,
        viewer_started_callback=viewer_started_callback,
    )
