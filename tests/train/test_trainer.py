from contracts import EnvironmentConfig, Role
from policies import FirstLegalPolicy
from train import DebugTrainer, TrainConfig
from train.config import EvaluationConfig, ModelConfig, OpponentConfig, ProjectConfig, RewardConfig, TrainingConfig
from train.logging import close_training_logger, configure_training_logger


def test_debug_trainer_runs_one_iteration_and_saves_checkpoint(tmp_path) -> None:
    class RecordingPolicy(FirstLegalPolicy):
        def save_pretrained(self, output_dir) -> None:
            from pathlib import Path

            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "saved.txt").write_text("ok")

    trainer = DebugTrainer(
        config=_config(),
        trainable_policy=RecordingPolicy(),
        checkpoint_root=tmp_path,
    )

    result = trainer.run_iteration(seed=7)

    assert len(result.grouped_batch.episodes) == 4
    assert result.optimizer_metrics["num_group_episodes"] == 4.0
    assert result.run_summary.total_updates == 1
    assert result.checkpoint_dir is not None
    from pathlib import Path

    assert (Path(result.checkpoint_dir) / "model" / "saved.txt").read_text() == "ok"


def test_role_sampler_covers_role_categories_uniformly() -> None:
    config = EnvironmentConfig()
    trainer_config = _config(environment=config)
    trainer = DebugTrainer(config=trainer_config, trainable_policy=FirstLegalPolicy())

    roles = [episode.metadata.trainable_role for episode in trainer.run_iteration(seed=9).grouped_batch.episodes]

    assert set(roles).issubset({Role.MAFIA, Role.DOCTOR, Role.DETECTIVE, Role.VILLAGER})


def test_debug_trainer_writes_debug_logs_to_files(tmp_path) -> None:
    logger = configure_training_logger(log_dir=tmp_path / "logs", run_name="trainer-test")
    trainer = DebugTrainer(
        config=_config(),
        trainable_policy=FirstLegalPolicy(),
        checkpoint_root=tmp_path,
        logger=logger,
    )

    trainer.run_iteration(seed=3)
    close_training_logger(logger)

    training_log = (tmp_path / "logs" / "training.log").read_text()
    events_log = (tmp_path / "logs" / "events.jsonl").read_text()

    assert "training_iteration_start" in training_log
    assert "episode_complete" in training_log
    assert "grpo_batch_built" in events_log
    assert "training_iteration_complete" in events_log


def test_debug_trainer_samples_opponent_models_from_pool() -> None:
    class StubOpenRouterClient:
        def __init__(self) -> None:
            self.calls = []

        def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int):
            self.calls.append(model)
            return {"action_type": "noop"}

    config = TrainConfig(
        project=ProjectConfig(experiment_name="pool", seed=123),
        environment=EnvironmentConfig(),
        model=ModelConfig(trainable_model_name="hf://qwen3-8b"),
        opponents=OpponentConfig(
            opponent_provider="openrouter",
            opponent_pool_id="pool-v1",
            model_names=("model-a", "model-b", "model-c"),
            sampling_strategy="random_per_seat",
            cache_behavior="disabled",
            prompt_version="v2",
        ),
        training=TrainingConfig(grpo_group_size=2, checkpoint_interval=100),
        rewards=RewardConfig(),
        evaluation=EvaluationConfig(),
    )
    client = StubOpenRouterClient()
    trainer = DebugTrainer(
        config=config,
        trainable_policy=FirstLegalPolicy(),
        opponent_client=client,
    )

    result = trainer.run_iteration(seed=11)

    assert client.calls
    for episode in result.grouped_batch.episodes:
        assert episode.metadata.opponent_pool_id == "pool-v1"
        assert episode.metadata.opponent_prompt_version == "v2"
        assert episode.metadata.opponent_cache_behavior == "disabled"
        assert len(episode.metadata.opponent_model_names) == len(episode.initial_state.roles) - 1
        assert set(episode.metadata.opponent_model_names).issubset({"model-a", "model-b", "model-c"})


def _config(*, environment: EnvironmentConfig | None = None) -> TrainConfig:
    return TrainConfig(
        project=ProjectConfig(experiment_name="debug", seed=123),
        environment=environment or EnvironmentConfig(),
        model=ModelConfig(trainable_model_name="hf://qwen3-8b"),
        opponents=OpponentConfig(),
        training=TrainingConfig(grpo_group_size=4, checkpoint_interval=1),
        rewards=RewardConfig(),
        evaluation=EvaluationConfig(),
    )
