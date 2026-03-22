import json

from contracts import EnvironmentConfig, Role
from eval.truthfulqa_eval import TruthfulQABenchmarkResult
from policies import FirstLegalPolicy
from train import DebugTrainer, TrainConfig
from train.checkpoints import load_checkpoint
from train.config import EvaluationConfig, ModelConfig, OpponentConfig, ProjectConfig, RewardConfig, TrainingConfig
from train.logging import close_training_logger, configure_training_logger


class StubOpenRouterClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int):
        self.calls.append(model)
        return {"action_type": "noop"}


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
        opponent_client=StubOpenRouterClient(),
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
    trainer = DebugTrainer(
        config=trainer_config,
        trainable_policy=FirstLegalPolicy(),
        opponent_client=StubOpenRouterClient(),
    )

    roles = [episode.metadata.trainable_role for episode in trainer.run_iteration(seed=9).grouped_batch.episodes]

    assert set(roles).issubset({Role.MAFIA, Role.DOCTOR, Role.DETECTIVE, Role.VILLAGER})


def test_debug_trainer_writes_debug_logs_to_files(tmp_path) -> None:
    logger = configure_training_logger(log_dir=tmp_path / "logs", run_name="trainer-test")
    trainer = DebugTrainer(
        config=_config(),
        trainable_policy=FirstLegalPolicy(),
        checkpoint_root=tmp_path,
        logger=logger,
        opponent_client=StubOpenRouterClient(),
    )

    trainer.run_iteration(seed=3)
    close_training_logger(logger)

    training_log = (tmp_path / "logs" / "training.log").read_text()
    events_log = (tmp_path / "logs" / "events.jsonl").read_text()

    assert "training_iteration_start" in training_log
    assert "episode_complete" in training_log
    assert "grpo_batch_built" in events_log
    assert "training_iteration_complete" in events_log


def test_debug_trainer_prints_logs_to_stdout(tmp_path, capsys) -> None:
    logger = configure_training_logger(log_dir=tmp_path / "logs", run_name="trainer-stdout-test")
    trainer = DebugTrainer(
        config=_config(),
        trainable_policy=FirstLegalPolicy(),
        checkpoint_root=tmp_path,
        logger=logger,
        opponent_client=StubOpenRouterClient(),
    )

    trainer.run_iteration(seed=4)
    close_training_logger(logger)
    captured = capsys.readouterr()

    assert "training_iteration_start" in captured.err
    assert "training_iteration_complete" in captured.err


def test_debug_trainer_runs_truthfulqa_after_checkpoint_and_logs_score(tmp_path) -> None:
    class RecordingPolicy(FirstLegalPolicy):
        def save_pretrained(self, output_dir) -> None:
            from pathlib import Path

            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "saved.txt").write_text("ok")

    logger = configure_training_logger(log_dir=tmp_path / "logs", run_name="truthfulqa-test")
    calls = []

    def truthfulqa_evaluator(**kwargs) -> TruthfulQABenchmarkResult:
        calls.append(kwargs)
        return TruthfulQABenchmarkResult(
            checkpoint_id=kwargs["checkpoint_id"],
            subset=kwargs["subset"],
            total_examples=3,
            score=2 / 3,
            mc1_score=1 / 3,
            mc2_score=2 / 3,
            per_question_mc1_correct={"q1": False, "q2": True, "q3": False},
            per_question_mc2_score={"q1": 0.0, "q2": 1.0, "q3": 1.0},
        )

    trainer = DebugTrainer(
        config=_config().model_copy(
            update={
                "evaluation": EvaluationConfig(
                    truthfulqa_subset="multiple_choice",
                    benchmark_interval=1,
                )
            }
        ),
        trainable_policy=RecordingPolicy(),
        checkpoint_root=tmp_path,
        logger=logger,
        opponent_client=StubOpenRouterClient(),
        truthfulqa_evaluator=truthfulqa_evaluator,
    )

    result = trainer.run_iteration(seed=5)
    close_training_logger(logger)

    assert result.truthfulqa_result is not None
    assert result.truthfulqa_result.score == 2 / 3
    assert len(calls) == 1
    assert calls[0]["subset"] == "multiple_choice"
    checkpoint = load_checkpoint(result.checkpoint_dir)
    assert checkpoint.metrics["truthfulqa_score"] == 2 / 3
    assert checkpoint.metrics["truthfulqa_mc1_score"] == 1 / 3
    assert checkpoint.metrics["truthfulqa_mc2_score"] == 2 / 3

    scores_log = (tmp_path / "logs" / "truthfulqa_scores.jsonl").read_text().strip().splitlines()
    assert len(scores_log) == 1
    score_record = json.loads(scores_log[0])
    assert score_record["checkpoint_id"] == "ckpt_00001"
    assert score_record["score"] == 2 / 3
    assert score_record["mc1_score"] == 1 / 3
    assert score_record["mc2_score"] == 2 / 3

    events_log = (tmp_path / "logs" / "events.jsonl").read_text()
    assert "truthfulqa_benchmark_start" in events_log
    assert "truthfulqa_benchmark_complete" in events_log


def test_debug_trainer_respects_truthfulqa_benchmark_interval(tmp_path) -> None:
    class RecordingPolicy(FirstLegalPolicy):
        def save_pretrained(self, output_dir) -> None:
            from pathlib import Path

            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "saved.txt").write_text("ok")

    logger = configure_training_logger(log_dir=tmp_path / "logs", run_name="truthfulqa-interval-test")
    calls = []

    def truthfulqa_evaluator(**kwargs) -> TruthfulQABenchmarkResult:
        calls.append(kwargs)
        return TruthfulQABenchmarkResult(
            checkpoint_id=kwargs["checkpoint_id"],
            subset=kwargs["subset"],
            total_examples=3,
            score=1.0,
            mc1_score=1.0,
            mc2_score=1.0,
            per_question_mc1_correct={"q1": True, "q2": True, "q3": True},
            per_question_mc2_score={"q1": 1.0, "q2": 1.0, "q3": 1.0},
        )

    config = _config().model_copy(
        update={
            "evaluation": EvaluationConfig(
                truthfulqa_subset="multiple_choice",
                benchmark_interval=2,
            )
        }
    )
    trainer = DebugTrainer(
        config=config,
        trainable_policy=RecordingPolicy(),
        checkpoint_root=tmp_path,
        logger=logger,
        opponent_client=StubOpenRouterClient(),
        truthfulqa_evaluator=truthfulqa_evaluator,
    )

    first = trainer.run_iteration(seed=1)
    second = trainer.run_iteration(seed=2)
    close_training_logger(logger)

    assert first.truthfulqa_result is None
    assert second.truthfulqa_result is not None
    assert len(calls) == 1
    assert calls[0]["checkpoint_id"] == "ckpt_00002"

    first_checkpoint = load_checkpoint(first.checkpoint_dir)
    second_checkpoint = load_checkpoint(second.checkpoint_dir)
    assert "truthfulqa_score" not in first_checkpoint.metrics
    assert second_checkpoint.metrics["truthfulqa_score"] == 1.0

    scores_log = (tmp_path / "logs" / "truthfulqa_scores.jsonl").read_text().strip().splitlines()
    assert len(scores_log) == 1


def test_debug_trainer_samples_opponent_models_from_pool() -> None:
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
        opponents=OpponentConfig(
            opponent_provider="openrouter",
            opponent_pool_id="pool-v1",
            model_names=("model-a", "model-b", "model-c"),
            sampling_strategy="random_per_seat",
            cache_behavior="disabled",
            prompt_version="v1",
        ),
        training=TrainingConfig(grpo_group_size=4, checkpoint_interval=1),
        rewards=RewardConfig(),
        evaluation=EvaluationConfig(benchmark_interval=0),
    )
