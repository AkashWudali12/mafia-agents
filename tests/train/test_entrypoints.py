from train.audio import CompositeViewerEventSink
from train.entrypoints import run_training_from_config, run_training_updates
from train.trainer import DebugTrainer
from train.config import EvaluationConfig, LoggingConfig, ModelConfig, OpponentConfig, ProjectConfig, RewardConfig, TrainConfig, TrainingConfig, ViewerConfig
from contracts import EnvironmentConfig


class _FakeTrainer(DebugTrainer):
    def __init__(self) -> None:
        self.calls = 0

    def run_iteration(self, *, seed=None):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(
            checkpoint_dir=f"/tmp/ckpt-{self.calls}",
            run_summary=SimpleNamespace(mean_reward=float(self.calls), win_rate=0.25 * self.calls),
        )


def test_run_training_updates_accumulates_iteration_summaries() -> None:
    summary = run_training_updates(trainer=_FakeTrainer(), total_updates=3, seed=10)

    assert summary.total_updates == 3
    assert summary.checkpoint_dirs == ("/tmp/ckpt-1", "/tmp/ckpt-2", "/tmp/ckpt-3")
    assert summary.mean_rewards == (1.0, 2.0, 3.0)
    assert summary.win_rates == (0.25, 0.5, 0.75)
    assert summary.final_checkpoint_dir == "/tmp/ckpt-3"


def test_run_training_from_config_invokes_viewer_started_callback(monkeypatch, tmp_path) -> None:
    opened: list[str] = []

    class _FakeViewer:
        def __init__(self, *, host: str, port: int, max_cached_events: int) -> None:
            self.host = host
            self.port = port
            self.max_cached_events = max_cached_events

        def start(self) -> str:
            return "http://127.0.0.1:9999"

        def stop(self) -> None:
            return None

    class _FakeTrainer:
        def run_iteration(self, *, seed=None):
            from types import SimpleNamespace

            return SimpleNamespace(
                checkpoint_dir=None,
                run_summary=SimpleNamespace(mean_reward=0.5, win_rate=0.25),
            )

    monkeypatch.setattr("train.entrypoints.LiveTrainingViewer", _FakeViewer)
    monkeypatch.setattr("train.entrypoints.build_local_huggingface_trainer", lambda **kwargs: _FakeTrainer())

    summary = run_training_from_config(
        config=TrainConfig(
            project=ProjectConfig(experiment_name="viewer-test", seed=123),
            environment=EnvironmentConfig(),
            model=ModelConfig(trainable_model_name="hf://qwen3-8b"),
            opponents=OpponentConfig(
                opponent_provider="openrouter",
                opponent_pool_id="pool-v1",
                model_names=("model-a",),
            ),
            training=TrainingConfig(total_updates=1),
            rewards=RewardConfig(),
            evaluation=EvaluationConfig(),
            logging=LoggingConfig(log_dir_name="logs"),
            viewer=ViewerConfig(enabled=True, port=9999),
        ),
        checkpoint_root=tmp_path,
        viewer_started_callback=opened.append,
    )

    assert opened == ["http://127.0.0.1:9999"]
    assert summary.viewer_url == "http://127.0.0.1:9999"


def test_run_training_from_config_wraps_viewer_with_tts_sink(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeTTS:
        def publish(self, event) -> None:
            return None

    class _FakeViewer:
        def __init__(self, *, host: str, port: int, max_cached_events: int) -> None:
            self.host = host
            self.port = port
            self.max_cached_events = max_cached_events

        def start(self) -> str:
            return "http://127.0.0.1:9998"

        def stop(self) -> None:
            return None

    class _FakeTrainer:
        def run_iteration(self, *, seed=None):
            from types import SimpleNamespace

            return SimpleNamespace(
                checkpoint_dir=None,
                run_summary=SimpleNamespace(mean_reward=0.5, win_rate=0.25),
            )

    def _build_trainer(**kwargs):
        captured["viewer_sink"] = kwargs.get("viewer")
        return _FakeTrainer()

    monkeypatch.setattr("train.entrypoints.LiveTrainingViewer", _FakeViewer)
    monkeypatch.setattr("train.entrypoints.build_elevenlabs_event_sink", lambda **kwargs: _FakeTTS())
    monkeypatch.setattr("train.entrypoints.build_local_huggingface_trainer", _build_trainer)

    summary = run_training_from_config(
        config=TrainConfig(
            project=ProjectConfig(experiment_name="tts-viewer-test", seed=123),
            environment=EnvironmentConfig(),
            model=ModelConfig(trainable_model_name="hf://qwen3-8b"),
            opponents=OpponentConfig(
                opponent_provider="openrouter",
                opponent_pool_id="pool-v1",
                model_names=("model-a",),
            ),
            training=TrainingConfig(total_updates=1),
            rewards=RewardConfig(),
            evaluation=EvaluationConfig(),
            logging=LoggingConfig(log_dir_name="logs"),
            viewer=ViewerConfig(enabled=True, port=9998, tts_enabled=True),
        ),
        checkpoint_root=tmp_path,
    )

    assert isinstance(captured["viewer_sink"], CompositeViewerEventSink)
    assert summary.viewer_url == "http://127.0.0.1:9998"
