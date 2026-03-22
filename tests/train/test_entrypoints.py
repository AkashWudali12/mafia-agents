from train.entrypoints import run_training_updates
from train.trainer import DebugTrainer


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
