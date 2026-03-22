from eval.mafia_eval import run_mafia_evaluation
from policies import FirstLegalPolicy


def test_mafia_evaluation_smoke_runs_and_aggregates_metrics() -> None:
    result = run_mafia_evaluation(
        trainable_policy=FirstLegalPolicy(),
        seeds=(11, 22, 33, 44),
        episodes_per_role=1,
        checkpoint_id="ckpt-1",
    )

    assert result.checkpoint_id == "ckpt-1"
    assert result.total_episodes == 4
    assert 0.0 <= result.overall_win_rate <= 1.0
    assert set(result.win_rate_by_role) == {"mafia", "doctor", "detective", "villager"}
    assert "terminal_win_loss" in result.average_reward_by_component
