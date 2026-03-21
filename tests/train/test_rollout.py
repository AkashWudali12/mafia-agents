from contracts import Phase, WinCondition
from game import new_game
from policies import FirstLegalPolicy
from train import next_actor, run_episode


def _policy_map() -> dict[int, FirstLegalPolicy]:
    return {seat: FirstLegalPolicy() for seat in range(5)}


def test_run_episode_reaches_terminal_and_records_steps() -> None:
    result = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=7)

    assert result.final_state.is_terminal is True
    assert result.final_state.winner in {WinCondition.MAFIA, WinCondition.TOWN}
    assert len(result.steps) > 0
    assert any(step.auto_advanced for step in result.steps)
    assert any(step.actor == 0 for step in result.steps)

    for step in result.steps:
        if step.auto_advanced:
            assert step.actor is None
            continue
        assert step.observation is not None
        assert step.submitted_action is not None
        assert step.normalized_action is not None


def test_run_episode_is_deterministic_for_same_seed_and_policies() -> None:
    first = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=11)
    second = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=11)

    assert first.final_state == second.final_state
    assert first.steps == second.steps


def test_next_actor_uses_dead_role_seat_to_progress_night_phase() -> None:
    state = new_game()
    state = state.model_copy(update={"phase": Phase.NIGHT_DOCTOR, "alive": (True, False, True, True, True)})

    assert next_actor(state) == 1
