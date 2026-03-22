from contracts import Action, ActionType, Alignment, Phase, Role, WinCondition
from game import apply_action, new_game
from train import build_scripted_policy_map, next_actor, run_episode


def _policy_map() -> dict[int, object]:
    return build_scripted_policy_map(new_game())


def test_run_episode_reaches_terminal_and_records_steps() -> None:
    result = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=7)

    assert result.final_state.is_terminal is True
    assert result.final_state.winner in {WinCondition.MAFIA, WinCondition.TOWN}
    assert result.metadata.seed == 7
    assert result.metadata.trainable_seat == 0
    assert result.metadata.trainable_role == Role.MAFIA
    assert result.metadata.trainable_alignment == Alignment.MAFIA
    assert result.metadata.environment_config_hash
    assert len(result.steps) > 0
    assert any(step.auto_advanced for step in result.steps)
    assert any(step.actor == 0 for step in result.steps)
    assert result.outcome.winner == result.final_state.winner
    assert result.outcome.num_days_reached == result.final_state.day
    bd = result.outcome.reward_breakdown
    assert result.outcome.final_reward is not None
    assert bd["terminal_win_loss"] in {1.0, -1.0}
    assert 0.0 <= bd["survival_bonus"] <= 0.2
    assert bd["total_reward"] == bd["terminal_win_loss"] + bd["survival_bonus"]
    assert result.outcome.final_reward == bd["total_reward"]

    for step in result.steps:
        if step.auto_advanced:
            assert step.actor is None
            assert step.is_trainable_actor is False
            continue
        assert step.is_valid is not None
        assert step.observation_summary is not None
        assert step.observation is not None
        assert step.legal_actions == step.observation.legal_actions
        assert step.submitted_action is not None
        assert step.normalized_action is not None


def test_run_episode_is_deterministic_for_same_seed_and_policies() -> None:
    first = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=11)
    second = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=11)

    assert first.final_state == second.final_state
    assert first.steps == second.steps


def test_run_episode_records_outcome_for_survival_and_notable_events() -> None:
    result = run_episode(trainable_seat=0, seat_policies=_policy_map(), seed=5)

    assert result.outcome.trainable_survived == result.final_state.alive[0]
    assert result.outcome.final_reward is not None
    assert result.outcome.reward_breakdown["terminal_win_loss"] in {1.0, -1.0}
    assert result.outcome.notable_events
    if result.outcome.trainable_survived:
        assert "trainable_survived_to_terminal" in result.outcome.notable_events
    else:
        assert result.outcome.elimination_day is not None


def test_next_actor_uses_dead_role_seat_to_progress_night_phase() -> None:
    state = new_game()
    state = state.model_copy(update={"phase": Phase.NIGHT_DOCTOR, "alive": (True, False, True, True, True, True)})

    assert next_actor(state) == 1


def test_apply_action_skips_dead_detective_phase_after_doctor_turn() -> None:
    state = new_game()
    state = state.model_copy(update={"phase": Phase.NIGHT_DOCTOR, "alive": (True, True, False, True, True, True)})

    next_state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=0))

    assert next_state.phase == Phase.DAY_ANNOUNCEMENT


def test_build_scripted_policy_map_matches_roles() -> None:
    state = new_game()

    policy_map = build_scripted_policy_map(state)

    assert sorted(policy_map.keys()) == [0, 1, 2, 3, 4, 5]
    assert policy_map[0].__class__.__name__ == "ScriptedMafiaPolicy"
    assert policy_map[1].__class__.__name__ == "ScriptedDoctorPolicy"
    assert policy_map[2].__class__.__name__ == "ScriptedDetectivePolicy"
    assert policy_map[3].__class__.__name__ == "ScriptedVillagerPolicy"
    assert policy_map[4].__class__.__name__ == "ScriptedVillagerPolicy"
    assert policy_map[5].__class__.__name__ == "ScriptedVillagerPolicy"
