from contracts import Action, ActionType, Role
from game import advance_phase, apply_action, build_observation, new_game


def _night_resolved_state():
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    return state


def test_detective_sees_only_detective_investigations() -> None:
    state = _night_resolved_state()

    detective_view = build_observation(state, 2)
    villager_view = build_observation(state, 3)

    assert detective_view.private_state.own_role == Role.DETECTIVE
    assert len(detective_view.private_state.investigation_results) == 1
    assert detective_view.private_state.investigation_results[0].target == 0
    assert villager_view.private_state.investigation_results == ()


def test_doctor_sees_last_protection_target_but_others_do_not() -> None:
    state = _night_resolved_state()

    doctor_view = build_observation(state, 1)
    mafia_view = build_observation(state, 0)

    assert doctor_view.private_state.own_role == Role.DOCTOR
    assert doctor_view.private_state.last_protection_target == 3
    assert mafia_view.private_state.last_protection_target is None


def test_non_mafia_roles_do_not_receive_hidden_mafia_data_in_v1() -> None:
    state = _night_resolved_state()

    mafia_view = build_observation(state, 0)
    doctor_view = build_observation(state, 1)
    detective_view = build_observation(state, 2)
    villager_view = build_observation(state, 3)

    assert mafia_view.private_state.mafia_teammates == ()
    assert doctor_view.private_state.mafia_teammates == ()
    assert detective_view.private_state.mafia_teammates == ()
    assert villager_view.private_state.mafia_teammates == ()


def test_private_observation_stays_role_agnostic_after_day_advance() -> None:
    state = _night_resolved_state()
    state = advance_phase(state)

    for actor in state.living_players:
        observation = build_observation(state, actor)
        assert observation.actor == actor
        assert observation.private_state.own_role == state.roles[actor]
