from contracts import Action, ActionType, DiscussionIntent, EnvironmentConfig, Phase, Role, ValidationErrorCode, WinCondition
from game import advance_phase, apply_action, build_observation, new_game, validate_action
from policies import FirstLegalPolicy


def test_full_phase_order_across_day_cycle() -> None:
    state = new_game()
    assert state.phase == Phase.NIGHT_MAFIA

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3))
    assert state.phase == Phase.NIGHT_DOCTOR

    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    assert state.phase == Phase.NIGHT_DETECTIVE

    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    assert state.phase == Phase.DAY_ANNOUNCEMENT

    state = advance_phase(state)
    assert state.phase == Phase.DAY_DISCUSSION

    living = state.living_players
    for _ in range(len(living) * state.config.discussion_rounds):
        speaker = build_observation(state, living[0]).public_state.current_speaker
        state = apply_action(
            state,
            Action(
                actor=speaker,
                action_type=ActionType.SPEAK,
                target=0,
                intent=DiscussionIntent.QUESTION,
                message="status",
            ),
        )
    assert state.phase == Phase.DAY_VOTING

    for actor in tuple(state.living_players):
        target = 1 if actor != 1 else 3
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=target))
    assert state.phase == Phase.RESOLUTION

    state = advance_phase(state)
    assert state.phase == Phase.NIGHT_MAFIA
    assert state.day == 2


def test_night_flow_skips_missing_doctor_phase() -> None:
    config = EnvironmentConfig(roles=(Role.MAFIA, Role.DETECTIVE, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER))
    state = new_game(config)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=2))
    assert state.phase == Phase.NIGHT_DETECTIVE

    state = apply_action(state, Action(actor=1, action_type=ActionType.INVESTIGATE, target=0))
    assert state.phase == Phase.DAY_ANNOUNCEMENT


def test_night_flow_skips_missing_detective_phase() -> None:
    config = EnvironmentConfig(roles=(Role.MAFIA, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER))
    state = new_game(config)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=2))
    assert state.phase == Phase.NIGHT_DOCTOR

    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    assert state.phase == Phase.DAY_ANNOUNCEMENT
    assert state.alive[2] is False
    assert state.last_night_outcome is not None
    assert state.last_night_outcome.victim == 2


def test_night_flow_without_support_roles_still_resolves_kill() -> None:
    config = EnvironmentConfig(roles=(Role.MAFIA, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER))
    state = new_game(config)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))

    assert state.phase == Phase.DAY_ANNOUNCEMENT
    assert state.alive[4] is False
    assert state.last_night_outcome is not None
    assert state.last_night_outcome.victim == 4


def test_noop_night_action_without_support_roles_still_resolves_kill() -> None:
    config = EnvironmentConfig(roles=(Role.MAFIA, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER))
    state = new_game(config)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NOOP))

    assert state.phase == Phase.DAY_ANNOUNCEMENT
    assert state.alive == (True, True, True, True, True)
    assert state.last_night_outcome is not None
    assert state.last_night_outcome.victim is None


def test_noop_doctor_action_without_detective_still_resolves_kill() -> None:
    config = EnvironmentConfig(roles=(Role.MAFIA, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER))
    state = new_game(config)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=2))
    state = apply_action(state, Action(actor=1, action_type=ActionType.NOOP))

    assert state.phase == Phase.DAY_ANNOUNCEMENT
    assert state.alive[2] is False
    assert state.last_night_outcome is not None
    assert state.last_night_outcome.victim == 2


def test_doctor_save_prevents_night_elimination() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))

    assert state.phase == Phase.DAY_ANNOUNCEMENT
    assert state.alive[3] is True
    assert state.last_night_outcome is not None
    assert state.last_night_outcome.saved is True


def test_detective_only_sees_private_investigation_results() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))

    detective_view = build_observation(state, 2)
    villager_view = build_observation(state, 3)

    assert detective_view.private_state.own_role == Role.DETECTIVE
    assert detective_view.private_state.investigation_results[0].target == 0
    assert villager_view.private_state.investigation_results == ()


def test_voting_tie_break_uses_lowest_player_id() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)

    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.ACCUSE, message="vote"),
        )

    votes = {
        0: 1,
        1: 0,
        2: 1,
        3: 0,
        4: 2,
        5: 2,
    }
    for actor, target in votes.items():
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=target))

    state = advance_phase(state)

    assert state.alive[0] is False
    assert state.elimination_history[-1].player == 0


def test_invalid_action_is_normalized_to_noop_and_logged() -> None:
    state = new_game()
    submitted = Action(actor=3, action_type=ActionType.NIGHT_KILL, target=1)
    validation = validate_action(state, submitted)

    assert validation.is_valid is False
    assert validation.normalized_action.action_type == ActionType.NOOP
    assert ValidationErrorCode.INVALID_ACTION_TYPE in validation.errors

    state = apply_action(state, submitted)
    assert state.phase == Phase.NIGHT_MAFIA
    assert state.validation_log[-1].submitted_action == submitted


def test_invalid_day_vote_does_not_create_placeholder_ballot() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.DEFEND, message="hold"),
        )

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=1))

    assert state.phase == Phase.DAY_VOTING
    assert state.current_voters == (0,)
    assert state.current_votes == ()
    assert state.vote_history == ()
    assert state.validation_log[-1].submitted_action.action_type == ActionType.NIGHT_KILL


def test_wrong_phase_error_is_reported_for_vote_during_night() -> None:
    state = new_game()

    validation = validate_action(state, Action(actor=0, action_type=ActionType.VOTE, target=1))

    assert validation.is_valid is False
    assert ValidationErrorCode.WRONG_PHASE in validation.errors


def test_day_announcement_skips_directly_to_voting_when_discussion_disabled() -> None:
    state = new_game(EnvironmentConfig(discussion_rounds=0))
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))

    state = advance_phase(state)

    assert state.phase == Phase.DAY_VOTING
    assert build_observation(state, 0).public_state.current_speaker is None


def test_explicit_noop_during_voting_skips_ballot_without_affecting_tally() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.DEFEND, message="hold"),
        )

    state = apply_action(state, Action(actor=0, action_type=ActionType.NOOP))
    for actor in (1, 2, 3, 4, 5):
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=1))
    state = advance_phase(state)

    assert state.elimination_history[-1].player == 1
    assert state.alive[1] is False


def test_doctor_repeat_target_can_be_disabled() -> None:
    config = EnvironmentConfig(doctor_can_repeat_target=False)
    state = new_game(config=config)
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.DEFEND, message="hold"),
        )
    for actor in tuple(state.living_players):
        target = 4 if actor != 4 else 3
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=target))
    state = advance_phase(state)

    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    illegal = validate_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    assert ValidationErrorCode.REPEATED_PROTECT_FORBIDDEN in illegal.errors


def test_duplicate_vote_reports_vote_already_cast() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.DEFEND, message="hold"),
        )

    state = apply_action(state, Action(actor=0, action_type=ActionType.VOTE, target=1))
    validation = validate_action(state, Action(actor=0, action_type=ActionType.VOTE, target=1))

    assert validation.is_valid is False
    assert ValidationErrorCode.VOTE_ALREADY_CAST in validation.errors


def test_first_legal_policy_returns_contract_action() -> None:
    observation = build_observation(new_game(), 0)
    policy = FirstLegalPolicy()

    action = policy.act(observation)

    assert action.action_type == ActionType.NIGHT_KILL
    assert action.actor == 0


def test_detective_can_receive_exact_role_when_configured() -> None:
    state = new_game(EnvironmentConfig(detective_returns_alignment_only=False))
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))

    detective_view = build_observation(state, 2)

    assert detective_view.private_state.investigation_results[0].role == Role.MAFIA


def test_hidden_roles_on_death_redact_elimination_history() -> None:
    state = new_game(EnvironmentConfig(reveal_roles_on_death=False))
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))

    assert state.elimination_history[-1].role is None


def test_exceeding_max_days_awards_town_and_ends_game() -> None:
    state = new_game(EnvironmentConfig(max_days=1))
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.DEFEND, message="hold"),
        )
    state = apply_action(state, Action(actor=0, action_type=ActionType.NOOP))
    for actor in (1, 2, 3, 4, 5):
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=1))

    state = advance_phase(state)

    assert state.day == 2
    assert state.phase == Phase.TERMINAL
    assert state.winner == WinCondition.TOWN


def test_town_wins_when_mafia_is_eliminated() -> None:
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=4))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)

    for _ in range(len(state.living_players) * state.config.discussion_rounds):
        speaker = state.living_players[state.discussion_turn_index % len(state.living_players)]
        state = apply_action(
            state,
            Action(actor=speaker, action_type=ActionType.SPEAK, target=0, intent=DiscussionIntent.ACCUSE, message="0"),
        )
    for actor in tuple(state.living_players):
        target = 0 if actor != 0 else 1
        state = apply_action(state, Action(actor=actor, action_type=ActionType.VOTE, target=target))
    state = advance_phase(state)

    assert state.winner == WinCondition.TOWN
    assert state.phase == Phase.TERMINAL
