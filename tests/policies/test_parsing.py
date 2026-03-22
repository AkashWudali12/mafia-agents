from contracts import Action, ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from game import new_game
from policies.parsing import normalize_action_for_observation, parse_action_payload, validate_action_for_observation


def _observation(*legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=4,
        public_state=PublicObservationState(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            living_players=(0, 1, 2, 3, 4),
            player_labels=("Player E", "Player A", "Player C", "Player D", "Player B"),
        ),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
        legal_actions=legal_actions,
    )


def test_parse_action_payload_parses_valid_structured_payload() -> None:
    action = parse_action_payload(
        '{"action_type":"speak","target":"Player A","intent":"question","message":"who do we trust?"}',
        actor=4,
        observation=_observation(),
    )

    assert action.actor == 4
    assert action.action_type == ActionType.SPEAK
    assert action.target == 1
    assert action.intent == DiscussionIntent.QUESTION
    assert action.message == "who do we trust?"


def test_parse_action_payload_degrades_malformed_payload_to_noop() -> None:
    action = parse_action_payload("not-json", actor=4)

    assert action.actor == 4
    assert action.action_type == ActionType.NOOP


def test_normalize_action_for_observation_fixes_actor_and_strips_disallowed_message() -> None:
    observation = _observation(
        LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(0, 1, 2, 3))
    )

    action = normalize_action_for_observation(
        Action(actor=1, action_type=ActionType.VOTE, target=2, message="should be removed"),
        observation,
    )

    assert action.actor == 4
    assert action.action_type == ActionType.VOTE
    assert action.target == 2
    assert action.message is None


def test_normalize_action_for_observation_rejects_illegal_target_and_intent() -> None:
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 2, 3),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    illegal_action = Action(
        actor=4,
        action_type=ActionType.SPEAK,
        target=4,
        intent=DiscussionIntent.QUESTION,
        message="bad",
    )

    normalized = normalize_action_for_observation(illegal_action, observation)

    assert normalized.actor == 4
    assert normalized.action_type == ActionType.NOOP


def test_normalize_action_for_observation_fills_missing_speak_message() -> None:
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 2, 3),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    normalized = normalize_action_for_observation(
        Action(actor=4, action_type=ActionType.SPEAK, target=1, intent=DiscussionIntent.ACCUSE, message=""),
        observation,
    )

    assert normalized.action_type == ActionType.SPEAK
    assert normalized.message == "Player A is my strongest suspicion right now."


def test_validate_action_for_observation_reuses_engine_validation_with_policy_normalization() -> None:
    state = new_game()
    observation = Observation(
        actor=0,
        public_state=PublicObservationState(
            day=state.day,
            phase=state.phase,
            living_players=state.living_players,
            current_speaker=None,
            discussion_round_index=None,
            transcript=state.transcript,
            vote_history=state.vote_history,
            elimination_history=state.elimination_history,
            last_night_outcome=state.last_night_outcome,
        ),
        private_state=PrivateObservationState(own_role=Role.MAFIA),
        legal_actions=(LegalActionSpec(action_type=ActionType.NIGHT_KILL, legal_targets=(1, 2, 3, 4)),),
    )

    validation = validate_action_for_observation(
        Action(actor=99, action_type=ActionType.NIGHT_KILL, target=3),
        observation,
        state,
    )

    assert validation.is_valid is True
    assert validation.normalized_action.actor == 0


def test_validate_action_for_observation_keeps_illegal_target_invalid() -> None:
    state = new_game()
    observation = Observation(
        actor=0,
        public_state=PublicObservationState(
            day=state.day,
            phase=state.phase,
            living_players=state.living_players,
            current_speaker=None,
            discussion_round_index=None,
            transcript=state.transcript,
            vote_history=state.vote_history,
            elimination_history=state.elimination_history,
            last_night_outcome=state.last_night_outcome,
        ),
        private_state=PrivateObservationState(own_role=Role.MAFIA),
        legal_actions=(LegalActionSpec(action_type=ActionType.NIGHT_KILL, legal_targets=(1, 2, 3, 4)),),
    )

    validation = validate_action_for_observation(
        Action(actor=0, action_type=ActionType.NIGHT_KILL, target=0),
        observation,
        state,
    )

    assert validation.is_valid is False
    assert validation.normalized_action.action_type == ActionType.NOOP
