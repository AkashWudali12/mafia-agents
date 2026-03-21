from contracts import Action, ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies.parsing import normalize_action_for_observation, parse_action_payload


def _observation(*legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=4,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
        legal_actions=legal_actions,
    )


def test_parse_action_payload_parses_valid_structured_payload() -> None:
    action = parse_action_payload(
        '{"action_type":"speak","target":1,"intent":"question","message":"who do we trust?"}',
        actor=4,
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
