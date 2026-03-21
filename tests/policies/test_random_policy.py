from contracts import ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import RandomLegalPolicy


def _observation(*legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=2,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
        legal_actions=legal_actions,
    )


def test_random_legal_policy_returns_noop_without_legal_actions() -> None:
    action = RandomLegalPolicy(seed=7).act(_observation())

    assert action.actor == 2
    assert action.action_type == ActionType.NOOP
    assert action.target is None
    assert action.intent is None


def test_random_legal_policy_preserves_actor_and_chooses_legal_fields() -> None:
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        ),
        LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(0, 1, 3, 4)),
    )

    action = RandomLegalPolicy(seed=3).act(observation)

    assert action.actor == 2
    assert action.action_type in {ActionType.SPEAK, ActionType.VOTE}
    if action.action_type == ActionType.SPEAK:
        assert action.target in (0, 1, 3, 4)
        assert action.intent in (DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND)
        assert action.message == "random-policy"
    else:
        assert action.target in (0, 1, 3, 4)
        assert action.intent is None
        assert action.message is None


def test_random_legal_policy_uses_only_available_choices() -> None:
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(4,),
            legal_intents=(DiscussionIntent.QUESTION,),
            allow_message=True,
        )
    )

    action = RandomLegalPolicy(seed=11).act(observation)

    assert action.actor == 2
    assert action.action_type == ActionType.SPEAK
    assert action.target == 4
    assert action.intent == DiscussionIntent.QUESTION
    assert action.message == "random-policy"
