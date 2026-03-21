from contracts import ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import ScriptedVillagerPolicy


def _observation(*legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=3,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
        legal_actions=legal_actions,
    )


def test_scripted_villager_returns_noop_without_legal_actions() -> None:
    action = ScriptedVillagerPolicy().act(_observation())

    assert action.actor == 3
    assert action.action_type == ActionType.NOOP


def test_scripted_villager_speaks_with_legal_target_and_intent() -> None:
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    action = ScriptedVillagerPolicy().act(observation)

    assert action.actor == 3
    assert action.action_type == ActionType.SPEAK
    assert action.target == 0
    assert action.intent == DiscussionIntent.ACCUSE
    assert action.message == "Player 0 looks suspicious to me. We should pressure them."


def test_scripted_villager_votes_for_lowest_non_self_target() -> None:
    observation = _observation(
        LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(1, 2, 4))
    )

    action = ScriptedVillagerPolicy().act(observation)

    assert action.actor == 3
    assert action.action_type == ActionType.VOTE
    assert action.target == 1
    assert action.intent is None
