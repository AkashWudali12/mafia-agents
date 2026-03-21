from contracts import ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import ScriptedMafiaPolicy


def _observation(phase: Phase, *legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=0,
        public_state=PublicObservationState(day=1, phase=phase, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.MAFIA),
        legal_actions=legal_actions,
    )


def test_scripted_mafia_returns_noop_without_legal_actions() -> None:
    action = ScriptedMafiaPolicy().act(_observation(Phase.NIGHT_MAFIA))

    assert action.actor == 0
    assert action.action_type == ActionType.NOOP


def test_scripted_mafia_chooses_lowest_night_kill_target() -> None:
    observation = _observation(
        Phase.NIGHT_MAFIA,
        LegalActionSpec(action_type=ActionType.NIGHT_KILL, legal_targets=(1, 2, 3, 4)),
    )

    action = ScriptedMafiaPolicy().act(observation)

    assert action.actor == 0
    assert action.action_type == ActionType.NIGHT_KILL
    assert action.target == 1


def test_scripted_mafia_uses_villager_style_day_fallback() -> None:
    observation = _observation(
        Phase.DAY_DISCUSSION,
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 2, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        ),
    )

    action = ScriptedMafiaPolicy().act(observation)

    assert action.actor == 0
    assert action.action_type == ActionType.SPEAK
    assert action.target == 1
    assert action.intent == DiscussionIntent.ACCUSE
