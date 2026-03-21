from contracts import ActionType, DiscussionIntent, InvestigationResult, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import ScriptedDetectivePolicy


def _observation(*, phase: Phase, investigations: tuple[InvestigationResult, ...] = (), legal_actions: tuple[LegalActionSpec, ...] = ()) -> Observation:
    return Observation(
        actor=2,
        public_state=PublicObservationState(day=1, phase=phase, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DETECTIVE, investigation_results=investigations),
        legal_actions=legal_actions,
    )


def test_scripted_detective_returns_noop_without_legal_actions() -> None:
    action = ScriptedDetectivePolicy().act(_observation(phase=Phase.NIGHT_DETECTIVE))

    assert action.actor == 2
    assert action.action_type == ActionType.NOOP


def test_scripted_detective_prefers_uninvestigated_targets() -> None:
    observation = _observation(
        phase=Phase.NIGHT_DETECTIVE,
        investigations=(InvestigationResult(day=1, target=0, alignment="mafia"),),
        legal_actions=(LegalActionSpec(action_type=ActionType.INVESTIGATE, legal_targets=(0, 1, 3, 4)),),
    )

    action = ScriptedDetectivePolicy().act(observation)

    assert action.actor == 2
    assert action.action_type == ActionType.INVESTIGATE
    assert action.target == 1


def test_scripted_detective_reuses_legal_targets_when_everyone_was_seen() -> None:
    observation = _observation(
        phase=Phase.NIGHT_DETECTIVE,
        investigations=(
            InvestigationResult(day=1, target=0, alignment="mafia"),
            InvestigationResult(day=1, target=1, alignment="town"),
            InvestigationResult(day=1, target=3, alignment="town"),
            InvestigationResult(day=1, target=4, alignment="town"),
        ),
        legal_actions=(LegalActionSpec(action_type=ActionType.INVESTIGATE, legal_targets=(0, 1, 3, 4)),),
    )

    action = ScriptedDetectivePolicy().act(observation)

    assert action.actor == 2
    assert action.action_type == ActionType.INVESTIGATE
    assert action.target == 0


def test_scripted_detective_uses_villager_style_day_fallback() -> None:
    observation = _observation(
        phase=Phase.DAY_DISCUSSION,
        legal_actions=(
            LegalActionSpec(
                action_type=ActionType.SPEAK,
                legal_targets=(0, 1, 2, 3, 4),
                legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
                allow_message=True,
            ),
        ),
    )

    action = ScriptedDetectivePolicy().act(observation)

    assert action.actor == 2
    assert action.action_type == ActionType.SPEAK
    assert action.target == 0
    assert action.intent == DiscussionIntent.ACCUSE
