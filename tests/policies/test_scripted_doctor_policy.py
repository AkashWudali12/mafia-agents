from contracts import ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import ScriptedDoctorPolicy


def _observation(phase: Phase, *legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=1,
        public_state=PublicObservationState(day=1, phase=phase, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
        legal_actions=legal_actions,
    )


def test_scripted_doctor_returns_noop_without_legal_actions() -> None:
    action = ScriptedDoctorPolicy().act(_observation(Phase.NIGHT_DOCTOR))

    assert action.actor == 1
    assert action.action_type == ActionType.NOOP


def test_scripted_doctor_prefers_self_protect_when_legal() -> None:
    observation = _observation(
        Phase.NIGHT_DOCTOR,
        LegalActionSpec(action_type=ActionType.PROTECT, legal_targets=(0, 1, 2, 3, 4)),
    )

    action = ScriptedDoctorPolicy().act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.PROTECT
    assert action.target == 1


def test_scripted_doctor_uses_first_available_non_self_target_when_self_unavailable() -> None:
    observation = _observation(
        Phase.NIGHT_DOCTOR,
        LegalActionSpec(action_type=ActionType.PROTECT, legal_targets=(0, 2, 3, 4)),
    )

    action = ScriptedDoctorPolicy().act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.PROTECT
    assert action.target == 0


def test_scripted_doctor_uses_villager_style_day_fallback() -> None:
    observation = _observation(
        Phase.DAY_DISCUSSION,
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 1, 2, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        ),
    )

    action = ScriptedDoctorPolicy().act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.SPEAK
    assert action.target == 0
    assert action.intent == DiscussionIntent.ACCUSE
