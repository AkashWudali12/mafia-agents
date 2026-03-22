from contracts import Action, ActionType, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies import (
    FirstLegalPolicy,
    OpenRouterPolicy,
    Policy,
    RandomLegalPolicy,
    ScriptedDetectivePolicy,
    ScriptedDoctorPolicy,
    ScriptedMafiaPolicy,
    ScriptedVillagerPolicy,
    normalize_action_for_observation,
    parse_action_payload,
    render_observation,
)


def test_policy_symbols_are_importable() -> None:
    policy: Policy = FirstLegalPolicy()

    assert isinstance(policy, FirstLegalPolicy)
    assert RandomLegalPolicy is not None
    assert ScriptedVillagerPolicy is not None
    assert ScriptedDoctorPolicy is not None
    assert ScriptedDetectivePolicy is not None
    assert ScriptedMafiaPolicy is not None
    assert OpenRouterPolicy is not None
    assert parse_action_payload is not None
    assert normalize_action_for_observation is not None
    assert render_observation is not None


def test_first_legal_policy_obeys_observation_to_action_contract() -> None:
    observation = Observation(
        actor=0,
        public_state=PublicObservationState(day=1, phase=Phase.NIGHT_MAFIA, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.MAFIA),
    )

    action = FirstLegalPolicy().act(observation)

    assert isinstance(action, Action)
    assert action.actor == 0
    assert action.action_type == ActionType.NOOP
