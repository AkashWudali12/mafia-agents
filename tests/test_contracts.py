import pytest
from pydantic import ValidationError

from contracts import (
    Action,
    ActionType,
    EnvironmentConfig,
    LegalActionSpec,
    Observation,
    Phase,
    PrivateObservationState,
    PublicObservationState,
    Role,
)


def test_environment_config_serializes_fixed_v1_defaults() -> None:
    config = EnvironmentConfig()
    dumped = config.model_dump()

    assert dumped["num_players"] == 6
    assert dumped["roles"] == (
        Role.MAFIA,
        Role.DOCTOR,
        Role.DETECTIVE,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
    )
    assert dumped["invalid_action_behavior"] == "noop"


def test_environment_config_accepts_supported_five_player_variant() -> None:
    config = EnvironmentConfig(
        roles=(Role.MAFIA, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER)
    )

    assert config.num_players == 5
    assert config.roles.count(Role.DOCTOR) == 1
    assert config.roles.count(Role.DETECTIVE) == 0


def test_environment_config_accepts_supported_six_player_variant() -> None:
    config = EnvironmentConfig(
        num_players=6,
        roles=(Role.MAFIA, Role.DOCTOR, Role.DETECTIVE, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER),
    )

    assert config.num_players == 6
    assert config.roles.count(Role.VILLAGER) == 3


def test_environment_config_rejects_non_supported_role_mix() -> None:
    with pytest.raises(ValidationError):
        EnvironmentConfig(roles=(Role.MAFIA, Role.DOCTOR, Role.DOCTOR, Role.VILLAGER, Role.VILLAGER))


def test_observation_includes_legal_actions() -> None:
    observation = Observation(
        actor=0,
        public_state=PublicObservationState(day=1, phase=Phase.NIGHT_MAFIA, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.MAFIA),
        legal_actions=(LegalActionSpec(action_type=ActionType.NIGHT_KILL, legal_targets=(1, 2, 3, 4)),),
    )

    assert observation.legal_actions[0].action_type == ActionType.NIGHT_KILL
    assert observation.legal_actions[0].legal_targets == (1, 2, 3, 4)


def test_action_requires_explicit_type() -> None:
    action = Action(actor=2, action_type=ActionType.VOTE, target=0)

    assert action.actor == 2
    assert action.target == 0
