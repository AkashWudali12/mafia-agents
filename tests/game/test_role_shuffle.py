from contracts import EnvironmentConfig, Role
from game import new_game


def test_new_game_can_shuffle_role_assignment_deterministically() -> None:
    config = EnvironmentConfig(shuffle_roles_each_game=True)

    first = new_game(config=config, seed=17)
    second = new_game(config=config, seed=17)
    third = new_game(config=config, seed=18)

    assert first.roles == second.roles
    assert sorted(first.roles) == sorted(config.roles)
    assert first.roles != config.roles
    assert third.roles != first.roles


def test_new_game_preserves_fixed_roles_when_shuffle_disabled() -> None:
    config = EnvironmentConfig(
        roles=(Role.MAFIA, Role.DOCTOR, Role.DETECTIVE, Role.VILLAGER, Role.VILLAGER),
        shuffle_roles_each_game=False,
    )

    state = new_game(config=config, seed=17)

    assert state.roles == config.roles
