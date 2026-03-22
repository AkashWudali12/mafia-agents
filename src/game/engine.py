from __future__ import annotations

import random

from contracts import EnvironmentConfig

from .resolution import advance_phase, apply_action
from .rules import build_observation, get_legal_actions, validate_action
from .state import GameState


def new_game(config: EnvironmentConfig | None = None, seed: int | None = None) -> GameState:
    resolved_config = config or EnvironmentConfig()
    roles = resolved_config.roles
    player_labels = _player_labels(resolved_config, seed)
    if resolved_config.shuffle_roles_each_game:
        shuffled_roles = list(roles)
        random.Random(seed).shuffle(shuffled_roles)
        roles = tuple(shuffled_roles)
    return GameState(
        config=resolved_config,
        seed=seed,
        roles=roles,
        player_labels=player_labels,
        alive=tuple(True for _ in range(resolved_config.num_players)),
    )


def _player_labels(config: EnvironmentConfig, seed: int | None) -> tuple[str, ...]:
    labels = [f"Player {chr(ord('A') + index)}" for index in range(config.num_players)]
    if config.shuffle_player_labels_each_game:
        random.Random(None if seed is None else seed + 10_000).shuffle(labels)
    return tuple(labels)
