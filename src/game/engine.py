from __future__ import annotations

from contracts import EnvironmentConfig

from .resolution import advance_phase, apply_action
from .rules import build_observation, get_legal_actions, validate_action
from .state import GameState


def new_game(config: EnvironmentConfig | None = None, seed: int | None = None) -> GameState:
    resolved_config = config or EnvironmentConfig()
    return GameState(
        config=resolved_config,
        seed=seed,
        roles=resolved_config.roles,
        alive=tuple(True for _ in range(resolved_config.num_players)),
    )
