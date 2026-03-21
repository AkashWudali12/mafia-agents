from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict

from contracts import Action, Observation, Phase, Role, ValidationErrorCode
from game import GameState, advance_phase, apply_action, build_observation, new_game, validate_action
from policies import Policy


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class RolloutStep(FrozenModel):
    step_index: int
    day: int
    phase: Phase
    actor: int | None = None
    auto_advanced: bool = False
    observation: Observation | None = None
    submitted_action: Action | None = None
    normalized_action: Action | None = None
    validation_errors: tuple[ValidationErrorCode, ...] = ()
    next_day: int
    next_phase: Phase


class EpisodeRollout(FrozenModel):
    seed: int | None = None
    trainable_seat: int
    initial_state: GameState
    final_state: GameState
    steps: tuple[RolloutStep, ...]


AUTO_ADVANCE_PHASES = frozenset({Phase.DAY_ANNOUNCEMENT, Phase.RESOLUTION})


def next_actor(state: GameState) -> int | None:
    if state.phase in AUTO_ADVANCE_PHASES or state.is_terminal:
        return None
    if state.phase == Phase.NIGHT_MAFIA:
        return _seat_for_role(state, Role.MAFIA)
    if state.phase == Phase.NIGHT_DOCTOR:
        return _seat_for_role(state, Role.DOCTOR)
    if state.phase == Phase.NIGHT_DETECTIVE:
        return _seat_for_role(state, Role.DETECTIVE)
    if state.phase == Phase.DAY_DISCUSSION:
        return build_observation(state, state.living_players[0]).public_state.current_speaker
    if state.phase == Phase.DAY_VOTING:
        for actor in state.living_players:
            if actor not in state.current_voters:
                return actor
        return None
    return None


def run_episode(
    *,
    trainable_seat: int,
    seat_policies: Mapping[int, Policy],
    seed: int | None = None,
    initial_state: GameState | None = None,
) -> EpisodeRollout:
    state = initial_state or new_game(seed=seed)
    start_state = state
    steps: list[RolloutStep] = []

    while not state.is_terminal:
        if state.phase in AUTO_ADVANCE_PHASES:
            next_state = advance_phase(state)
            steps.append(
                RolloutStep(
                    step_index=len(steps),
                    day=state.day,
                    phase=state.phase,
                    auto_advanced=True,
                    next_day=next_state.day,
                    next_phase=next_state.phase,
                )
            )
            state = next_state
            continue

        actor = next_actor(state)
        if actor is None:
            raise ValueError(f"unable to determine actor for phase {state.phase}")
        if actor not in seat_policies:
            raise ValueError(f"missing policy for actor {actor}")

        observation = build_observation(state, actor)
        submitted_action = seat_policies[actor].act(observation)
        validation = validate_action(state, submitted_action)
        next_state = apply_action(state, submitted_action)

        steps.append(
            RolloutStep(
                step_index=len(steps),
                day=state.day,
                phase=state.phase,
                actor=actor,
                observation=observation,
                submitted_action=submitted_action,
                normalized_action=validation.normalized_action,
                validation_errors=validation.errors,
                next_day=next_state.day,
                next_phase=next_state.phase,
            )
        )
        state = next_state

    return EpisodeRollout(
        seed=state.seed,
        trainable_seat=trainable_seat,
        initial_state=start_state,
        final_state=state,
        steps=tuple(steps),
    )


def _seat_for_role(state: GameState, role: Role) -> int:
    # Night phases still need the canonical role seat even if that player is dead,
    # because the engine advances dead-role turns via noop normalization.
    return state.roles.index(role)
