from __future__ import annotations

import hashlib
import json
from typing import Mapping

from contracts import Observation, Phase, Role, role_alignment
from game import GameState, advance_phase, apply_action, build_observation, new_game, validate_action
from policies import Policy
from train.trajectory import EpisodeMetadata, EpisodeRollout, ObservationSummary, OutcomeRecord, RolloutStep


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
    config_hash = _environment_config_hash(state)
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
                is_trainable_actor=actor == trainable_seat,
                observation_summary=_observation_summary(observation),
                observation=observation,
                legal_actions=observation.legal_actions,
                submitted_action=submitted_action,
                normalized_action=validation.normalized_action,
                is_valid=validation.is_valid,
                validation_errors=validation.errors,
                next_day=next_state.day,
                next_phase=next_state.phase,
            )
        )
        state = next_state

    metadata = EpisodeMetadata(
        episode_id=_episode_id(seed=state.seed, trainable_seat=trainable_seat, config_hash=config_hash),
        seed=state.seed,
        trainable_seat=trainable_seat,
        trainable_role=state.roles[trainable_seat],
        trainable_alignment=role_alignment(state.roles[trainable_seat]),
        environment_config_hash=config_hash,
    )
    return EpisodeRollout(
        metadata=metadata,
        initial_state=start_state,
        final_state=state,
        steps=tuple(steps),
        outcome=_build_outcome(state, trainable_seat, steps),
    )


def _seat_for_role(state: GameState, role: Role) -> int:
    # Night phases still need the canonical role seat even if that player is dead,
    # because the engine advances dead-role turns via noop normalization.
    return state.roles.index(role)


def _observation_summary(observation: Observation) -> ObservationSummary:
    public_state = observation.public_state
    private_state = observation.private_state
    return ObservationSummary(
        day=public_state.day,
        phase=public_state.phase,
        living_players=public_state.living_players,
        own_role=private_state.own_role,
        current_speaker=public_state.current_speaker,
        discussion_round_index=public_state.discussion_round_index,
        transcript_length=len(public_state.transcript),
        vote_history_length=len(public_state.vote_history),
        elimination_history_length=len(public_state.elimination_history),
        investigation_result_count=len(private_state.investigation_results),
    )


def _build_outcome(state: GameState, trainable_seat: int, steps: list[RolloutStep]) -> OutcomeRecord:
    elimination_day = next(
        (record.day for record in state.elimination_history if record.player == trainable_seat),
        None,
    )
    notable_events: list[str] = []
    if any(step.validation_errors for step in steps):
        notable_events.append("invalid_actions_present")
    if elimination_day is not None:
        notable_events.append(f"trainable_eliminated_day_{elimination_day}")
    else:
        notable_events.append("trainable_survived_to_terminal")
    if state.winner is not None:
        notable_events.append(f"winner_{state.winner}")
    return OutcomeRecord(
        winner=state.winner,
        trainable_survived=state.alive[trainable_seat],
        elimination_day=elimination_day,
        num_days_reached=state.day,
        notable_events=tuple(notable_events),
    )


def _environment_config_hash(state: GameState) -> str:
    payload = json.dumps(state.config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _episode_id(*, seed: int | None, trainable_seat: int, config_hash: str) -> str:
    payload = f"{seed}:{trainable_seat}:{config_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
