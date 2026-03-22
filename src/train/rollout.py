from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping

from contracts import Observation, Phase, Role, role_alignment
from game import GameState, advance_phase, apply_action, build_observation, new_game, validate_action
from game.rules import current_speaker, seat_for_role
from policies import (
    Policy,
    ScriptedDetectivePolicy,
    ScriptedDoctorPolicy,
    ScriptedMafiaPolicy,
    ScriptedVillagerPolicy,
)
from train.logging import log_debug_event
from train.rewards import compute_outcome_reward
from train.trajectory import EpisodeMetadata, EpisodeRollout, ObservationSummary, OutcomeRecord, RolloutStep


AUTO_ADVANCE_PHASES = frozenset({Phase.DAY_ANNOUNCEMENT, Phase.RESOLUTION})


def next_actor(state: GameState) -> int | None:
    if state.phase in AUTO_ADVANCE_PHASES or state.is_terminal:
        return None
    if state.phase == Phase.NIGHT_MAFIA:
        return seat_for_role(state, Role.MAFIA)
    if state.phase == Phase.NIGHT_DOCTOR:
        return seat_for_role(state, Role.DOCTOR)
    if state.phase == Phase.NIGHT_DETECTIVE:
        return seat_for_role(state, Role.DETECTIVE)
    if state.phase == Phase.DAY_DISCUSSION:
        return current_speaker(state)
    if state.phase == Phase.DAY_VOTING:
        for actor in state.living_players:
            if actor not in state.current_voters:
                return actor
        return None
    return None


def build_scripted_policy_map(state: GameState) -> dict[int, Policy]:
    return {
        seat: _scripted_policy_for_role(role)
        for seat, role in enumerate(state.roles)
    }


def run_episode(
    *,
    trainable_seat: int,
    seat_policies: Mapping[int, Policy],
    seed: int | None = None,
    initial_state: GameState | None = None,
    logger: logging.Logger | None = None,
) -> EpisodeRollout:
    state = initial_state or new_game(seed=seed)
    start_state = state
    config_hash = _environment_config_hash(state)
    steps: list[RolloutStep] = []

    while not state.is_terminal:
        if state.phase in AUTO_ADVANCE_PHASES:
            next_state = advance_phase(state)
            log_debug_event(
                logger,
                "rollout_auto_advance",
                day=state.day,
                phase=state.phase,
                next_day=next_state.day,
                next_phase=next_state.phase,
            )
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
        policy = seat_policies[actor]
        policy_trace = _act_with_optional_metadata(policy=policy, observation=observation, state=state)
        submitted_action = policy_trace["submitted_action"]
        validation = validate_action(state, submitted_action)
        next_state = apply_action(state, submitted_action)
        log_debug_event(
            logger,
            "rollout_step",
            step_index=len(steps),
            actor=actor,
            day=state.day,
            phase=state.phase,
            is_trainable_actor=actor == trainable_seat,
            submitted_action=policy_trace["logged_submitted_action"],
            normalized_action=validation.normalized_action,
            is_valid=validation.is_valid,
            validation_errors=validation.errors,
            next_day=next_state.day,
            next_phase=next_state.phase,
            model_prompt=policy_trace["model_prompt"],
            raw_model_output=policy_trace["raw_model_output"],
            logprob=policy_trace["logprob"],
        )

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
                submitted_action=policy_trace["logged_submitted_action"],
                normalized_action=policy_trace["logged_normalized_action"],
                is_valid=policy_trace["is_valid"],
                validation_errors=policy_trace["validation_errors"],
                model_prompt=policy_trace["model_prompt"],
                raw_model_output=policy_trace["raw_model_output"],
                logprob=policy_trace["logprob"],
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
    rollout = EpisodeRollout(
        metadata=metadata,
        initial_state=start_state,
        final_state=state,
        steps=tuple(steps),
        outcome=_build_outcome(state, trainable_seat, steps),
    )
    log_debug_event(
        logger,
        "episode_complete",
        episode_id=rollout.metadata.episode_id,
        seed=rollout.metadata.seed,
        trainable_seat=trainable_seat,
        trainable_role=rollout.metadata.trainable_role,
        winner=rollout.outcome.winner,
        final_reward=rollout.outcome.final_reward,
        reward_breakdown=rollout.outcome.reward_breakdown,
        notable_events=rollout.outcome.notable_events,
    )
    return rollout


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
    final_reward, reward_breakdown = compute_outcome_reward(
        trainable_seat=trainable_seat,
        trainable_role=state.roles[trainable_seat],
        winner=state.winner,
        final_day=state.day,
        elimination_day=elimination_day,
        final_state=state,
        steps=steps,
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
        final_reward=final_reward,
        reward_breakdown=reward_breakdown,
        notable_events=tuple(notable_events),
    )


def _environment_config_hash(state: GameState) -> str:
    payload = json.dumps(state.config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _episode_id(*, seed: int | None, trainable_seat: int, config_hash: str) -> str:
    payload = f"{seed}:{trainable_seat}:{config_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _scripted_policy_for_role(role: Role) -> Policy:
    if role == Role.MAFIA:
        return ScriptedMafiaPolicy()
    if role == Role.DOCTOR:
        return ScriptedDoctorPolicy()
    if role == Role.DETECTIVE:
        return ScriptedDetectivePolicy()
    return ScriptedVillagerPolicy()


def _act_with_optional_metadata(*, policy: Policy, observation: Observation, state: GameState) -> dict[str, Any]:
    act_with_metadata = getattr(policy, "act_with_metadata", None)
    if callable(act_with_metadata):
        trace = act_with_metadata(observation=observation, state=state)
        submitted_action = trace.submitted_action or trace.normalized_action
        validation = validate_action(state, submitted_action)
        return {
            "submitted_action": submitted_action,
            "logged_submitted_action": trace.submitted_action,
            "logged_normalized_action": validation.normalized_action,
            "is_valid": trace.is_valid and validation.is_valid,
            "validation_errors": tuple(dict.fromkeys((*trace.validation_errors, *validation.errors))),
            "model_prompt": trace.prompt,
            "raw_model_output": trace.raw_output,
            "logprob": trace.logprob,
        }

    submitted_action = policy.act(observation)
    validation = validate_action(state, submitted_action)
    return {
        "submitted_action": submitted_action,
        "logged_submitted_action": submitted_action,
        "logged_normalized_action": validation.normalized_action,
        "is_valid": validation.is_valid,
        "validation_errors": validation.errors,
        "model_prompt": None,
        "raw_model_output": None,
        "logprob": None,
    }
