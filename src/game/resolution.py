from __future__ import annotations

from contracts import (
    Action,
    ActionType,
    EliminationRecord,
    InvestigationResult,
    NightOutcome,
    Phase,
    Role,
    TieBreakRule,
    TranscriptEvent,
    ValidationLogEntry,
    VoteRecord,
    WinCondition,
    role_alignment,
)

from .rules import current_speaker, seat_for_role, validate_action
from .state import GameState, PendingNightActions


def apply_action(state: GameState, action: Action) -> GameState:
    validation = validate_action(state, action)
    normalized = validation.normalized_action
    next_state = state
    if validation.errors:
        next_state = next_state.model_copy(
            update={
                "validation_log": next_state.validation_log
                + (
                    ValidationLogEntry(
                        actor=action.actor,
                        phase=state.phase,
                        submitted_action=action,
                        normalized_action=normalized,
                        errors=validation.errors,
                    ),
                )
            }
        )
    if normalized.action_type == ActionType.NOOP:
        return _progress_after_noop(next_state, normalized.actor)
    match state.phase:
        case Phase.NIGHT_MAFIA:
            progressed = next_state.model_copy(
                update={
                    "pending_night_actions": next_state.pending_night_actions.model_copy(
                        update={"mafia_target": normalized.target}
                    ),
                    "phase": _next_phase_after_mafia(next_state),
                }
            )
            if progressed.phase == Phase.DAY_ANNOUNCEMENT:
                return _resolve_night(progressed)
            return progressed
        case Phase.NIGHT_DOCTOR:
            progressed = next_state.model_copy(
                update={
                    "pending_night_actions": next_state.pending_night_actions.model_copy(
                        update={"doctor_target": normalized.target}
                    ),
                    "doctor_last_target": normalized.target,
                    "phase": _next_phase_after_doctor(next_state),
                }
            )
            if progressed.phase == Phase.DAY_ANNOUNCEMENT:
                return _resolve_night(progressed)
            return progressed
        case Phase.NIGHT_DETECTIVE:
            resolved = next_state.model_copy(
                update={
                    "pending_night_actions": next_state.pending_night_actions.model_copy(
                        update={"detective_target": normalized.target}
                    )
                }
            )
            return _resolve_night(resolved)
        case Phase.DAY_DISCUSSION:
            spoken = next_state.model_copy(
                update={
                    "transcript": next_state.transcript
                    + (
                        TranscriptEvent(
                            day=next_state.day,
                            phase=next_state.phase,
                            speaker=normalized.actor,
                            action_type=normalized.action_type,
                            intent=normalized.intent,
                            target=normalized.target,
                            message=normalized.message,
                        ),
                    ),
                }
            )
            return _advance_discussion(spoken)
        case Phase.DAY_VOTING:
            voted = next_state.model_copy(
                update={
                    "current_votes": next_state.current_votes
                    + (VoteRecord(day=next_state.day, voter=normalized.actor, target=normalized.target),),
                    "current_voters": next_state.current_voters + (normalized.actor,),
                    "vote_history": next_state.vote_history
                    + (VoteRecord(day=next_state.day, voter=normalized.actor, target=normalized.target),),
                    "transcript": next_state.transcript
                    + (
                        TranscriptEvent(
                            day=next_state.day,
                            phase=next_state.phase,
                            speaker=normalized.actor,
                            action_type=normalized.action_type,
                            target=normalized.target,
                        ),
                    ),
                }
            )
            if len(voted.current_voters) == len(voted.living_players):
                return voted.model_copy(update={"phase": Phase.RESOLUTION})
            return voted
        case _:
            return next_state


def advance_phase(state: GameState) -> GameState:
    if state.is_terminal:
        return state.model_copy(update={"phase": Phase.TERMINAL})
    match state.phase:
        case Phase.DAY_ANNOUNCEMENT:
            next_phase = Phase.DAY_DISCUSSION if state.config.discussion_rounds > 0 else Phase.DAY_VOTING
            return state.model_copy(update={"phase": next_phase, "discussion_turn_index": 0})
        case Phase.RESOLUTION:
            return _resolve_votes(state)
        case _:
            return state


def determine_winner(alive: tuple[bool, ...], roles: tuple[Role, ...]) -> WinCondition | None:
    mafia_alive = sum(1 for is_alive, role in zip(alive, roles, strict=True) if is_alive and role == Role.MAFIA)
    town_alive = sum(1 for is_alive, role in zip(alive, roles, strict=True) if is_alive and role != Role.MAFIA)
    if mafia_alive == 0:
        return WinCondition.TOWN
    if mafia_alive >= town_alive:
        return WinCondition.MAFIA
    return None


def _advance_discussion(state: GameState) -> GameState:
    next_index = state.discussion_turn_index + 1
    total_turns = len(state.living_players) * state.config.discussion_rounds
    if next_index >= total_turns:
        return state.model_copy(update={"phase": Phase.DAY_VOTING, "discussion_turn_index": next_index})
    return state.model_copy(update={"discussion_turn_index": next_index})


def _resolve_night(state: GameState) -> GameState:
    night = state.pending_night_actions
    victim = night.mafia_target
    saved = victim is not None and victim == night.doctor_target
    alive = list(state.alive)
    elimination_history = state.elimination_history
    if victim is not None and not saved:
        alive[victim] = False
        elimination_history = elimination_history + (
            EliminationRecord(
                day=state.day,
                player=victim,
                role=state.roles[victim] if state.config.reveal_roles_on_death else None,
                reason="night_kill",
            ),
        )
    investigation_history = state.investigation_history
    if night.detective_target is not None:
        target_role = state.roles[night.detective_target]
        investigation_history = investigation_history + (
            InvestigationResult(
                day=state.day,
                target=night.detective_target,
                alignment=role_alignment(target_role),
                role=None if state.config.detective_returns_alignment_only else target_role,
            ),
        )
    winner = determine_winner(tuple(alive), state.roles)
    return state.model_copy(
        update={
            "alive": tuple(alive),
            "elimination_history": elimination_history,
            "investigation_history": investigation_history,
            "last_night_outcome": NightOutcome(
                victim=victim,
                saved=saved,
                announced_death=None if saved else victim,
            ),
            "pending_night_actions": PendingNightActions(),
            "phase": Phase.TERMINAL if winner else Phase.DAY_ANNOUNCEMENT,
            "winner": winner,
        }
    )


def _resolve_votes(state: GameState) -> GameState:
    tally: dict[int, int] = {}
    for vote in state.current_votes:
        tally[vote.target] = tally.get(vote.target, 0) + 1
    if not tally:
        return _start_next_day(state)
    top_count = max(tally.values())
    top_targets = [target for target, count in tally.items() if count == top_count]
    if state.config.tie_break_rule != TieBreakRule.LOWEST_ID:
        raise ValueError("unsupported tie break rule")
    eliminated = min(top_targets)
    alive = list(state.alive)
    alive[eliminated] = False
    elimination = EliminationRecord(
        day=state.day,
        player=eliminated,
        role=state.roles[eliminated] if state.config.reveal_roles_on_death else None,
        reason="vote",
    )
    winner = determine_winner(tuple(alive), state.roles)
    updated = state.model_copy(
        update={
            "alive": tuple(alive),
            "elimination_history": state.elimination_history + (elimination,),
            "winner": winner,
            "phase": Phase.TERMINAL if winner else state.phase,
        }
    )
    if winner:
        return updated.model_copy(update={"phase": Phase.TERMINAL})
    return _start_next_day(updated)


def _start_next_day(state: GameState) -> GameState:
    next_day = state.day + 1
    winner = state.winner
    next_phase = Phase.TERMINAL if winner or next_day > state.config.max_days else Phase.NIGHT_MAFIA
    if not winner and next_day > state.config.max_days:
        winner = WinCondition.TOWN
    return state.model_copy(
        update={
            "day": next_day,
            "phase": next_phase,
            "current_votes": (),
            "current_voters": (),
            "discussion_turn_index": 0,
            "winner": winner,
            "last_night_outcome": None,
        }
    )


def _progress_after_noop(state: GameState, actor: int) -> GameState:
    match state.phase:
        case Phase.NIGHT_MAFIA if actor == seat_for_role(state, Role.MAFIA):
            progressed = state.model_copy(update={"phase": _next_phase_after_mafia(state)})
            if progressed.phase == Phase.DAY_ANNOUNCEMENT:
                return _resolve_night(progressed)
            return progressed
        case Phase.NIGHT_DOCTOR if actor == seat_for_role(state, Role.DOCTOR):
            progressed = state.model_copy(update={"phase": _next_phase_after_doctor(state)})
            if progressed.phase == Phase.DAY_ANNOUNCEMENT:
                return _resolve_night(progressed)
            return progressed
        case Phase.NIGHT_DETECTIVE if actor == seat_for_role(state, Role.DETECTIVE):
            return _resolve_night(state)
        case Phase.DAY_DISCUSSION if actor == current_speaker(state):
            return _advance_discussion(state)
        case Phase.DAY_VOTING if actor in state.living_players and actor not in state.current_voters:
            progressed = state.model_copy(
                update={
                    "current_voters": state.current_voters + (actor,),
                }
            )
            if len(progressed.current_voters) == len(progressed.living_players):
                return progressed.model_copy(update={"phase": Phase.RESOLUTION})
            return progressed
        case _:
            return state


def _next_phase_after_mafia(state: GameState) -> Phase:
    if _living_seat_for_role(state, Role.DOCTOR) != -1:
        return Phase.NIGHT_DOCTOR
    if _living_seat_for_role(state, Role.DETECTIVE) != -1:
        return Phase.NIGHT_DETECTIVE
    return Phase.DAY_ANNOUNCEMENT


def _next_phase_after_doctor(state: GameState) -> Phase:
    if _living_seat_for_role(state, Role.DETECTIVE) != -1:
        return Phase.NIGHT_DETECTIVE
    return Phase.DAY_ANNOUNCEMENT


def _living_seat_for_role(state: GameState, role: Role) -> int:
    seat = seat_for_role(state, role)
    if seat == -1:
        return -1
    return seat if state.alive[seat] else -1
