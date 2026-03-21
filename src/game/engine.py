from __future__ import annotations

from pydantic import Field

from contracts import (
    Action,
    ActionType,
    Alignment,
    DiscussionIntent,
    EliminationRecord,
    EnvironmentConfig,
    FrozenModel,
    InvestigationResult,
    LegalActionSpec,
    NightOutcome,
    Observation,
    Phase,
    PrivateObservationState,
    PublicObservationState,
    Role,
    TieBreakRule,
    TranscriptEvent,
    ValidationErrorCode,
    ValidationLogEntry,
    ValidationResult,
    VoteRecord,
    WinCondition,
    noop_action,
    role_alignment,
)


class PendingNightActions(FrozenModel):
    mafia_target: int | None = None
    doctor_target: int | None = None
    detective_target: int | None = None


class GameState(FrozenModel):
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    seed: int | None = None
    day: int = 1
    phase: Phase = Phase.NIGHT_MAFIA
    roles: tuple[Role, ...]
    alive: tuple[bool, ...]
    pending_night_actions: PendingNightActions = Field(default_factory=PendingNightActions)
    transcript: tuple[TranscriptEvent, ...] = ()
    vote_history: tuple[VoteRecord, ...] = ()
    elimination_history: tuple[EliminationRecord, ...] = ()
    investigation_history: tuple[InvestigationResult, ...] = ()
    validation_log: tuple[ValidationLogEntry, ...] = ()
    last_night_outcome: NightOutcome | None = None
    doctor_last_target: int | None = None
    discussion_turn_index: int = 0
    current_votes: tuple[VoteRecord, ...] = ()
    current_voters: tuple[int, ...] = ()
    winner: WinCondition | None = None

    @property
    def living_players(self) -> tuple[int, ...]:
        return tuple(index for index, is_alive in enumerate(self.alive) if is_alive)

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.TERMINAL or self.winner is not None


def new_game(config: EnvironmentConfig | None = None, seed: int | None = None) -> GameState:
    resolved_config = config or EnvironmentConfig()
    return GameState(
        config=resolved_config,
        seed=seed,
        roles=resolved_config.roles,
        alive=tuple(True for _ in range(resolved_config.num_players)),
    )


def get_legal_actions(state: GameState, actor: int) -> tuple[LegalActionSpec, ...]:
    if actor < 0 or actor >= state.config.num_players:
        return ()
    if state.is_terminal or not state.alive[actor]:
        return ()
    match state.phase:
        case Phase.NIGHT_MAFIA:
            return _night_legal_actions(state, actor, Role.MAFIA, ActionType.NIGHT_KILL)
        case Phase.NIGHT_DOCTOR:
            return _doctor_legal_actions(state, actor)
        case Phase.NIGHT_DETECTIVE:
            return _night_legal_actions(state, actor, Role.DETECTIVE, ActionType.INVESTIGATE)
        case Phase.DAY_DISCUSSION:
            if actor != _current_speaker(state):
                return ()
            return (
                LegalActionSpec(
                    action_type=ActionType.SPEAK,
                    legal_targets=state.living_players,
                    legal_intents=tuple(intent for intent in DiscussionIntent),
                    allow_message=True,
                ),
            )
        case Phase.DAY_VOTING:
            if actor in state.current_voters:
                return ()
            return (
                LegalActionSpec(
                    action_type=ActionType.VOTE,
                    legal_targets=tuple(player for player in state.living_players if player != actor),
                ),
            )
        case _:
            return ()


def validate_action(state: GameState, action: Action) -> ValidationResult:
    errors: list[ValidationErrorCode] = []
    actor = action.actor
    normalized = action
    if actor < 0 or actor >= state.config.num_players:
        return ValidationResult(
            is_valid=False,
            normalized_action=noop_action(max(actor, 0)),
            errors=(ValidationErrorCode.ACTOR_OUT_OF_RANGE,),
        )
    if state.is_terminal:
        errors.append(ValidationErrorCode.GAME_TERMINAL)
    if not state.alive[actor]:
        errors.append(ValidationErrorCode.ACTOR_DEAD)
    legal = get_legal_actions(state, actor)
    if not legal:
        errors.append(ValidationErrorCode.WRONG_ACTOR)
    matched = next((item for item in legal if item.action_type == action.action_type), None)
    if matched is None and action.action_type != ActionType.NOOP:
        errors.append(ValidationErrorCode.INVALID_ACTION_TYPE)
    if action.action_type == ActionType.SPEAK:
        if action.intent is None:
            errors.append(ValidationErrorCode.INVALID_INTENT)
    if action.action_type in {ActionType.NIGHT_KILL, ActionType.PROTECT, ActionType.INVESTIGATE, ActionType.VOTE}:
        if action.target is None:
            errors.append(ValidationErrorCode.MISSING_TARGET)
        elif action.target < 0 or action.target >= state.config.num_players:
            errors.append(ValidationErrorCode.TARGET_OUT_OF_RANGE)
        elif not state.alive[action.target]:
            errors.append(ValidationErrorCode.TARGET_DEAD)
        elif matched is not None and action.target not in matched.legal_targets:
            if action.action_type == ActionType.PROTECT and action.target == actor:
                errors.append(ValidationErrorCode.SELF_TARGET_FORBIDDEN)
            elif action.action_type == ActionType.PROTECT:
                errors.append(ValidationErrorCode.REPEATED_PROTECT_FORBIDDEN)
            else:
                errors.append(ValidationErrorCode.INVALID_ACTION_TYPE)
    if matched is not None and action.action_type == ActionType.SPEAK and action.intent not in matched.legal_intents:
        errors.append(ValidationErrorCode.INVALID_INTENT)
    if errors:
        normalized = noop_action(actor)
    return ValidationResult(
        is_valid=not errors,
        normalized_action=normalized,
        errors=tuple(dict.fromkeys(errors)),
    )


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
            return next_state.model_copy(
                update={
                    "pending_night_actions": next_state.pending_night_actions.model_copy(
                        update={"mafia_target": normalized.target}
                    ),
                    "phase": Phase.NIGHT_DOCTOR,
                }
            )
        case Phase.NIGHT_DOCTOR:
            return next_state.model_copy(
                update={
                    "pending_night_actions": next_state.pending_night_actions.model_copy(
                        update={"doctor_target": normalized.target}
                    ),
                    "doctor_last_target": normalized.target,
                    "phase": Phase.NIGHT_DETECTIVE,
                }
            )
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


def build_observation(state: GameState, actor: int) -> Observation:
    role = state.roles[actor]
    investigations = tuple(item for item in state.investigation_history if role == Role.DETECTIVE and actor == _seat_for_role(state, Role.DETECTIVE))
    return Observation(
        actor=actor,
        public_state=PublicObservationState(
            day=state.day,
            phase=state.phase,
            living_players=state.living_players,
            current_speaker=_current_speaker(state),
            discussion_round_index=_discussion_round_index(state),
            transcript=state.transcript,
            vote_history=state.vote_history,
            elimination_history=state.elimination_history,
            last_night_outcome=state.last_night_outcome,
        ),
        private_state=PrivateObservationState(
            own_role=role,
            mafia_teammates=(),
            investigation_results=investigations,
            last_protection_target=state.doctor_last_target if role == Role.DOCTOR else None,
        ),
        legal_actions=get_legal_actions(state, actor),
    )


def _night_legal_actions(
    state: GameState,
    actor: int,
    required_role: Role,
    action_type: ActionType,
) -> tuple[LegalActionSpec, ...]:
    if state.roles[actor] != required_role:
        return ()
    targets = tuple(player for player in state.living_players if player != actor)
    return (LegalActionSpec(action_type=action_type, legal_targets=targets),)


def _doctor_legal_actions(state: GameState, actor: int) -> tuple[LegalActionSpec, ...]:
    if state.roles[actor] != Role.DOCTOR:
        return ()
    targets = []
    for player in state.living_players:
        if player == actor and not state.config.doctor_can_self_protect:
            continue
        if (
            player == state.doctor_last_target
            and not state.config.doctor_can_repeat_target
        ):
            continue
        targets.append(player)
    return (LegalActionSpec(action_type=ActionType.PROTECT, legal_targets=tuple(targets)),)


def _current_speaker(state: GameState) -> int | None:
    if state.phase != Phase.DAY_DISCUSSION:
        return None
    living = state.living_players
    if not living:
        return None
    total_turns = len(living) * state.config.discussion_rounds
    if state.discussion_turn_index >= total_turns:
        return None
    return living[state.discussion_turn_index % len(living)]


def _discussion_round_index(state: GameState) -> int | None:
    if state.phase != Phase.DAY_DISCUSSION or not state.living_players:
        return None
    return state.discussion_turn_index // len(state.living_players)


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
    winner = _determine_winner(tuple(alive), state.roles)
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
    winner = _determine_winner(tuple(alive), state.roles)
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
        case Phase.NIGHT_MAFIA if actor == _seat_for_role(state, Role.MAFIA):
            return state.model_copy(update={"phase": Phase.NIGHT_DOCTOR})
        case Phase.NIGHT_DOCTOR if actor == _seat_for_role(state, Role.DOCTOR):
            return state.model_copy(update={"phase": Phase.NIGHT_DETECTIVE})
        case Phase.NIGHT_DETECTIVE if actor == _seat_for_role(state, Role.DETECTIVE):
            return _resolve_night(state)
        case Phase.DAY_DISCUSSION if actor == _current_speaker(state):
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


def _seat_for_role(state: GameState, role: Role) -> int:
    return state.roles.index(role)


def _determine_winner(alive: tuple[bool, ...], roles: tuple[Role, ...]) -> WinCondition | None:
    mafia_alive = sum(1 for is_alive, role in zip(alive, roles, strict=True) if is_alive and role == Role.MAFIA)
    town_alive = sum(1 for is_alive, role in zip(alive, roles, strict=True) if is_alive and role != Role.MAFIA)
    if mafia_alive == 0:
        return WinCondition.TOWN
    if mafia_alive >= town_alive:
        return WinCondition.MAFIA
    return None
