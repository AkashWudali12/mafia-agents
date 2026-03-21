from __future__ import annotations

from contracts import (
    Action,
    ActionType,
    DiscussionIntent,
    LegalActionSpec,
    Observation,
    Phase,
    PrivateObservationState,
    PublicObservationState,
    Role,
    ValidationErrorCode,
    ValidationResult,
    noop_action,
)

from .state import GameState


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
            if actor != current_speaker(state):
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
    if state.phase == Phase.DAY_VOTING and actor in state.current_voters and action.action_type == ActionType.VOTE:
        errors.append(ValidationErrorCode.VOTE_ALREADY_CAST)
    legal = get_legal_actions(state, actor)
    if not legal:
        errors.append(ValidationErrorCode.WRONG_ACTOR)
    matched = next((item for item in legal if item.action_type == action.action_type), None)
    if matched is None and action.action_type != ActionType.NOOP:
        if action_type_allowed_in_phase(state.phase, action.action_type):
            errors.append(ValidationErrorCode.INVALID_ACTION_TYPE)
        else:
            errors.append(ValidationErrorCode.WRONG_PHASE)
    if action.action_type == ActionType.SPEAK and action.intent is None:
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


def build_observation(state: GameState, actor: int) -> Observation:
    role = state.roles[actor]
    detective_seat = seat_for_role(state, Role.DETECTIVE)
    investigations = tuple(
        item
        for item in state.investigation_history
        if role == Role.DETECTIVE and actor == detective_seat
    )
    return Observation(
        actor=actor,
        public_state=PublicObservationState(
            day=state.day,
            phase=state.phase,
            living_players=state.living_players,
            current_speaker=current_speaker(state),
            discussion_round_index=discussion_round_index(state),
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


def seat_for_role(state: GameState, role: Role) -> int:
    try:
        return state.roles.index(role)
    except ValueError:
        return -1


def current_speaker(state: GameState) -> int | None:
    if state.phase != Phase.DAY_DISCUSSION:
        return None
    living = state.living_players
    if not living:
        return None
    total_turns = len(living) * state.config.discussion_rounds
    if state.discussion_turn_index >= total_turns:
        return None
    return living[state.discussion_turn_index % len(living)]


def discussion_round_index(state: GameState) -> int | None:
    if state.phase != Phase.DAY_DISCUSSION or not state.living_players:
        return None
    return state.discussion_turn_index // len(state.living_players)


def action_type_allowed_in_phase(phase: Phase, action_type: ActionType) -> bool:
    allowed_by_phase = {
        Phase.NIGHT_MAFIA: {ActionType.NIGHT_KILL},
        Phase.NIGHT_DOCTOR: {ActionType.PROTECT},
        Phase.NIGHT_DETECTIVE: {ActionType.INVESTIGATE},
        Phase.DAY_DISCUSSION: {ActionType.SPEAK},
        Phase.DAY_VOTING: {ActionType.VOTE},
    }
    return action_type in allowed_by_phase.get(phase, set())


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
        if player == state.doctor_last_target and not state.config.doctor_can_repeat_target:
            continue
        targets.append(player)
    return (LegalActionSpec(action_type=ActionType.PROTECT, legal_targets=tuple(targets)),)
