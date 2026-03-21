from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from contracts import Action, ActionType, DiscussionIntent, ValidationErrorCode, noop_action
from game import GameState, validate_action


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class ActionOutput(FrozenModel):
    action_type: ActionType
    target: int | None = None
    intent: DiscussionIntent | None = None
    message: str | None = None


class ParsedActionResult(FrozenModel):
    actor: int
    raw_output: str | None = None
    submitted_action: Action | None = None
    normalized_action: Action
    is_valid: bool = False
    validation_errors: tuple[ValidationErrorCode, ...] = ()
    parse_error: str | None = None


def adapt_action_output(
    *,
    actor: int,
    output: ActionOutput,
    state: GameState,
    raw_output: str | None = None,
) -> ParsedActionResult:
    submitted_action = Action(
        actor=actor,
        action_type=output.action_type,
        target=output.target,
        intent=output.intent,
        message=output.message,
    )
    validation = validate_action(state, submitted_action)
    return ParsedActionResult(
        actor=actor,
        raw_output=raw_output,
        submitted_action=submitted_action,
        normalized_action=validation.normalized_action,
        is_valid=validation.is_valid,
        validation_errors=validation.errors,
    )


def malformed_action_result(*, actor: int, raw_output: str | None = None, parse_error: str | None = None) -> ParsedActionResult:
    return ParsedActionResult(
        actor=actor,
        raw_output=raw_output,
        normalized_action=noop_action(actor),
        validation_errors=(ValidationErrorCode.MALFORMED_ACTION,),
        parse_error=parse_error,
    )
