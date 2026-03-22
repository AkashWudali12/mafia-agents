from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from contracts import Action, Observation, ValidationErrorCode, noop_action
from game import GameState
from policies.parsing import parse_action_payload, validate_action_for_observation
from policies.schemas import ModelActionPayload


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


ActionOutput = ModelActionPayload


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
    observation: Observation,
    state: GameState,
    raw_output: str | None = None,
) -> ParsedActionResult:
    submitted_action = parse_action_payload(
        {
            "action_type": output.action_type.value,
            "target": output.target,
            "intent": output.intent.value if output.intent is not None else None,
            "message": output.message,
        },
        actor=actor,
        observation=observation,
    )
    validation = validate_action_for_observation(submitted_action, observation, state)
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
