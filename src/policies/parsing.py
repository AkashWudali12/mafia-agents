from __future__ import annotations

import json
from typing import Any

from contracts import Action, ActionType, DiscussionIntent, Observation, ValidationResult, noop_action
from game import GameState, validate_action


def parse_action_payload(payload: str | dict[str, Any], actor: int, observation: Observation | None = None) -> Action:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(data, dict):
            return noop_action(actor)
        action_type = ActionType(data["action_type"])
        target = data.get("target")
        intent = data.get("intent")
        message = data.get("message")
        return Action(
            actor=actor,
            action_type=action_type,
            target=_resolve_target_value(target, observation),
            intent=DiscussionIntent(intent) if intent is not None else None,
            message=message if isinstance(message, str) else None,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return noop_action(actor)


def normalize_action_for_observation(action: Action, observation: Observation) -> Action:
    normalized = action.model_copy(update={"actor": observation.actor})
    if normalized.action_type == ActionType.NOOP:
        return normalized
    for legal_action in observation.legal_actions:
        if legal_action.action_type != normalized.action_type:
            continue
        if legal_action.legal_targets and normalized.target not in legal_action.legal_targets:
            continue
        if legal_action.legal_intents and normalized.intent not in legal_action.legal_intents:
            continue
        if not legal_action.allow_message:
            return normalized.model_copy(update={"message": None})
        if normalized.action_type == ActionType.SPEAK and not _has_meaningful_message(normalized.message):
            return normalized.model_copy(
                update={
                    "message": _default_speak_message(
                        intent=normalized.intent,
                        target_label=_player_label(observation, normalized.target),
                    )
                }
            )
        return normalized
    return noop_action(observation.actor)


def validate_action_for_observation(action: Action, observation: Observation, state: GameState) -> ValidationResult:
    validation = validate_action(state, action.model_copy(update={"actor": observation.actor}))
    return ValidationResult(
        is_valid=validation.is_valid,
        normalized_action=normalize_action_for_observation(validation.normalized_action, observation),
        errors=validation.errors,
    )


def _has_meaningful_message(message: str | None) -> bool:
    return message is not None and bool(message.strip())


def _default_speak_message(*, intent: DiscussionIntent | None, target_label: str | None) -> str:
    if intent == DiscussionIntent.ACCUSE and target_label is not None:
        return f"{target_label} is my strongest suspicion right now."
    if intent == DiscussionIntent.DEFEND and target_label is not None:
        return f"I want to defend {target_label} for now."
    if intent == DiscussionIntent.CLAIM_DETECTIVE and target_label is not None:
        return f"I am claiming detective and I want the table to focus on {target_label}."
    if intent == DiscussionIntent.CLAIM_DOCTOR and target_label is not None:
        return f"I am claiming doctor, so be careful about voting {target_label} too quickly."
    if intent == DiscussionIntent.CLAIM_VILLAGER:
        return "I am a villager, and I want us to compare contradictions carefully."
    if intent == DiscussionIntent.QUESTION and target_label is not None:
        return f"{target_label}, explain your position to the rest of us."
    if intent == DiscussionIntent.COORDINATE and target_label is not None:
        return f"Let's coordinate around {target_label} as our next point of pressure."
    return "I want the table to compare reads before we lock in a vote."


def _resolve_target_value(target: Any, observation: Observation | None) -> int | None:
    if isinstance(target, int):
        return target
    if not isinstance(target, str) or observation is None:
        return None
    cleaned = target.strip()
    if not cleaned:
        return None
    for seat, label in enumerate(observation.public_state.player_labels):
        if cleaned.casefold() == label.casefold():
            return seat
    return None


def _player_label(observation: Observation, target: int | None) -> str | None:
    if target is None:
        return None
    labels = observation.public_state.player_labels
    if 0 <= target < len(labels):
        return labels[target]
    return f"Player {target}"
