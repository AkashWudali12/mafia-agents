from __future__ import annotations

import json
from typing import Any

from contracts import Action, ActionType, DiscussionIntent, Observation, noop_action


def parse_action_payload(payload: str | dict[str, Any], actor: int) -> Action:
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
            target=target if isinstance(target, int) else None,
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
        return normalized
    return noop_action(observation.actor)
