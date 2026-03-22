from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from contracts import Action, Observation, noop_action

from .interfaces import Policy
from .parsing import normalize_action_for_observation, parse_action_payload
from .rendering import render_model_prompt


class OpenRouterClient(Protocol):
    def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int) -> str | dict[str, Any]:
        """Return a structured action payload as JSON text or a dict."""


class OpenRouterPolicy(Policy):
    def __init__(
        self,
        *,
        client: OpenRouterClient,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_transcript_events: int | None = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_transcript_events = max_transcript_events
        self._logger = logger

    def act(self, observation: Observation) -> Action:
        prompt = render_model_prompt(observation, max_transcript_events=self._max_transcript_events)
        try:
            payload = self._client.complete(
                prompt=prompt,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            _log_error_event(
                self._logger,
                "openrouter_policy_request_failed",
                actor=observation.actor,
                model=self._model,
                error=str(exc),
            )
            return noop_action(observation.actor)
        raw_payload = payload if isinstance(payload, dict) else str(payload)
        payload_text = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else str(payload)
        parsed = parse_action_payload(payload, actor=observation.actor)
        normalized = normalize_action_for_observation(parsed, observation)
        if parsed.action_type.value == "noop" and payload_text.strip():
            _log_error_event(
                self._logger,
                "openrouter_policy_malformed_payload",
                actor=observation.actor,
                model=self._model,
                raw_payload=raw_payload,
            )
        elif normalized.action_type.value == "noop" and parsed.action_type.value != "noop":
            _log_error_event(
                self._logger,
                "openrouter_policy_invalid_action",
                actor=observation.actor,
                model=self._model,
                parsed_action=parsed,
                raw_payload=raw_payload,
            )
        return normalized


def _log_error_event(logger: logging.Logger | None, event: str, **payload: Any) -> None:
    if logger is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **_serialize_payload(payload),
    }
    logger.error("%s", json.dumps(record, sort_keys=True, default=str))


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, BaseModel):
            serialized[key] = value.model_dump(mode="json")
        elif isinstance(value, tuple):
            serialized[key] = list(value)
        elif isinstance(value, list):
            serialized[key] = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        elif isinstance(value, dict):
            serialized[key] = _serialize_payload(value)
        else:
            serialized[key] = value
    return serialized
