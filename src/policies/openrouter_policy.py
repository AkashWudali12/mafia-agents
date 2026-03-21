from __future__ import annotations

from typing import Any, Protocol

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
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_transcript_events = max_transcript_events

    def act(self, observation: Observation) -> Action:
        prompt = render_model_prompt(observation, max_transcript_events=self._max_transcript_events)
        try:
            payload = self._client.complete(
                prompt=prompt,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception:
            return noop_action(observation.actor)
        parsed = parse_action_payload(payload, actor=observation.actor)
        return normalize_action_for_observation(parsed, observation)
