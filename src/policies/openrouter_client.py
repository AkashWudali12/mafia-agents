from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from .schemas import ModelActionPayload


class PydanticAiOpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._app_url = app_url
        self._app_title = app_title

    def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        provider = OpenRouterProvider(
            api_key=self._resolve_api_key(),
            app_url=self._app_url,
            app_title=self._app_title,
        )
        openrouter_model = OpenRouterModel(model, provider=provider)
        agent = Agent(openrouter_model, output_type=ModelActionPayload)
        result = agent.run_sync(
            prompt,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        return result.output.model_dump(mode="json", exclude_none=True)

    def _resolve_api_key(self) -> str:
        api_key = self._api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for real OpenRouter model calls")
        return api_key
