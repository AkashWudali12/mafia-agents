from contracts import ActionType, DiscussionIntent
from policies.openrouter_client import PydanticAiOpenRouterClient
from policies.schemas import ModelActionPayload


class FakeRunResult:
    def __init__(self, output: ModelActionPayload) -> None:
        self.output = output


class FakeAgent:
    def __init__(self, model, output_type):
        self.model = model
        self.output_type = output_type

    def run_sync(self, prompt, model_settings):
        FakeAgent.last_call = {
            "prompt": prompt,
            "model_settings": model_settings,
        }
        return FakeRunResult(
            ModelActionPayload(
                action_type=ActionType.SPEAK,
                target=2,
                intent=DiscussionIntent.ACCUSE,
                message="push 2",
            )
        )


class FakeProvider:
    def __init__(self, *, api_key, app_url, app_title):
        self.api_key = api_key
        self.app_url = app_url
        self.app_title = app_title


class FakeOpenRouterModel:
    def __init__(self, model_name, *, provider):
        self.model_name = model_name
        self.provider = provider


def test_openrouter_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    client = PydanticAiOpenRouterClient()

    try:
        client.complete(prompt="hello", model="openai/gpt-4o-mini", temperature=0.0, max_tokens=64)
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key to raise")


def test_openrouter_client_returns_json_ready_payload(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("policies.openrouter_client.Agent", FakeAgent)
    monkeypatch.setattr("policies.openrouter_client.OpenRouterProvider", FakeProvider)
    monkeypatch.setattr("policies.openrouter_client.OpenRouterModel", FakeOpenRouterModel)

    client = PydanticAiOpenRouterClient(app_url="https://example.com", app_title="Mafia Agents")
    payload = client.complete(
        prompt="rendered observation",
        model="openai/gpt-4o-mini",
        temperature=0.25,
        max_tokens=96,
    )

    assert payload == {
        "action_type": "speak",
        "target": 2,
        "intent": "accuse",
        "message": "push 2",
    }
    assert FakeAgent.last_call["prompt"] == "rendered observation"
    assert FakeAgent.last_call["model_settings"] == {"temperature": 0.25, "max_tokens": 96}
