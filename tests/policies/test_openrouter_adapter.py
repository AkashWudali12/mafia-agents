from contracts import ActionType, DiscussionIntent, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role
from policies.openrouter_policy import OpenRouterPolicy


class StubClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int):
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response


class FailingClient:
    def complete(self, *, prompt: str, model: str, temperature: float, max_tokens: int):
        raise RuntimeError("network failure")


def _observation(*legal_actions: LegalActionSpec) -> Observation:
    return Observation(
        actor=1,
        public_state=PublicObservationState(day=2, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
        legal_actions=legal_actions,
    )


def test_openrouter_policy_returns_normalized_action_for_valid_payload() -> None:
    client = StubClient(
        '{"action_type":"speak","target":2,"intent":"accuse","message":"push 2"}'
    )
    policy = OpenRouterPolicy(client=client, model="test-model", max_transcript_events=2)
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 2, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    action = policy.act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.SPEAK
    assert action.target == 2
    assert action.intent == DiscussionIntent.ACCUSE
    assert action.message == "push 2"
    assert client.calls[0]["model"] == "test-model"
    assert "Return exactly one JSON object and no extra text." in client.calls[0]["prompt"]
    assert "role: doctor" in client.calls[0]["prompt"]


def test_openrouter_policy_degrades_malformed_payload_to_noop() -> None:
    client = StubClient("not-json")
    policy = OpenRouterPolicy(client=client, model="test-model")
    observation = _observation(LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(0, 2, 3, 4)))

    action = policy.act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.NOOP


def test_openrouter_policy_rejects_illegal_model_output() -> None:
    client = StubClient(
        '{"action_type":"speak","target":1,"intent":"question","message":"illegal"}'
    )
    policy = OpenRouterPolicy(client=client, model="test-model")
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 2, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    action = policy.act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.NOOP


def test_openrouter_policy_degrades_provider_failure_to_noop() -> None:
    policy = OpenRouterPolicy(client=FailingClient(), model="test-model")
    observation = _observation(
        LegalActionSpec(
            action_type=ActionType.SPEAK,
            legal_targets=(0, 2, 3, 4),
            legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
            allow_message=True,
        )
    )

    action = policy.act(observation)

    assert action.actor == 1
    assert action.action_type == ActionType.NOOP
