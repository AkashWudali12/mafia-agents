import os

import pytest

from contracts import Action, ActionType
from game import advance_phase, apply_action, build_observation, new_game
from policies.openrouter_client import PydanticAiOpenRouterClient
from policies.openrouter_policy import OpenRouterPolicy


def _discussion_observation():
    state = new_game()
    state = apply_action(state, Action(actor=0, action_type=ActionType.NIGHT_KILL, target=4))
    state = apply_action(state, Action(actor=1, action_type=ActionType.PROTECT, target=3))
    state = apply_action(state, Action(actor=2, action_type=ActionType.INVESTIGATE, target=0))
    state = advance_phase(state)
    speaker = build_observation(state, state.living_players[0]).public_state.current_speaker
    assert speaker is not None
    return build_observation(state, speaker)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENROUTER_LIVE_TESTS") != "1" or not os.getenv("OPENROUTER_API_KEY"),
    reason="requires RUN_OPENROUTER_LIVE_TESTS=1 and OPENROUTER_API_KEY",
)


def test_openrouter_live_model_can_return_a_validated_action() -> None:
    observation = _discussion_observation()
    client = PydanticAiOpenRouterClient()
    policy = OpenRouterPolicy(
        client=client,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        temperature=0.0,
        max_tokens=128,
        max_transcript_events=4,
    )

    action = policy.act(observation)

    assert isinstance(action, Action)
    assert action.actor == observation.actor
    assert action.action_type in {spec.action_type for spec in observation.legal_actions} | {ActionType.NOOP}
