from game import build_observation, new_game
from train import render_observation_prompt


def test_render_observation_prompt_includes_core_sections() -> None:
    observation = build_observation(new_game(), 0)

    prompt = render_observation_prompt(observation)

    assert "You are player 0." in prompt
    assert "Role: mafia" in prompt
    assert "Day: 1" in prompt
    assert "Phase: night_mafia" in prompt
    assert "Living players: [0, 1, 2, 3, 4]" in prompt
    assert "Private information:" in prompt
    assert "Legal actions:" in prompt
    assert '{"action_type": "...", "target": ..., "intent": "...", "message": "..."}' in prompt


def test_render_observation_prompt_renders_empty_transcript_compactly() -> None:
    observation = build_observation(new_game(), 0)

    prompt = render_observation_prompt(observation)

    assert "Recent transcript:" in prompt
    assert "- none" in prompt
