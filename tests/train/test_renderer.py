from game import build_observation, new_game
from train import render_observation_prompt


def test_render_observation_prompt_includes_core_sections() -> None:
    observation = build_observation(new_game(), 0)

    prompt = render_observation_prompt(observation)

    assert "You are playing Mafia in a structured environment." in prompt
    assert "Return exactly one JSON object and no extra text." in prompt
    assert "actor: 0" in prompt
    assert "role: mafia" in prompt
    assert "day: 1" in prompt
    assert "phase: night_mafia" in prompt
    assert "living_players: 0, 1, 2, 3, 4, 5" in prompt
    assert "legal_actions:" in prompt
    assert '"action_type"' in prompt


def test_render_observation_prompt_renders_empty_transcript_compactly() -> None:
    observation = build_observation(new_game(), 0)

    prompt = render_observation_prompt(observation)

    assert "transcript:" in prompt
    assert "- none" in prompt
