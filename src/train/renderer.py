from __future__ import annotations

from contracts import Observation
from policies.rendering import render_model_prompt


def render_observation_prompt(observation: Observation, *, transcript_window: int = 8) -> str:
    return render_model_prompt(
        observation,
        max_transcript_events=transcript_window,
    )
