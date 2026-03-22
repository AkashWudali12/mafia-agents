"""Smoke tests: model prompts must not embed private observation fields."""

from __future__ import annotations

import re

from contracts import (
    ActionType,
    Alignment,
    InvestigationResult,
    LegalActionSpec,
    Observation,
    Phase,
    PrivateObservationState,
    PublicObservationState,
    Role,
)
from game import new_game
from policies.rendering import render_model_prompt, render_observation
from train import build_scripted_policy_map, run_episode
from train.renderer import render_observation_prompt


def test_render_observation_excludes_private_payload_except_own_role() -> None:
    """PrivateObservationState may carry secrets; only own_role may appear in the text."""
    poison_teammate_seat = 88
    poison_protect_target = 99
    poison_investigation_target = 77
    observation = Observation(
        actor=0,
        public_state=PublicObservationState(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            living_players=(0, 1, 2),
            transcript=(),
            vote_history=(),
            elimination_history=(),
        ),
        private_state=PrivateObservationState(
            own_role=Role.VILLAGER,
            mafia_teammates=(poison_teammate_seat,),
            investigation_results=(
                InvestigationResult(
                    day=1,
                    target=poison_investigation_target,
                    alignment=Alignment.MAFIA,
                    role=None,
                ),
            ),
            last_protection_target=poison_protect_target,
        ),
        legal_actions=(),
    )

    rendered_obs = render_observation(observation)
    prompt = render_model_prompt(observation)

    for secret in (str(poison_teammate_seat), str(poison_protect_target), str(poison_investigation_target)):
        assert secret not in rendered_obs, f"leaked private numeric token {secret} into render_observation"
        assert secret not in prompt, f"leaked private numeric token {secret} into render_model_prompt"

    assert "role: villager" in rendered_obs
    assert "role: villager" in prompt


def test_train_renderer_matches_policy_prompt_path() -> None:
    observation = Observation(
        actor=1,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_VOTING, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
        legal_actions=(LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(0, 2, 3, 4)),),
    )
    assert render_observation_prompt(observation, transcript_window=8) == render_model_prompt(
        observation,
        max_transcript_events=8,
    )


_ELIM_ROLE_LINE = re.compile(
    r"-\s*day=(?P<day>\d+)\s+player=(?P<player>\d+)\s+role=(?P<role>[^\s]+)\s+reason=(?P<reason>\S+)"
)


def _elimination_section(prompt: str) -> str:
    if "elimination_history:" not in prompt:
        return ""
    after = prompt.split("elimination_history:", 1)[1]
    if "transcript:" in after:
        return after.split("transcript:", 1)[0]
    if "legal_actions:" in after:
        return after.split("legal_actions:", 1)[0]
    return after


def _assert_prompt_eliminations_match_living_players(observation: Observation, prompt: str) -> None:
    """Engine must never list a revealed role in elimination_history for a still-living seat."""
    living = set(observation.public_state.living_players)
    section = _elimination_section(prompt)
    for match in _ELIM_ROLE_LINE.finditer(section):
        player = int(match.group("player"))
        role_token = match.group("role")
        if role_token == "hidden":
            continue
        assert player not in living, (
            f"prompt shows elimination role for living player {player}: {match.group(0)!r}"
        )


def test_scripted_rollout_prompts_do_not_leak_elimination_roles_for_living_players() -> None:
    state = new_game(seed=42)
    policies = build_scripted_policy_map(state)
    result = run_episode(trainable_seat=0, seat_policies=policies, seed=42, initial_state=state)

    for step in result.steps:
        obs = step.observation
        if obs is None:
            continue
        prompt = render_model_prompt(obs)
        _assert_prompt_eliminations_match_living_players(obs, prompt)
        train_prompt = render_observation_prompt(obs, transcript_window=8)
        _assert_prompt_eliminations_match_living_players(obs, train_prompt)
