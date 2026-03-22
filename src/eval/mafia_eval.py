from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from contracts import EnvironmentConfig, Role
from game import new_game
from policies import Policy
from train.rollout import build_scripted_policy_map, run_episode


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class MafiaEvaluationResult(FrozenModel):
    checkpoint_id: str | None = None
    total_episodes: int
    overall_win_rate: float
    invalid_action_rate: float
    average_game_length: float
    win_rate_by_role: dict[str, float] = Field(default_factory=dict)
    average_reward_by_component: dict[str, float] = Field(default_factory=dict)


def run_mafia_evaluation(
    *,
    trainable_policy: Policy,
    seeds: Sequence[int],
    episodes_per_role: int,
    environment_config: EnvironmentConfig | None = None,
    checkpoint_id: str | None = None,
) -> MafiaEvaluationResult:
    config = environment_config or EnvironmentConfig()
    roles = sorted(set(config.roles), key=lambda role: role.value)
    role_results: dict[str, list[float]] = defaultdict(list)
    reward_components: dict[str, list[float]] = defaultdict(list)
    episodes = []

    for role in roles:
        compatible_seats = [seat for seat, seat_role in enumerate(config.roles) if seat_role == role]
        for episode_index in range(episodes_per_role):
            seed = seeds[(episode_index + len(episodes)) % len(seeds)]
            state = new_game(config=config, seed=seed)
            compatible_state_seats = [seat for seat, seat_role in enumerate(state.roles) if seat_role == role]
            trainable_seat = compatible_state_seats[episode_index % len(compatible_state_seats)]
            seat_policies = build_scripted_policy_map(state)
            seat_policies[trainable_seat] = trainable_policy
            rollout = run_episode(
                trainable_seat=trainable_seat,
                seat_policies=seat_policies,
                seed=seed,
                initial_state=state,
            )
            episodes.append(rollout)
            won = float(rollout.metadata.trainable_alignment.value == rollout.outcome.winner.value)
            role_results[role.value].append(won)
            for name, value in rollout.outcome.reward_breakdown.items():
                reward_components[name].append(value)

    invalid_steps = sum(1 for episode in episodes for step in episode.steps if step.validation_errors)
    total_steps = sum(1 for episode in episodes for step in episode.steps if step.actor is not None)
    return MafiaEvaluationResult(
        checkpoint_id=checkpoint_id,
        total_episodes=len(episodes),
        overall_win_rate=sum(sum(values) for values in role_results.values()) / len(episodes) if episodes else 0.0,
        invalid_action_rate=invalid_steps / total_steps if total_steps else 0.0,
        average_game_length=sum(episode.outcome.num_days_reached for episode in episodes) / len(episodes) if episodes else 0.0,
        win_rate_by_role={
            role: sum(values) / len(values)
            for role, values in role_results.items()
        },
        average_reward_by_component={
            name: sum(values) / len(values)
            for name, values in reward_components.items()
        },
    )
