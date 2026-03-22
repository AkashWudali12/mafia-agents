from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from contracts import Action, Alignment, LegalActionSpec, Observation, Phase, Role, ValidationErrorCode, WinCondition
from game import GameState


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class ObservationSummary(FrozenModel):
    day: int
    phase: Phase
    living_players: tuple[int, ...]
    own_role: Role
    current_speaker: int | None = None
    discussion_round_index: int | None = None
    transcript_length: int = 0
    vote_history_length: int = 0
    elimination_history_length: int = 0
    investigation_result_count: int = 0


class RolloutStep(FrozenModel):
    step_index: int
    day: int
    phase: Phase
    actor: int | None = None
    is_trainable_actor: bool = False
    auto_advanced: bool = False
    observation_summary: ObservationSummary | None = None
    observation: Observation | None = None
    legal_actions: tuple[LegalActionSpec, ...] = ()
    submitted_action: Action | None = None
    normalized_action: Action | None = None
    is_valid: bool | None = None
    validation_errors: tuple[ValidationErrorCode, ...] = ()
    model_prompt: str | None = None
    raw_model_output: str | None = None
    logprob: float | None = None
    group_normalized_score: float | None = None
    next_day: int
    next_phase: Phase


class EpisodeMetadata(FrozenModel):
    episode_id: str
    seed: int | None = None
    checkpoint_id: str | None = None
    group_id: str | None = None
    trainable_seat: int
    trainable_role: Role
    trainable_alignment: Alignment
    opponent_pool_id: str | None = None
    opponent_model_names: tuple[str, ...] = ()
    opponent_prompt_version: str | None = None
    opponent_cache_behavior: str | None = None
    environment_config_hash: str


class OutcomeRecord(FrozenModel):
    winner: WinCondition | None = None
    trainable_survived: bool
    elimination_day: int | None = None
    num_days_reached: int
    final_reward: float | None = None
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    notable_events: tuple[str, ...] = ()


class EpisodeRollout(FrozenModel):
    metadata: EpisodeMetadata
    initial_state: GameState
    final_state: GameState
    steps: tuple[RolloutStep, ...]
    outcome: OutcomeRecord


class GroupBatchRecord(FrozenModel):
    group_id: str
    episode_ids: tuple[str, ...]
    raw_rewards: tuple[float | None, ...]
    normalized_group_scores: tuple[float | None, ...]
    best_episode_id: str | None = None
    worst_episode_id: str | None = None
