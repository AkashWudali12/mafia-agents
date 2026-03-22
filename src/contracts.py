from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class Role(StrEnum):
    MAFIA = "mafia"
    DOCTOR = "doctor"
    DETECTIVE = "detective"
    VILLAGER = "villager"


class Alignment(StrEnum):
    MAFIA = "mafia"
    TOWN = "town"


class Phase(StrEnum):
    NIGHT_MAFIA = "night_mafia"
    NIGHT_DOCTOR = "night_doctor"
    NIGHT_DETECTIVE = "night_detective"
    DAY_ANNOUNCEMENT = "day_announcement"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    RESOLUTION = "resolution"
    TERMINAL = "terminal"


class ActionType(StrEnum):
    NOOP = "noop"
    SPEAK = "speak"
    VOTE = "vote"
    NIGHT_KILL = "night_kill"
    PROTECT = "protect"
    INVESTIGATE = "investigate"


class DiscussionIntent(StrEnum):
    ACCUSE = "accuse"
    DEFEND = "defend"
    CLAIM_VILLAGER = "claim_villager"
    CLAIM_DETECTIVE = "claim_detective"
    CLAIM_DOCTOR = "claim_doctor"
    QUESTION = "question"
    COORDINATE = "coordinate"


class WinCondition(StrEnum):
    TOWN = "town"
    MAFIA = "mafia"


class TieBreakRule(StrEnum):
    LOWEST_ID = "lowest_id"


class ValidationErrorCode(StrEnum):
    GAME_TERMINAL = "game_terminal"
    ACTOR_OUT_OF_RANGE = "actor_out_of_range"
    ACTOR_DEAD = "actor_dead"
    WRONG_PHASE = "wrong_phase"
    WRONG_ACTOR = "wrong_actor"
    INVALID_ACTION_TYPE = "invalid_action_type"
    MISSING_TARGET = "missing_target"
    TARGET_OUT_OF_RANGE = "target_out_of_range"
    TARGET_DEAD = "target_dead"
    SELF_TARGET_FORBIDDEN = "self_target_forbidden"
    REPEATED_PROTECT_FORBIDDEN = "repeated_protect_forbidden"
    INVALID_INTENT = "invalid_intent"
    VOTE_ALREADY_CAST = "vote_already_cast"
    MALFORMED_ACTION = "malformed_action"


class NightOutcome(FrozenModel):
    victim: int | None = None
    saved: bool = False
    announced_death: int | None = None


class Action(FrozenModel):
    actor: int
    action_type: ActionType
    target: int | None = None
    intent: DiscussionIntent | None = None
    message: str | None = None


class ValidationResult(FrozenModel):
    is_valid: bool
    normalized_action: Action
    errors: tuple[ValidationErrorCode, ...] = ()


class ValidationLogEntry(FrozenModel):
    actor: int
    phase: Phase
    submitted_action: Action
    normalized_action: Action
    errors: tuple[ValidationErrorCode, ...]


class TranscriptEvent(FrozenModel):
    day: int
    phase: Phase
    speaker: int
    action_type: ActionType
    intent: DiscussionIntent | None = None
    target: int | None = None
    message: str | None = None


class VoteRecord(FrozenModel):
    day: int
    voter: int
    target: int


class EliminationRecord(FrozenModel):
    day: int
    player: int
    role: Role | None
    reason: str


class InvestigationResult(FrozenModel):
    day: int
    target: int
    alignment: Alignment
    role: Role | None = None


class LegalActionSpec(FrozenModel):
    action_type: ActionType
    legal_targets: tuple[int, ...] = ()
    legal_intents: tuple[DiscussionIntent, ...] = ()
    allow_message: bool = False


class PublicObservationState(FrozenModel):
    day: int
    phase: Phase
    living_players: tuple[int, ...]
    current_speaker: int | None = None
    discussion_round_index: int | None = None
    transcript: tuple[TranscriptEvent, ...] = ()
    vote_history: tuple[VoteRecord, ...] = ()
    elimination_history: tuple[EliminationRecord, ...] = ()
    last_night_outcome: NightOutcome | None = None


class PrivateObservationState(FrozenModel):
    own_role: Role
    mafia_teammates: tuple[int, ...] = ()
    investigation_results: tuple[InvestigationResult, ...] = ()
    last_protection_target: int | None = None


class Observation(FrozenModel):
    actor: int
    public_state: PublicObservationState
    private_state: PrivateObservationState
    legal_actions: tuple[LegalActionSpec, ...] = ()


class EnvironmentConfig(FrozenModel):
    num_players: int = 6
    roles: tuple[Role, ...] = (
        Role.MAFIA,
        Role.DOCTOR,
        Role.DETECTIVE,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
    )
    discussion_rounds: int = 2
    max_days: int = 10
    doctor_can_self_protect: bool = True
    doctor_can_repeat_target: bool = True
    detective_returns_alignment_only: bool = True
    reveal_roles_on_death: bool = True
    tie_break_rule: TieBreakRule = TieBreakRule.LOWEST_ID
    invalid_action_behavior: str = "noop"

    @model_validator(mode="before")
    @classmethod
    def align_num_players_with_explicit_roles(cls, data):
        if isinstance(data, dict) and "roles" in data and "num_players" not in data:
            roles = data["roles"]
            if isinstance(roles, (list, tuple)):
                return {**data, "num_players": len(roles)}
        return data

    @model_validator(mode="after")
    def validate_roles(self) -> "EnvironmentConfig":
        if self.num_players not in {5, 6}:
            raise ValueError("foundational engine supports 5- or 6-player games")
        if len(self.roles) != self.num_players:
            raise ValueError("roles must match num_players")
        counts = Counter(self.roles)
        if counts[Role.MAFIA] != 1:
            raise ValueError("roles must contain exactly one mafia")
        if counts[Role.DOCTOR] > 1:
            raise ValueError("roles may contain at most one doctor")
        if counts[Role.DETECTIVE] > 1:
            raise ValueError("roles may contain at most one detective")
        if counts[Role.VILLAGER] < 2:
            raise ValueError("roles must contain at least two villagers")
        if sum(counts.values()) != self.num_players:
            raise ValueError("roles must match num_players")
        if self.discussion_rounds < 0:
            raise ValueError("discussion_rounds must be non-negative")
        if self.max_days < 1:
            raise ValueError("max_days must be positive")
        if self.invalid_action_behavior != "noop":
            raise ValueError("foundational implementation supports only noop invalid handling")
        return self


def noop_action(actor: int) -> Action:
    return Action(actor=actor, action_type=ActionType.NOOP)


def role_alignment(role: Role) -> Alignment:
    return Alignment.MAFIA if role == Role.MAFIA else Alignment.TOWN
