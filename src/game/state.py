from __future__ import annotations

from pydantic import Field

from contracts import (
    EliminationRecord,
    EnvironmentConfig,
    FrozenModel,
    InvestigationResult,
    NightOutcome,
    Phase,
    Role,
    TranscriptEvent,
    ValidationLogEntry,
    VoteRecord,
    WinCondition,
)


class PendingNightActions(FrozenModel):
    mafia_target: int | None = None
    doctor_target: int | None = None
    detective_target: int | None = None


class GameState(FrozenModel):
    """Immutable engine state shared across engine, policy, and training layers.

    Invariants:
    - `roles` and `alive` always align with `config.num_players`.
    - Day progression always follows PRD order: announcement -> discussion/voting -> resolution.
    - History fields are append-only so downstream consumers can rely on stable audit trails.
    """

    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    seed: int | None = None
    day: int = 1
    phase: Phase = Phase.NIGHT_MAFIA
    roles: tuple[Role, ...]
    alive: tuple[bool, ...]
    pending_night_actions: PendingNightActions = Field(default_factory=PendingNightActions)
    transcript: tuple[TranscriptEvent, ...] = ()
    vote_history: tuple[VoteRecord, ...] = ()
    elimination_history: tuple[EliminationRecord, ...] = ()
    investigation_history: tuple[InvestigationResult, ...] = ()
    validation_log: tuple[ValidationLogEntry, ...] = ()
    last_night_outcome: NightOutcome | None = None
    doctor_last_target: int | None = None
    discussion_turn_index: int = 0
    current_votes: tuple[VoteRecord, ...] = ()
    current_voters: tuple[int, ...] = ()
    winner: WinCondition | None = None

    @property
    def living_players(self) -> tuple[int, ...]:
        return tuple(index for index, is_alive in enumerate(self.alive) if is_alive)

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.TERMINAL or self.winner is not None
