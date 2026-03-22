from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from contracts import ActionType, DiscussionIntent, FrozenModel, Phase, Role, WinCondition, role_alignment

if TYPE_CHECKING:
    from game.state import GameState
    from train.trajectory import RolloutStep

# Phase B: small additive shaping; keep well below |terminal| = 1.0
DEFAULT_SURVIVAL_PER_DAY = 0.04
DEFAULT_SURVIVAL_CAP = 0.2
DEFAULT_VOTE_CORRECT_BONUS = 0.03
DEFAULT_VOTE_INCORRECT_PENALTY = -0.03
DEFAULT_VOTE_BONUS_CAP = 0.12
DEFAULT_MAFIA_KILL_BONUS = 0.05
DEFAULT_MAFIA_KILL_CAP = 0.2
DEFAULT_DETECTIVE_HIT_BONUS = 0.06
DEFAULT_DETECTIVE_CONFIRM_BONUS = 0.03
DEFAULT_DOCTOR_SAVE_BONUS = 0.05
DEFAULT_DOCTOR_SAVE_CAP = 0.15
DEFAULT_DECEPTION_BONUS = 0.05


class DeceptionRewardConfig(FrozenModel):
    enabled: bool = True
    false_detective_claim_bonus: float = DEFAULT_DECEPTION_BONUS


DEFAULT_DECEPTION_CONFIG = DeceptionRewardConfig()


def compute_terminal_reward(*, trainable_role: Role, winner: WinCondition | None) -> tuple[float | None, dict[str, float]]:
    if winner is None:
        return None, {}

    won = role_alignment(trainable_role).value == winner.value
    reward = 1.0 if won else -1.0
    return reward, {
        "terminal_win_loss": reward,
        "total_reward": reward,
    }


def count_trainable_survival_days(*, final_day: int, elimination_day: int | None) -> int:
    """Days the trainable seat completed before elimination (or full game length if never eliminated)."""
    if elimination_day is None:
        return max(0, final_day)
    return max(0, elimination_day - 1)


def compute_survival_bonus(
    survival_days: int,
    *,
    per_day: float = DEFAULT_SURVIVAL_PER_DAY,
    cap: float = DEFAULT_SURVIVAL_CAP,
) -> float:
    return min(float(survival_days) * per_day, cap)


def compute_outcome_reward(
    *,
    trainable_role: Role,
    winner: WinCondition | None,
    final_day: int,
    elimination_day: int | None,
    trainable_seat: int | None = None,
    final_state: GameState | None = None,
    steps: Sequence[RolloutStep] = (),
    survival_per_day: float = DEFAULT_SURVIVAL_PER_DAY,
    survival_cap: float = DEFAULT_SURVIVAL_CAP,
    vote_correct_bonus: float = DEFAULT_VOTE_CORRECT_BONUS,
    vote_incorrect_penalty: float = DEFAULT_VOTE_INCORRECT_PENALTY,
    vote_bonus_cap: float = DEFAULT_VOTE_BONUS_CAP,
    mafia_kill_bonus: float = DEFAULT_MAFIA_KILL_BONUS,
    mafia_kill_cap: float = DEFAULT_MAFIA_KILL_CAP,
    detective_hit_bonus: float = DEFAULT_DETECTIVE_HIT_BONUS,
    detective_confirm_bonus: float = DEFAULT_DETECTIVE_CONFIRM_BONUS,
    doctor_save_bonus: float = DEFAULT_DOCTOR_SAVE_BONUS,
    doctor_save_cap: float = DEFAULT_DOCTOR_SAVE_CAP,
    deception_config: DeceptionRewardConfig = DEFAULT_DECEPTION_CONFIG,
) -> tuple[float | None, dict[str, float]]:
    """Terminal ±1 plus capped survival and role-specific shaping for the trainable seat."""
    terminal, _ = compute_terminal_reward(trainable_role=trainable_role, winner=winner)
    if terminal is None:
        return None, {}

    survival_days = count_trainable_survival_days(
        final_day=final_day,
        elimination_day=elimination_day,
    )
    survival_bonus = compute_survival_bonus(
        survival_days,
        per_day=survival_per_day,
        cap=survival_cap,
    )
    breakdown = {
        "terminal_win_loss": terminal,
        "survival_bonus": survival_bonus,
        "survival_day_count": float(survival_days),
    }
    total = terminal + survival_bonus
    if final_state is not None and trainable_seat is not None:
        role_breakdown = compute_role_specific_reward(
            trainable_seat=trainable_seat,
            trainable_role=trainable_role,
            final_state=final_state,
            steps=steps,
            elimination_day=elimination_day,
            vote_correct_bonus=vote_correct_bonus,
            vote_incorrect_penalty=vote_incorrect_penalty,
            vote_bonus_cap=vote_bonus_cap,
            mafia_kill_bonus=mafia_kill_bonus,
            mafia_kill_cap=mafia_kill_cap,
            detective_hit_bonus=detective_hit_bonus,
            detective_confirm_bonus=detective_confirm_bonus,
            doctor_save_bonus=doctor_save_bonus,
            doctor_save_cap=doctor_save_cap,
            deception_config=deception_config,
        )
        breakdown.update(role_breakdown)
        total += (
            role_breakdown["vote_accuracy_bonus"]
            + role_breakdown["mafia_kill_bonus"]
            + role_breakdown["detective_hit_bonus"]
            + role_breakdown["doctor_save_bonus"]
            + role_breakdown["deception_bonus"]
        )
    breakdown["total_reward"] = total
    return total, breakdown


def compute_role_specific_reward(
    *,
    trainable_seat: int,
    trainable_role: Role,
    final_state: GameState,
    steps: Sequence[RolloutStep],
    elimination_day: int | None,
    vote_correct_bonus: float,
    vote_incorrect_penalty: float,
    vote_bonus_cap: float,
    mafia_kill_bonus: float,
    mafia_kill_cap: float,
    detective_hit_bonus: float,
    detective_confirm_bonus: float,
    doctor_save_bonus: float,
    doctor_save_cap: float,
    deception_config: DeceptionRewardConfig,
) -> dict[str, float]:
    trainable_won = role_alignment(trainable_role).value == final_state.winner.value
    vote_correct_count = 0
    vote_incorrect_count = 0
    mafia_kill_count = 0
    detective_hit_count = 0
    detective_confirm_count = 0
    doctor_save_count = 0
    deception_count = 0
    deception_detected_count = 0

    if trainable_won and role_alignment(trainable_role).value == WinCondition.TOWN.value:
        vote_records = [record for record in final_state.vote_history if record.voter == trainable_seat]
        vote_correct_count = sum(1 for record in vote_records if final_state.roles[record.target] == Role.MAFIA)
        vote_incorrect_count = len(vote_records) - vote_correct_count

    night_kill_eliminations = {
        record.day: record.player
        for record in final_state.elimination_history
        if record.reason == "night_kill"
    }
    trainable_steps = [step for step in steps if step.actor == trainable_seat and step.normalized_action is not None]

    if trainable_won and trainable_role == Role.MAFIA:
        mafia_kill_count = sum(
            1
            for step in trainable_steps
            if step.phase == Phase.NIGHT_MAFIA
            and step.normalized_action.action_type == ActionType.NIGHT_KILL
            and step.normalized_action.target is not None
            and night_kill_eliminations.get(step.day) == step.normalized_action.target
        )

    if trainable_won and trainable_role == Role.DETECTIVE:
        first_hit_day = min(
            (
                step.day
                for step in trainable_steps
                if step.phase == Phase.NIGHT_DETECTIVE
                and step.normalized_action.action_type == ActionType.INVESTIGATE
                and step.normalized_action.target is not None
                and final_state.roles[step.normalized_action.target] == Role.MAFIA
            ),
            default=None,
        )
        detective_hit_count = 1 if first_hit_day is not None else 0
        mafia_elimination_day = next(
            (
                record.day
                for record in final_state.elimination_history
                if record.player < len(final_state.roles) and final_state.roles[record.player] == Role.MAFIA
            ),
            None,
        )
        if first_hit_day is not None and mafia_elimination_day is not None and mafia_elimination_day >= first_hit_day:
            detective_confirm_count = 1

    if trainable_won and trainable_role == Role.DOCTOR:
        mafia_targets_by_day = {
            step.day: step.normalized_action.target
            for step in steps
            if step.phase == Phase.NIGHT_MAFIA
            and step.normalized_action is not None
            and step.normalized_action.action_type == ActionType.NIGHT_KILL
        }
        doctor_save_count = sum(
            1
            for step in trainable_steps
            if step.phase == Phase.NIGHT_DOCTOR
            and step.normalized_action.action_type == ActionType.PROTECT
            and step.normalized_action.target is not None
            and mafia_targets_by_day.get(step.day) == step.normalized_action.target
        )

    if trainable_won and trainable_role == Role.MAFIA and deception_config.enabled:
        deception_count, deception_detected_count = _count_false_detective_claim_bonus_events(
            trainable_seat=trainable_seat,
            final_state=final_state,
            steps=steps,
            elimination_day=elimination_day,
        )

    vote_accuracy = _clamp(
        vote_correct_count * vote_correct_bonus + vote_incorrect_count * vote_incorrect_penalty,
        lower=-vote_bonus_cap,
        upper=vote_bonus_cap,
    )
    mafia_kill_value = min(mafia_kill_count * mafia_kill_bonus, mafia_kill_cap)
    detective_value = detective_hit_count * detective_hit_bonus + detective_confirm_count * detective_confirm_bonus
    doctor_save_value = min(doctor_save_count * doctor_save_bonus, doctor_save_cap)
    deception_value = deception_count * deception_config.false_detective_claim_bonus
    return {
        "vote_accuracy_bonus": vote_accuracy,
        "vote_correct_count": float(vote_correct_count),
        "vote_incorrect_count": float(vote_incorrect_count),
        "mafia_kill_bonus": mafia_kill_value,
        "mafia_successful_kill_count": float(mafia_kill_count),
        "detective_hit_bonus": detective_value,
        "detective_hit_count": float(detective_hit_count),
        "detective_confirmed_elimination_count": float(detective_confirm_count),
        "doctor_save_bonus": doctor_save_value,
        "doctor_save_count": float(doctor_save_count),
        "deception_bonus": deception_value,
        "deception_detected_event_count": float(deception_detected_count),
        "deception_claim_detective_win_count": float(deception_count),
        "invalid_action_penalty": 0.0,
    }


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _count_false_detective_claim_bonus_events(
    *,
    trainable_seat: int,
    final_state: GameState,
    steps: Sequence[RolloutStep],
    elimination_day: int | None,
) -> tuple[int, int]:
    if final_state.winner != WinCondition.MAFIA:
        return 0, 0

    valid_claim_days = {
        step.day
        for step in steps
        if step.actor == trainable_seat
        and step.phase == Phase.DAY_DISCUSSION
        and step.is_valid is True
        and step.normalized_action is not None
        and step.normalized_action.action_type == ActionType.SPEAK
        and step.normalized_action.intent == DiscussionIntent.CLAIM_DETECTIVE
    }
    if not valid_claim_days:
        return 0, 0

    transcript_claim_days = {
        event.day
        for event in final_state.transcript
        if event.phase == Phase.DAY_DISCUSSION
        and event.speaker == trainable_seat
        and event.action_type == ActionType.SPEAK
        and event.intent == DiscussionIntent.CLAIM_DETECTIVE
    }
    detected_claim_days = valid_claim_days & transcript_claim_days
    if not detected_claim_days:
        return 0, 0

    first_claim_day = min(detected_claim_days)
    if elimination_day is not None and elimination_day <= first_claim_day:
        return len(detected_claim_days), 0
    return len(detected_claim_days), 1
