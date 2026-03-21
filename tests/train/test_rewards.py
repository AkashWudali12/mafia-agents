from contracts import (
    Action,
    ActionType,
    DiscussionIntent,
    EliminationRecord,
    Phase,
    Role,
    TranscriptEvent,
    VoteRecord,
    WinCondition,
)
from game import new_game
from train import DeceptionRewardConfig, compute_outcome_reward, compute_terminal_reward
from train.rewards import compute_survival_bonus, count_trainable_survival_days
from train.trajectory import RolloutStep


def test_compute_terminal_reward_rewards_matching_faction_win() -> None:
    reward, breakdown = compute_terminal_reward(
        trainable_role=Role.MAFIA,
        winner=WinCondition.MAFIA,
    )

    assert reward == 1.0
    assert breakdown == {
        "terminal_win_loss": 1.0,
        "total_reward": 1.0,
    }


def test_compute_terminal_reward_penalizes_opposing_faction_loss() -> None:
    reward, breakdown = compute_terminal_reward(
        trainable_role=Role.DOCTOR,
        winner=WinCondition.MAFIA,
    )

    assert reward == -1.0
    assert breakdown == {
        "terminal_win_loss": -1.0,
        "total_reward": -1.0,
    }


def test_count_trainable_survival_days_eliminated_vs_survived() -> None:
    assert count_trainable_survival_days(final_day=5, elimination_day=None) == 5
    assert count_trainable_survival_days(final_day=5, elimination_day=3) == 2
    assert count_trainable_survival_days(final_day=5, elimination_day=1) == 0


def test_compute_survival_bonus_respects_cap() -> None:
    assert compute_survival_bonus(2, per_day=0.04, cap=0.2) == 0.08
    assert compute_survival_bonus(100, per_day=0.04, cap=0.2) == 0.2


def test_compute_outcome_reward_adds_capped_survival() -> None:
    total, bd = compute_outcome_reward(
        trainable_role=Role.VILLAGER,
        winner=WinCondition.TOWN,
        final_day=10,
        elimination_day=None,
        survival_per_day=0.04,
        survival_cap=0.2,
    )

    assert bd["terminal_win_loss"] == 1.0
    assert bd["survival_bonus"] == 0.2  # 10 * 0.04 capped at 0.2
    assert total == 1.2
    assert bd["total_reward"] == total
    assert bd["survival_day_count"] == 10.0


def test_mafia_kill_bonus_fires_only_for_mafia() -> None:
    final_state = new_game().model_copy(
        update={
            "winner": WinCondition.MAFIA,
            "phase": Phase.TERMINAL,
            "day": 1,
            "alive": (True, True, True, False, True),
            "elimination_history": (
                EliminationRecord(day=1, player=3, role=Role.VILLAGER, reason="night_kill"),
            ),
        }
    )
    steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_MAFIA,
            actor=0,
            action=Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3),
        ),
    )

    mafia_total, mafia_breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.MAFIA,
        final_day=1,
        elimination_day=None,
        final_state=final_state,
        steps=steps,
    )
    town_total, town_breakdown = compute_outcome_reward(
        trainable_seat=1,
        trainable_role=Role.DOCTOR,
        winner=WinCondition.MAFIA,
        final_day=1,
        elimination_day=None,
        final_state=final_state,
        steps=steps,
    )

    assert mafia_breakdown["mafia_successful_kill_count"] == 1.0
    assert mafia_breakdown["mafia_kill_bonus"] > 0.0
    assert mafia_total == mafia_breakdown["total_reward"]
    assert town_breakdown["mafia_successful_kill_count"] == 0.0
    assert town_breakdown["mafia_kill_bonus"] == 0.0
    assert town_total == town_breakdown["total_reward"]


def test_role_specific_bonuses_are_zero_when_trainable_agent_loses() -> None:
    losing_mafia_state = new_game().model_copy(
        update={
            "winner": WinCondition.TOWN,
            "phase": Phase.TERMINAL,
            "day": 2,
            "alive": (False, True, True, True, True),
            "elimination_history": (
                EliminationRecord(day=1, player=3, role=Role.VILLAGER, reason="night_kill"),
                EliminationRecord(day=2, player=0, role=Role.MAFIA, reason="vote"),
            ),
            "transcript": (
                TranscriptEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=0,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.CLAIM_DETECTIVE,
                    target=2,
                    message="I checked 2",
                ),
            ),
        }
    )
    mafia_steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_MAFIA,
            actor=0,
            action=Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3),
        ),
        _step(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            actor=0,
            action=Action(
                actor=0,
                action_type=ActionType.SPEAK,
                intent=DiscussionIntent.CLAIM_DETECTIVE,
                target=2,
                message="I checked 2",
            ),
        ),
    )
    _, mafia_breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.TOWN,
        final_day=2,
        elimination_day=2,
        final_state=losing_mafia_state,
        steps=mafia_steps,
    )

    assert mafia_breakdown["mafia_successful_kill_count"] == 0.0
    assert mafia_breakdown["mafia_kill_bonus"] == 0.0
    assert mafia_breakdown["deception_detected_event_count"] == 0.0
    assert mafia_breakdown["deception_claim_detective_win_count"] == 0.0
    assert mafia_breakdown["deception_bonus"] == 0.0

    losing_town_state = new_game().model_copy(
        update={
            "winner": WinCondition.MAFIA,
            "phase": Phase.TERMINAL,
            "day": 2,
            "vote_history": (
                VoteRecord(day=1, voter=1, target=0),
            ),
            "elimination_history": (
                EliminationRecord(day=1, player=3, role=Role.VILLAGER, reason="night_kill"),
            ),
        }
    )
    town_steps = (
        _step(day=1, phase=Phase.DAY_VOTING, actor=1, action=Action(actor=1, action_type=ActionType.VOTE, target=0)),
        _step(
            day=1,
            phase=Phase.NIGHT_DOCTOR,
            actor=1,
            action=Action(actor=1, action_type=ActionType.PROTECT, target=3),
        ),
        _step(
            day=1,
            phase=Phase.NIGHT_DETECTIVE,
            actor=2,
            action=Action(actor=2, action_type=ActionType.INVESTIGATE, target=0),
        ),
    )
    _, town_breakdown = compute_outcome_reward(
        trainable_seat=1,
        trainable_role=Role.DOCTOR,
        winner=WinCondition.MAFIA,
        final_day=2,
        elimination_day=None,
        final_state=losing_town_state,
        steps=town_steps,
    )

    assert town_breakdown["vote_correct_count"] == 0.0
    assert town_breakdown["vote_incorrect_count"] == 0.0
    assert town_breakdown["vote_accuracy_bonus"] == 0.0
    assert town_breakdown["doctor_save_count"] == 0.0
    assert town_breakdown["doctor_save_bonus"] == 0.0


def test_doctor_save_bonus_fires_only_on_real_prevented_kills() -> None:
    saved_state = new_game().model_copy(
        update={
            "winner": WinCondition.TOWN,
            "phase": Phase.TERMINAL,
            "day": 1,
        }
    )
    unsaved_state = new_game().model_copy(
        update={
            "winner": WinCondition.TOWN,
            "phase": Phase.TERMINAL,
            "day": 1,
            "alive": (True, True, True, False, True),
            "elimination_history": (
                EliminationRecord(day=1, player=3, role=Role.VILLAGER, reason="night_kill"),
            ),
        }
    )
    saved_steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_MAFIA,
            actor=0,
            action=Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3),
        ),
        _step(
            day=1,
            phase=Phase.NIGHT_DOCTOR,
            actor=1,
            action=Action(actor=1, action_type=ActionType.PROTECT, target=3),
        ),
    )
    unsaved_steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_MAFIA,
            actor=0,
            action=Action(actor=0, action_type=ActionType.NIGHT_KILL, target=3),
        ),
        _step(
            day=1,
            phase=Phase.NIGHT_DOCTOR,
            actor=1,
            action=Action(actor=1, action_type=ActionType.PROTECT, target=4),
        ),
    )

    _, saved_breakdown = compute_outcome_reward(
        trainable_seat=1,
        trainable_role=Role.DOCTOR,
        winner=WinCondition.TOWN,
        final_day=1,
        elimination_day=None,
        final_state=saved_state,
        steps=saved_steps,
    )
    _, unsaved_breakdown = compute_outcome_reward(
        trainable_seat=1,
        trainable_role=Role.DOCTOR,
        winner=WinCondition.TOWN,
        final_day=1,
        elimination_day=None,
        final_state=unsaved_state,
        steps=unsaved_steps,
    )

    assert saved_breakdown["doctor_save_count"] == 1.0
    assert saved_breakdown["doctor_save_bonus"] > 0.0
    assert unsaved_breakdown["doctor_save_count"] == 0.0
    assert unsaved_breakdown["doctor_save_bonus"] == 0.0


def test_detective_hit_and_confirm_bonus_require_true_mafia_hit() -> None:
    final_state = new_game().model_copy(
        update={
            "winner": WinCondition.TOWN,
            "phase": Phase.TERMINAL,
            "day": 1,
            "alive": (False, True, True, True, True),
            "elimination_history": (
                EliminationRecord(day=1, player=0, role=Role.MAFIA, reason="vote"),
            ),
        }
    )
    hit_steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_DETECTIVE,
            actor=2,
            action=Action(actor=2, action_type=ActionType.INVESTIGATE, target=0),
        ),
    )
    miss_steps = (
        _step(
            day=1,
            phase=Phase.NIGHT_DETECTIVE,
            actor=2,
            action=Action(actor=2, action_type=ActionType.INVESTIGATE, target=3),
        ),
    )

    _, hit_breakdown = compute_outcome_reward(
        trainable_seat=2,
        trainable_role=Role.DETECTIVE,
        winner=WinCondition.TOWN,
        final_day=1,
        elimination_day=None,
        final_state=final_state,
        steps=hit_steps,
    )
    _, miss_breakdown = compute_outcome_reward(
        trainable_seat=2,
        trainable_role=Role.DETECTIVE,
        winner=WinCondition.TOWN,
        final_day=1,
        elimination_day=None,
        final_state=final_state,
        steps=miss_steps,
    )

    assert hit_breakdown["detective_hit_count"] == 1.0
    assert hit_breakdown["detective_confirmed_elimination_count"] == 1.0
    assert hit_breakdown["detective_hit_bonus"] > 0.0
    assert miss_breakdown["detective_hit_count"] == 0.0
    assert miss_breakdown["detective_confirmed_elimination_count"] == 0.0
    assert miss_breakdown["detective_hit_bonus"] == 0.0


def test_vote_accuracy_bonus_tracks_correct_and_incorrect_town_votes() -> None:
    final_state = new_game().model_copy(
        update={
            "winner": WinCondition.TOWN,
            "phase": Phase.TERMINAL,
            "day": 2,
            "vote_history": (
                VoteRecord(day=1, voter=1, target=0),
                VoteRecord(day=2, voter=1, target=3),
            ),
            "alive": (False, True, True, False, True),
            "elimination_history": (
                EliminationRecord(day=1, player=0, role=Role.MAFIA, reason="vote"),
                EliminationRecord(day=2, player=3, role=Role.VILLAGER, reason="vote"),
            ),
        }
    )
    steps = (
        _step(day=1, phase=Phase.DAY_VOTING, actor=1, action=Action(actor=1, action_type=ActionType.VOTE, target=0)),
        _step(day=2, phase=Phase.DAY_VOTING, actor=1, action=Action(actor=1, action_type=ActionType.VOTE, target=3)),
    )

    _, breakdown = compute_outcome_reward(
        trainable_seat=1,
        trainable_role=Role.DOCTOR,
        winner=WinCondition.TOWN,
        final_day=2,
        elimination_day=None,
        final_state=final_state,
        steps=steps,
    )

    assert breakdown["vote_correct_count"] == 1.0
    assert breakdown["vote_incorrect_count"] == 1.0
    assert breakdown["vote_accuracy_bonus"] == 0.0


def test_deception_bonus_fires_only_when_claim_is_confirmed_by_mafia_win() -> None:
    winning_state = new_game().model_copy(
        update={
            "winner": WinCondition.MAFIA,
            "phase": Phase.TERMINAL,
            "day": 2,
            "transcript": (
                TranscriptEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=0,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.CLAIM_DETECTIVE,
                    target=2,
                    message="I checked 2",
                ),
            ),
        }
    )
    losing_state = winning_state.model_copy(update={"winner": WinCondition.TOWN})
    valid_claim_steps = (
        _step(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            actor=0,
            action=Action(
                actor=0,
                action_type=ActionType.SPEAK,
                intent=DiscussionIntent.CLAIM_DETECTIVE,
                target=2,
                message="I checked 2",
            ),
        ),
    )

    _, winning_breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.MAFIA,
        final_day=2,
        elimination_day=None,
        final_state=winning_state,
        steps=valid_claim_steps,
    )
    _, losing_breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.TOWN,
        final_day=2,
        elimination_day=1,
        final_state=losing_state,
        steps=valid_claim_steps,
    )

    assert winning_breakdown["deception_detected_event_count"] == 1.0
    assert winning_breakdown["deception_claim_detective_win_count"] == 1.0
    assert winning_breakdown["deception_bonus"] > 0.0
    assert losing_breakdown["deception_detected_event_count"] == 0.0
    assert losing_breakdown["deception_claim_detective_win_count"] == 0.0
    assert losing_breakdown["deception_bonus"] == 0.0


def test_deception_bonus_requires_valid_trainable_discussion_turn() -> None:
    winning_state = new_game().model_copy(
        update={
            "winner": WinCondition.MAFIA,
            "phase": Phase.TERMINAL,
            "day": 2,
            "transcript": (
                TranscriptEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=0,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.CLAIM_DETECTIVE,
                    target=2,
                    message="I checked 2",
                ),
            ),
        }
    )
    invalid_steps = (
        RolloutStep(
            step_index=0,
            day=1,
            phase=Phase.DAY_DISCUSSION,
            actor=0,
            is_trainable_actor=True,
            submitted_action=Action(
                actor=0,
                action_type=ActionType.SPEAK,
                intent=DiscussionIntent.CLAIM_DETECTIVE,
                target=2,
                message="I checked 2",
            ),
            normalized_action=Action(actor=0, action_type=ActionType.NOOP),
            is_valid=False,
            next_day=1,
            next_phase=Phase.DAY_DISCUSSION,
        ),
    )

    _, breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.MAFIA,
        final_day=2,
        elimination_day=None,
        final_state=winning_state,
        steps=invalid_steps,
    )

    assert breakdown["deception_detected_event_count"] == 0.0
    assert breakdown["deception_claim_detective_win_count"] == 0.0
    assert breakdown["deception_bonus"] == 0.0


def test_deception_bonus_is_easy_to_disable_in_config() -> None:
    winning_state = new_game().model_copy(
        update={
            "winner": WinCondition.MAFIA,
            "phase": Phase.TERMINAL,
            "day": 2,
            "transcript": (
                TranscriptEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=0,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.CLAIM_DETECTIVE,
                    target=2,
                    message="I checked 2",
                ),
            ),
        }
    )
    steps = (
        _step(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            actor=0,
            action=Action(
                actor=0,
                action_type=ActionType.SPEAK,
                intent=DiscussionIntent.CLAIM_DETECTIVE,
                target=2,
                message="I checked 2",
            ),
        ),
    )

    _, breakdown = compute_outcome_reward(
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        winner=WinCondition.MAFIA,
        final_day=2,
        elimination_day=None,
        final_state=winning_state,
        steps=steps,
        deception_config=DeceptionRewardConfig(enabled=False),
    )

    assert breakdown["deception_detected_event_count"] == 0.0
    assert breakdown["deception_claim_detective_win_count"] == 0.0
    assert breakdown["deception_bonus"] == 0.0


def _step(*, day: int, phase: Phase, actor: int, action: Action) -> RolloutStep:
    return RolloutStep(
        step_index=0,
        day=day,
        phase=phase,
        actor=actor,
        is_trainable_actor=True,
        submitted_action=action,
        normalized_action=action,
        is_valid=True,
        next_day=day,
        next_phase=phase,
    )
