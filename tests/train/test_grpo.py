from contracts import Alignment, Phase, Role, WinCondition
from game import new_game
from train import build_grouped_episode_batch, normalize_group_rewards
from train.trajectory import EpisodeMetadata, EpisodeRollout, OutcomeRecord


def test_normalize_group_rewards_orders_episodes_by_reward() -> None:
    scores = normalize_group_rewards((1.0, 0.0, -1.0))

    assert scores[0] is not None and scores[1] is not None and scores[2] is not None
    assert scores[0] > scores[1] > scores[2]


def test_build_grouped_episode_batch_records_best_and_worst_without_dropping_episodes() -> None:
    first = _episode("ep-1", 1.2, seed=1)
    second = _episode("ep-2", 0.2, seed=2)
    third = _episode("ep-3", -0.8, seed=3)

    grouped = build_grouped_episode_batch((first, second, third), group_id="group-a")

    assert grouped.batch.group_id == "group-a"
    assert grouped.batch.episode_ids == ("ep-1", "ep-2", "ep-3")
    assert grouped.batch.best_episode_id == "ep-1"
    assert grouped.batch.worst_episode_id == "ep-3"
    assert len(grouped.episodes) == 3
    for episode, score in zip(grouped.episodes, grouped.batch.normalized_group_scores, strict=True):
        assert episode.metadata.group_id == "group-a"
        for step in episode.steps:
            if step.is_trainable_actor:
                assert step.group_normalized_score == score


def test_build_grouped_episode_batch_is_reproducible_from_saved_inputs() -> None:
    episodes = (
        _episode("ep-1", 0.5, seed=10),
        _episode("ep-2", 1.5, seed=11),
    )

    first = build_grouped_episode_batch(episodes)
    second = build_grouped_episode_batch(episodes)

    assert first == second


def _episode(episode_id: str, reward: float, *, seed: int) -> EpisodeRollout:
    state = new_game(seed=seed).model_copy(update={"phase": Phase.TERMINAL, "winner": WinCondition.MAFIA})
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        seed=seed,
        checkpoint_id="ckpt-1",
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        trainable_alignment=Alignment.MAFIA,
        opponent_pool_id="scripted_v1",
        environment_config_hash="env-1",
    )
    outcome = OutcomeRecord(
        winner=WinCondition.MAFIA,
        trainable_survived=True,
        num_days_reached=1,
        final_reward=reward,
        reward_breakdown={"total_reward": reward},
    )
    return EpisodeRollout(
        metadata=metadata,
        initial_state=state,
        final_state=state,
        steps=(),
        outcome=outcome,
    )
