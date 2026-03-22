import json

from contracts import Alignment, Phase, Role, WinCondition
from game import new_game
from train import CheckpointState, build_grouped_episode_batch, load_checkpoint, save_checkpoint
from train.trajectory import EpisodeMetadata, EpisodeRollout, OutcomeRecord


def test_checkpoint_save_and_load_roundtrip(tmp_path) -> None:
    episode = _episode("ep-1", 1.0)
    grouped = build_grouped_episode_batch((episode,), group_id="group-1")
    checkpoint = CheckpointState(
        checkpoint_id="ckpt-00001",
        update_index=1,
        step_counters={"updates": 1, "episodes": 1},
        model_reference="hf://qwen3-8b",
        reward_config_snapshot={"survival_bonus_per_day": 0.04},
        train_config_snapshot={"training": {"grpo_group_size": 1}},
        metrics={"mean_reward": 1.0},
    )

    checkpoint_dir = save_checkpoint(
        tmp_path,
        state=checkpoint,
        grouped_batch=grouped,
        episodes=grouped.episodes,
    )

    loaded = load_checkpoint(checkpoint_dir)

    assert loaded == checkpoint
    artifacts = json.loads((checkpoint_dir / "sample_artifacts.json").read_text())
    assert artifacts["best_episodes"]
    assert artifacts["worst_episodes"]


def _episode(episode_id: str, reward: float) -> EpisodeRollout:
    state = new_game(seed=5).model_copy(update={"phase": Phase.TERMINAL, "winner": WinCondition.MAFIA})
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        seed=5,
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
