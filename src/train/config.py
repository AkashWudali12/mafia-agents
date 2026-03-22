from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from contracts import EnvironmentConfig
from train.hf_policy import DEFAULT_TRAINABLE_MODEL
from train.rewards import DeceptionRewardConfig


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class ProjectConfig(FrozenModel):
    experiment_name: str
    seed: int = 0
    run_notes: str = ""


class ModelConfig(FrozenModel):
    trainable_model_name: str = DEFAULT_TRAINABLE_MODEL
    tokenizer_name: str | None = None
    provider: str = "huggingface"
    cache_dir: str = ".cache/huggingface"
    max_context_length: int = 4096
    output_format_mode: str = "typed_json"
    device: str | None = None
    max_new_tokens: int = 96
    temperature: float = 0.0


class OpponentConfig(FrozenModel):
    opponent_provider: str = "openrouter"
    opponent_pool_id: str = "openrouter_pool_v1"
    model_names: tuple[str, ...] = ()
    sampling_strategy: str = "random_per_seat"
    cache_behavior: str = "disabled"
    prompt_version: str = "v1"


class TrainingConfig(FrozenModel):
    algorithm_name: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 1
    episodes_per_batch: int = 4
    grpo_group_size: int = 4
    gradient_accumulation: int = 1
    total_updates: int = 1
    checkpoint_interval: int = 1
    evaluation_interval: int = 1


class RewardConfig(FrozenModel):
    terminal_win_value: float = 1.0
    terminal_loss_value: float = -1.0
    survival_bonus_per_day: float = 0.04
    survival_bonus_cap: float = 0.2
    vote_correct_bonus: float = 0.03
    vote_incorrect_penalty: float = -0.03
    vote_bonus_cap: float = 0.12
    mafia_kill_bonus: float = 0.05
    mafia_kill_cap: float = 0.2
    detective_hit_bonus: float = 0.06
    detective_confirm_bonus: float = 0.03
    doctor_save_bonus: float = 0.05
    doctor_save_cap: float = 0.15
    deception: DeceptionRewardConfig = Field(default_factory=DeceptionRewardConfig)
    invalid_action_penalty: float = 0.0


class LoggingConfig(FrozenModel):
    backend: str = "jsonl"
    level: str = "DEBUG"
    log_dir_name: str = "logs"
    transcript_retention: str = "summary"
    observation_snapshot_retention: str = "trainable_only"
    action_trace_retention: str = "full"


class ModalConfig(FrozenModel):
    app_name: str = "mafia-train"
    gpu_type: str = "none"
    cpu_count: int = 2
    memory_gb: int = 4
    timeout_seconds: int = 3600
    checkpoint_location: str = "checkpoints"


class EvaluationConfig(FrozenModel):
    mafia_eval_seeds: tuple[int, ...] = (101, 202, 303, 404)
    episodes_per_role: int = 1
    truthfulqa_subset: str = "multiple_choice"
    benchmark_interval: int = 1


class TrainConfig(FrozenModel):
    project: ProjectConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    model: ModelConfig
    opponents: OpponentConfig = Field(default_factory=OpponentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_train_config(path: str | Path) -> TrainConfig:
    source_path = Path(path)
    data = _load_yaml_like_data(source_path)
    return TrainConfig.model_validate(data)


def _load_yaml_like_data(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only when yaml is missing
        raise ImportError("PyYAML is required to load train.yaml configuration files") from exc
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("train config must deserialize to a mapping")
    return loaded
