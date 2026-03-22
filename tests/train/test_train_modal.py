import importlib.util
from pathlib import Path

import pytest


def _load_train_modal_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_modal.py"
    spec = importlib.util.spec_from_file_location("train_modal_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Modal script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_MODAL = _load_train_modal_module()


def _write_config(tmp_path: Path, checkpoint_location: str = "checkpoints/modal") -> Path:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: modal-test",
                "model:",
                "  trainable_model_name: hf://qwen3-8b",
                "modal:",
                f"  checkpoint_location: {checkpoint_location}",
            ]
        )
    )
    return config_path


def test_reported_truthfulqa_scores_path_uses_configured_log_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: modal-test",
                "model:",
                "  trainable_model_name: hf://qwen3-8b",
                "logging:",
                "  log_dir_name: metrics",
                "modal:",
                "  checkpoint_location: checkpoints/modal",
            ]
        )
    )

    config = TRAIN_MODAL.load_train_config(str(config_path))
    checkpoint_root = TRAIN_MODAL._resolve_modal_checkpoint_root(
        checkpoint_root=None,
        config_path=str(config_path),
    )

    truthfulqa_scores_path = (
        Path(checkpoint_root) / config.logging.log_dir_name / "truthfulqa_scores.jsonl"
    )

    assert str(truthfulqa_scores_path) == (
        f"{TRAIN_MODAL.DEFAULT_MODAL_VOLUME_MOUNT_PATH}/checkpoints/modal/metrics/truthfulqa_scores.jsonl"
    )


def test_resolve_modal_checkpoint_root_uses_configured_relative_path(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, checkpoint_location="checkpoints/modal")

    resolved = TRAIN_MODAL._resolve_modal_checkpoint_root(
        checkpoint_root=None,
        config_path=str(config_path),
    )

    assert resolved == f"{TRAIN_MODAL.DEFAULT_MODAL_VOLUME_MOUNT_PATH}/checkpoints/modal"


def test_resolve_modal_checkpoint_root_accepts_absolute_path_inside_volume(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    persisted_path = f"{TRAIN_MODAL.DEFAULT_MODAL_VOLUME_MOUNT_PATH}/runs/exp-1"

    resolved = TRAIN_MODAL._resolve_modal_checkpoint_root(
        checkpoint_root=persisted_path,
        config_path=str(config_path),
    )

    assert resolved == persisted_path


def test_resolve_modal_checkpoint_root_rejects_absolute_path_outside_volume(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="must live under"):
        TRAIN_MODAL._resolve_modal_checkpoint_root(
            checkpoint_root="/root/checkpoints/modal",
            config_path=str(config_path),
        )


def test_resolve_modal_checkpoint_root_rejects_relative_path_outside_volume(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="must live under"):
        TRAIN_MODAL._resolve_modal_checkpoint_root(
            checkpoint_root="../checkpoints/modal",
            config_path=str(config_path),
        )
