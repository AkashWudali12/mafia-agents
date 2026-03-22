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


def test_resolve_repo_root_supports_local_scripts_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "train.yaml").write_text("project:\n  experiment_name: test\nmodel:\n  trainable_model_name: hf://qwen3-8b\n")

    resolved = TRAIN_MODAL._resolve_repo_root(scripts_dir / "train_modal.py")

    assert resolved == repo_root


def test_resolve_repo_root_supports_modal_container_layout(tmp_path: Path) -> None:
    container_root = tmp_path / "root"
    container_root.mkdir()
    (container_root / "src").mkdir()
    (container_root / "train.yaml").write_text("project:\n  experiment_name: test\nmodel:\n  trainable_model_name: hf://qwen3-8b\n")

    resolved = TRAIN_MODAL._resolve_repo_root(container_root / "train_modal.py")

    assert resolved == container_root


def _write_config(
    tmp_path: Path,
    checkpoint_location: str = "checkpoints/modal",
    *,
    gpu_type: str = "none",
    cpu_count: int = 2,
    memory_gb: int = 4,
    timeout_seconds: int = 3600,
) -> Path:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: modal-test",
                "model:",
                "  trainable_model_name: hf://qwen3-8b",
                "modal:",
                f"  gpu_type: {gpu_type}",
                f"  cpu_count: {cpu_count}",
                f"  memory_gb: {memory_gb}",
                f"  timeout_seconds: {timeout_seconds}",
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


def test_modal_resource_kwargs_include_gpu_request_when_configured(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        gpu_type="A10G",
        cpu_count=8,
        memory_gb=32,
        timeout_seconds=21600,
    )

    resource_kwargs = TRAIN_MODAL._modal_resource_kwargs(config_path)

    assert resource_kwargs == {
        "gpu": "A10G",
        "cpu": 8.0,
        "memory": 32768,
        "timeout": 21600,
    }


def test_modal_resource_kwargs_omit_gpu_when_disabled(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, gpu_type="none", cpu_count=4, memory_gb=8, timeout_seconds=7200)

    resource_kwargs = TRAIN_MODAL._modal_resource_kwargs(config_path)

    assert resource_kwargs == {
        "cpu": 4.0,
        "memory": 8192,
        "timeout": 7200,
    }


def test_collect_forwarded_env_includes_explicit_and_prefixed_keys() -> None:
    forwarded = TRAIN_MODAL._collect_forwarded_env(
        {
            "HF_TOKEN": "hf-secret",
            "OPENROUTER_API_KEY": "or-secret",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "UNRELATED_VAR": "ignore-me",
        }
    )

    assert forwarded == {
        "HF_TOKEN": "hf-secret",
        "OPENROUTER_API_KEY": "or-secret",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    }


def test_collect_forwarded_env_skips_empty_values() -> None:
    forwarded = TRAIN_MODAL._collect_forwarded_env(
        {
            "HF_TOKEN": "",
            "PYTORCH_ALLOC_CONF": "",
            "CUDA_LAUNCH_BLOCKING": "",
            "OPENROUTER_API_KEY": "or-secret",
        }
    )

    assert forwarded == {
        "OPENROUTER_API_KEY": "or-secret",
    }


def test_modal_secrets_uses_configured_openrouter_secret_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: modal-test",
                "model:",
                "  trainable_model_name: hf://qwen3-8b",
                "modal:",
                "  openrouter_secret_name: mafia-openrouter-secret",
            ]
        )
    )

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> str:
            return f"secret:{name}"

    class FakeModal:
        Secret = FakeSecret

    monkeypatch.setattr(TRAIN_MODAL, "modal", FakeModal)

    secrets = TRAIN_MODAL._modal_secrets(config_path)

    assert secrets == ["secret:mafia-openrouter-secret"]


def test_modal_secrets_returns_empty_when_secret_is_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: modal-test",
                "model:",
                "  trainable_model_name: hf://qwen3-8b",
                "modal:",
                "  openrouter_secret_name: null",
            ]
        )
    )

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> str:
            return f"secret:{name}"

    class FakeModal:
        Secret = FakeSecret

    monkeypatch.setattr(TRAIN_MODAL, "modal", FakeModal)

    secrets = TRAIN_MODAL._modal_secrets(config_path)

    assert secrets == []


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
