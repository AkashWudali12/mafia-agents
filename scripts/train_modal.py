from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_repo_root(script_path: str | Path) -> Path:
    resolved_script = Path(script_path).resolve()
    candidates = [resolved_script.parent, *resolved_script.parents]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "train.yaml").exists():
            return candidate
    return resolved_script.parent


REPO_ROOT = _resolve_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_env import load_project_env
from train.config import load_train_config

try:
    import modal
except ImportError:  # pragma: no cover - modal is optional in local unit tests
    modal = None


DEFAULT_CONFIG_PATH = "/root/train.yaml"
DEFAULT_MODAL_VOLUME_NAME = "mafia-train-artifacts"
DEFAULT_MODAL_VOLUME_MOUNT_PATH = "/root/artifacts"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/modal"
LOCAL_CONFIG_PATH = REPO_ROOT / "train.yaml"


def _modal_resource_kwargs(config_path: str | Path) -> dict[str, object]:
    config = load_train_config(config_path)
    resource_kwargs: dict[str, object] = {
        "cpu": float(config.modal.cpu_count),
        "memory": config.modal.memory_gb * 1024,
        "timeout": config.modal.timeout_seconds,
    }
    if config.modal.gpu_type.lower() != "none":
        resource_kwargs["gpu"] = config.modal.gpu_type
    return resource_kwargs


def _resolve_modal_checkpoint_root(
    *,
    checkpoint_root: str | None,
    config_path: str,
    volume_mount_path: str = DEFAULT_MODAL_VOLUME_MOUNT_PATH,
) -> str:
    config = load_train_config(config_path)
    configured_root = checkpoint_root or config.modal.checkpoint_location
    candidate = Path(configured_root)
    volume_root = Path(volume_mount_path).resolve(strict=False)
    resolved_candidate = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (volume_root / candidate).resolve(strict=False)
    )
    try:
        resolved_candidate.relative_to(volume_root)
    except ValueError as exc:
        raise ValueError(
            f"Modal checkpoint_root must live under {volume_root}, got {resolved_candidate}"
        ) from exc
    return str(resolved_candidate)


if modal is not None:
    local_config = load_train_config(LOCAL_CONFIG_PATH)
    volume = modal.Volume.from_name(DEFAULT_MODAL_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_dir("src", remote_path="/root/src", copy=True)
        .add_local_file("train.yaml", remote_path=DEFAULT_CONFIG_PATH, copy=True)
        .env({"PYTHONPATH": "/root/src"})
    )
    app = modal.App(local_config.modal.app_name)

    @app.function(
        image=image,
        **_modal_resource_kwargs(LOCAL_CONFIG_PATH),
        volumes={DEFAULT_MODAL_VOLUME_MOUNT_PATH: volume},
    )
    def run_remote_training(
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_root: str | None = None,
        seed: int | None = None,
        hf_token: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> dict[str, object]:
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

        from train import run_training_from_config_path

        config = load_train_config(config_path)
        resolved_checkpoint_root = _resolve_modal_checkpoint_root(
            checkpoint_root=checkpoint_root,
            config_path=config_path,
        )
        try:
            summary = run_training_from_config_path(
                config_path=config_path,
                checkpoint_root=resolved_checkpoint_root,
                seed=seed,
            )
        finally:
            volume.commit()
        result = summary.model_dump(mode="json")
        result["modal_volume_name"] = DEFAULT_MODAL_VOLUME_NAME
        result["modal_volume_mount_path"] = DEFAULT_MODAL_VOLUME_MOUNT_PATH
        result["checkpoint_root"] = resolved_checkpoint_root
        result["truthfulqa_scores_path"] = str(
            Path(resolved_checkpoint_root) / config.logging.log_dir_name / "truthfulqa_scores.jsonl"
        )
        return result

    @app.local_entrypoint()
    def main(
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_root: str | None = None,
        seed: int | None = None,
    ) -> None:
        load_project_env(REPO_ROOT / ".env")
        result = run_remote_training.remote(
            config_path=config_path,
            checkpoint_root=checkpoint_root,
            seed=seed,
            hf_token=os.getenv("HF_TOKEN"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        print(result)

else:
    if __name__ == "__main__":
        raise RuntimeError("Modal is not installed. Run `uv add modal` or install it before using scripts/train_modal.py.")
