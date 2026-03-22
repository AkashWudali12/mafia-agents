from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_env import load_project_env

try:
    import modal
except ImportError:  # pragma: no cover - modal is optional in local unit tests
    modal = None


DEFAULT_CONFIG_PATH = "/root/train.yaml"
DEFAULT_CHECKPOINT_ROOT = "/root/checkpoints/modal"


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_dir("src", remote_path="/root/src", copy=True)
        .add_local_file("train.yaml", remote_path=DEFAULT_CONFIG_PATH, copy=True)
        .env({"PYTHONPATH": "/root/src"})
    )
    app = modal.App("mafia-train")

    @app.function(image=image, timeout=7200)
    def run_remote_training(
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
        seed: int | None = None,
        hf_token: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> dict[str, object]:
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

        from train import run_training_from_config_path

        summary = run_training_from_config_path(
            config_path=config_path,
            checkpoint_root=checkpoint_root,
            seed=seed,
        )
        return summary.model_dump(mode="json")

    @app.local_entrypoint()
    def main(
        config_path: str = DEFAULT_CONFIG_PATH,
        checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
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
