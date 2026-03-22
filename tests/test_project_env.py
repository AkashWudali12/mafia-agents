import os

from project_env import load_project_env


def test_load_project_env_supports_export_prefix(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        'export OPENROUTER_API_KEY=test-key\nHF_TOKEN="hf-secret"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = load_project_env()

    assert loaded == env_path.resolve()
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert os.environ["HF_TOKEN"] == "hf-secret"


def test_load_project_env_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-value")

    load_project_env()

    assert os.environ["OPENROUTER_API_KEY"] == "existing-value"
