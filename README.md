# mafia-agents

Foundational contracts and engine skeleton for a Mafia multi-agent RL environment.

## Prerequisites

- **Python 3.13+** (see `requires-python` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** — installs Python if needed and manages the virtual environment

Install uv (pick one):

- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Or follow the [official installation guide](https://docs.astral.sh/uv/installation/)

## Setup

From the repository root:

```bash
uv sync --group dev
```

This creates a `.venv`, installs runtime dependencies (e.g. Pydantic), and dev tools (e.g. pytest). The lockfile `uv.lock` pins versions; use `uv lock` after dependency changes.

To sync every dependency group (here, same as above):

```bash
uv sync --all-groups
```

## Run tests

```bash
uv run pytest
```

Tests live under `tests/`; `pythonpath` is configured so imports resolve from `src/`.

## Project layout

| Path | Role |
|------|------|
| `src/` | Application code (`game`, `policies`, `contracts`) |
| `tests/` | Unit tests |
| `PRD.md` | Product requirements |
| `AGENTS.md` | Contributor conventions |

## Adding dependencies

```bash
uv add <package>
uv add --group dev <package>   # dev-only (e.g. tooling)
```

Then commit `pyproject.toml` and `uv.lock`.
