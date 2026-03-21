# mafia-agents

Foundational contracts and engine skeleton for a Mafia multi-agent RL environment.

The foundational engine supports 5- or 6-player games with exactly one mafia, optional doctor and detective seats, and villagers filling the remaining seats.

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

## Real model proof path

Engineer 2 now includes a narrow, opt-in proof that one real OpenRouter-backed model can complete:

`Observation -> prompt -> JSON -> Action`

Relevant modules:

- `src/policies/openrouter_client.py`: real OpenRouter client built on Pydantic AI
- `src/policies/openrouter_policy.py`: policy adapter that renders, calls the client, parses, and normalizes
- `src/policies/rendering.py`: model prompt construction
- `src/policies/parsing.py`: JSON parsing and legality normalization

Required environment:

- `OPENROUTER_API_KEY`: OpenRouter API key
- `RUN_OPENROUTER_LIVE_TESTS=1`: explicit opt-in to run the live networked proof
- `OPENROUTER_MODEL`: optional override for the live proof model id
  - default: `openai/gpt-4.1-mini`

Run only the live proof:

```bash
RUN_OPENROUTER_LIVE_TESTS=1 OPENROUTER_API_KEY=... uv run pytest tests/integration/test_openrouter_live.py
```

Engineer 3 handoff:

- instantiate `PydanticAiOpenRouterClient`
- pass it into `OpenRouterPolicy`
- source `model`, `temperature`, and `max_tokens` from future config
- keep rollout assignment and seat routing outside the policy layer

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
