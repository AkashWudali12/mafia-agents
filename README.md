# mafia-agents

Foundational contracts and engine skeleton for a Mafia multi-agent RL environment.

The foundational engine defaults to a 6-player setup with exactly one mafia, one doctor, one detective, and three villagers. It also supports compatible 5-player variants with exactly one mafia, optional doctor and detective seats, and villagers filling the remaining seats.

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

If you keep these in a repo-root `.env`, the runtime now auto-loads that file before resolving credentials. Both `KEY=value` and `export KEY=value` formats are supported.

Run only the live proof:

```bash
uv run env RUN_OPENROUTER_LIVE_TESTS=1 pytest tests/integration/test_openrouter_live.py
```

## Local training

The local trainable-model path uses Hugging Face and PyTorch, with a Mac-friendly default model:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

The current default model is public on Hugging Face, so the current implementation does not require an `HF_TOKEN`.
Model files are downloaded and cached locally under `.cache/huggingface/` by default.

Sync dependencies first:

```bash
uv sync --group dev
```

Then run one local training job:

```bash
python scripts/train_local.py --config train.yaml --checkpoint-root checkpoints/local
```

This will:

- load `train.yaml`
- instantiate the local Hugging Face trainable policy
- run grouped rollouts
- apply a PyTorch optimizer step
- save a checkpoint under `checkpoints/local`
- write detailed debug logs under `checkpoints/local/logs/`

The debug log files are:

- `training.log`: readable timestamped debug log
- `events.jsonl`: structured JSONL event stream for step-by-step debugging

Environment you may need:

- no `HF_TOKEN` is required for the current default public model
- `HF_TOKEN` only if you later switch to a gated/private Hugging Face model
- `OPENROUTER_API_KEY` is required for training because opponent seats are sampled from the OpenRouter model pool in `train.yaml`

If you store these in a repo-root `.env`, `scripts/train_local.py`, `scripts/train_modal.py`, the OpenRouter client, and the Hugging Face runtime now auto-load that file during startup.

## Modal training

For remote runs on Modal:

```bash
modal run scripts/train_modal.py
```

Remote artifacts are persisted on a Modal `Volume`, not the container filesystem. By default this script mounts the volume at `/root/artifacts` inside the container and writes training outputs under:

- checkpoints: `/root/artifacts/checkpoints/modal/...`
- TruthfulQA scores: `/root/artifacts/checkpoints/modal/logs/truthfulqa_scores.jsonl`

If you pass `--checkpoint-root`, use a relative path such as `checkpoints/experiment-a` or an absolute path already under `/root/artifacts`. Writing to other absolute paths in Modal will not persist after the container exits.

Expected credentials:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN` if the selected Hugging Face model requires authentication
- `OPENROUTER_API_KEY` only if your opponent setup uses OpenRouter

The Modal training function now attaches the Modal secret named by `modal.openrouter_secret_name` in `train.yaml`. The default is `OPENROUTER_API_KEY`, so if you created a Modal secret with that name and it exposes the `OPENROUTER_API_KEY` env var, the remote worker will see it automatically.

The script still forwards `HF_TOKEN`, `OPENROUTER_API_KEY`, and CUDA/PyTorch runtime env vars such as `PYTORCH_ALLOC_CONF` from your local environment into the remote run. This matters for allocator settings because they must be present in the Modal worker before `torch` initializes. If both are present, the locally forwarded value wins because it is applied inside the worker after startup.

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
