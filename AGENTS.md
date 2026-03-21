# Repository Guidelines

## Project Structure & Module Organization
This repository is currently documentation-first. The top level contains:

- `README.md`: one-line project summary.
- `PRD.md`: product requirements for the Mafia multi-agent RL environment.

As implementation lands, keep runtime code under `src/`, tests under `tests/`, and experiment or config files under clearly named paths such as `configs/` or `train.yaml`. Match module names to domain concepts from the PRD, for example `src/game/`, `src/policies/`, and `src/eval/`.

## Build, Test, and Development Commands
There is no checked-in build system yet. Until one is added, contributor work is primarily documentation and design review. Use `uv` as the default Python package and environment manager when scaffolding begins.

- `git status`: verify a clean working tree before and after changes.
- `git diff -- README.md PRD.md AGENTS.md`: review documentation edits before committing.
- `rg "train.yaml|Modal|OpenRouter" .`: trace key requirements across repo docs.
- `uv sync`: install project dependencies once `pyproject.toml` is present.
- `uv run pytest`: run the test suite once tests are configured.

When Python tooling is introduced, prefer `uv add`, `uv sync`, and `uv run ...` over ad hoc virtualenv or `pip` workflows.

## Coding Style & Naming Conventions
Use concise, technical prose in docs and keep Markdown sections scannable. For future Python code:

- follow PEP 8 with 4-space indentation
- prefer type hints on public interfaces
- use `snake_case` for modules, functions, and config keys
- use `PascalCase` for classes

Name files after their responsibility, not a temporary experiment name. Example: `src/game/state.py`, not `src/game/stuff.py`.

## Testing Guidelines
No test framework is configured yet. When code is added, place tests in `tests/` and name them `test_<unit>.py`. Favor deterministic unit tests for phase transitions, role logic, action validation, and win conditions described in `PRD.md`.

Every PR that adds behavior should add or update tests. If a feature is not testable yet, state the gap explicitly in the PR description.

## Commit & Pull Request Guidelines
Current history uses short, plain-language commit subjects such as `init` and `added first draft of PRD`. Keep commits imperative, specific, and under roughly 72 characters, for example `add night phase state machine`.

Pull requests should include:

- a short summary of the change
- links to the relevant requirement in `PRD.md`
- notes on validation performed
- screenshots only when UI or rendered artifacts are introduced

## Configuration & Research Notes
Treat training settings as versioned configuration, not inline constants. Prefer checked-in YAML for reproducibility, and document any dependency on Modal, OpenRouter, model names, or secrets without committing credentials. Manage Python dependencies through `uv` and commit the lockfile once dependency management is in place.
