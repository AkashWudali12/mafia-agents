# Parallel Engineering Task Split

This project can be split across three engineers with low coupling, but only if the team agrees on a small set of interfaces first. The critical contracts are:

- `GameState` and phase transition API
- action schema and validation rules
- observation format exposed to policies
- policy adapter interface used by training and evaluation

Without those, parallel work is possible in theory but inefficient in practice.

## Engineer 1: Core Game Engine
Own the deterministic Mafia environment and game-state transitions.

Deliverables:

- `src/game/` modules for roles, phases, state, and resolution logic
- action validation for `night_kill`, `protect`, `investigate`, `vote`, and `speak`
- transcript and vote-history recording
- win-condition checks and terminal state handling
- unit tests for phase ordering, doctor save logic, detective results, and voting tie-breaks

Suggested first milestones:

1. Define `GameState`, `Action`, and `Phase` types.
2. Implement the 6-player fixed-role flow.
3. Add deterministic tests under `tests/game/`.

## Engineer 2: Policy, Observation, and Opponent Layer
Own the interface between the environment and agents.

Deliverables:

- `src/policies/` interfaces for scripted, random, and model-backed agents
- observation builder with public state, private state, and legal actions
- natural-language rendering for transcript-aware agents
- OpenRouter-backed opponent adapter behind a stable client interface
- tests for legal-action generation and role-specific private observations

Suggested first milestones:

1. Define `Policy.act(observation) -> Action`.
2. Build observation serialization from `GameState`.
3. Add a simple scripted baseline for all supported roles and mixed-role training rollouts.

## Engineer 3: Training, Evaluation, and Configuration
Own experiment orchestration and reproducibility.

Deliverables:

- `train.yaml` schema and config loader
- `src/train/` loop for rollouts, logging, checkpoints, and seeds
- `src/eval/` harness for benchmark runs and opponent-pool evaluation
- Modal entrypoints and job packaging
- tests for config parsing and basic rollout execution
- role-conditioned training support so the trainable policy can occupy any supported seat per episode

Suggested first milestones:

1. Define config schema and defaults.
2. Wire one local rollout using the policy interface.
3. Add evaluation outputs for win rate, role-conditioned reward summaries, and episode summaries.

## Coordination Rules
- Freeze shared types before implementation starts.
- Merge Engineer 1’s state model first; Engineers 2 and 3 should code against the agreed interface, not implementation details.
- Use `uv` for Python environment and dependency management once `pyproject.toml` is added.
- Track cross-team decisions in `PRD.md` or a follow-up architecture note, not only in PR comments.
