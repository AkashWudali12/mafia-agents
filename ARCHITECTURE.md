# Foundational Contracts

This repository now freezes the initial interfaces needed for parallel work across engine, policy, and training.

## Shared Contracts

- `src/contracts.py` is the single source of truth for `Action`, `Observation`, `EnvironmentConfig`, transcript records, and validation results.
- `Action` is the only action schema used across engine, policies, training, and evaluation.
- `Observation` always contains `public_state`, `private_state`, and `legal_actions`.
- `EnvironmentConfig` captures the foundational v1 rule toggles and defaults to the fixed six-player role mix from the PRD.

## Engine Boundary

- `src/game/engine.py` owns `GameState` and the state transition API.
- `new_game`, `validate_action`, `apply_action`, `advance_phase`, `get_legal_actions`, and `build_observation` are the stable entry points other modules should code against.
- `GameState` is immutable at the interface boundary. Transitions return copied state objects instead of mutating caller-owned structures in place.

## Policy Boundary

- `src/policies/base.py` defines the synchronous `Policy.act(observation) -> Action` adapter.
- The foundational baseline policy only proves the contract; network-backed or async model adapters should wrap this interface later rather than change it.

## Invalid Actions

- The foundational default is normalization to `noop`.
- Validation preserves machine-readable error codes and logs the original submission in `GameState.validation_log`.
