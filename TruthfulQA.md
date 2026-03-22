# TruthfulQA Benchmark Integration Plan

## Goal

When a new checkpoint model is created, automatically run the TruthfulQA benchmark against that saved model and persist the result in two places:

1. the existing training logs so benchmark execution is visible in the main run history
2. a separate time-series log dedicated to TruthfulQA scores across checkpoints

## Current State

- Checkpoints are created in [`src/train/trainer.py`](/Users/akashwudali/hoohacks/mafia/src/train/trainer.py) inside `DebugTrainer.run_iteration()`.
- The saved checkpoint directory already includes a `model/` subdirectory when the trainable policy exposes `save_pretrained()`.
- Training logging is centralized in [`src/train/logging.py`](/Users/akashwudali/hoohacks/mafia/src/train/logging.py) via `training.log` and `events.jsonl`.
- A minimal TruthfulQA evaluator already exists in [`src/eval/truthfulqa_eval.py`](/Users/akashwudali/hoohacks/mafia/src/eval/truthfulqa_eval.py), but it only supports caller-supplied examples and has no trainer integration.
- Evaluation config already contains `truthfulqa_subset` and `benchmark_interval` in [`src/train/config.py`](/Users/akashwudali/hoohacks/mafia/src/train/config.py), but those settings are not yet used by the trainer.

## Implementation Steps

### 1. Expand the TruthfulQA evaluation module

Update [`src/eval/truthfulqa_eval.py`](/Users/akashwudali/hoohacks/mafia/src/eval/truthfulqa_eval.py) so it can use the official Hugging Face TruthfulQA dataset directly from the training system instead of only caller-supplied test examples.

Planned changes:

- add a loader that reads the official Hugging Face `truthful_qa` dataset for supported configs, starting with `multiple_choice`
- add a result model that includes benchmark metadata useful for logging, at minimum:
  - `checkpoint_id`
  - `subset`
  - `total_examples`
  - `score`
  - `mc1_score`
  - `mc2_score`
  - optional per-question correctness map
- add a helper that evaluates a Hugging Face checkpoint directory by loading the saved model and tokenizer and scoring the official multiple-choice benchmark

Design choice:

- keep the exact-match scorer intact for unit tests
- make the trainer-facing benchmark entrypoint read the official dataset instead of a local in-repo sample set

### 2. Add benchmark-specific logging support

Extend [`src/train/logging.py`](/Users/akashwudali/hoohacks/mafia/src/train/logging.py) with a dedicated TruthfulQA score log file.

Planned changes:

- create a third file in the run log directory, for example `truthfulqa_scores.jsonl`
- add a helper such as `log_truthfulqa_result(...)` that appends one JSON line per evaluated checkpoint
- log enough fields to support plotting score over time:
  - timestamp
  - checkpoint id
  - checkpoint dir
  - update index
  - subset
  - score
  - mc1 score
  - mc2 score
  - total examples

Output expectations:

- the benchmark should also continue to emit standard events through `log_debug_event(...)` so `training.log` and `events.jsonl` capture start/completion/failure records
- the separate score log should stay focused on benchmark results, not generic training events

### 3. Wire evaluation into checkpoint creation

Modify [`src/train/trainer.py`](/Users/akashwudali/hoohacks/mafia/src/train/trainer.py) so benchmark execution happens immediately after a checkpoint model is saved.

Planned flow inside `DebugTrainer.run_iteration()`:

1. save checkpoint state and artifacts
2. save the trainable model into `checkpoint_dir / "model"`
3. if the current update is eligible for evaluation, run TruthfulQA against that saved model
4. log the benchmark result to the main logs and the dedicated score log
5. attach the benchmark score to checkpoint metrics before returning

Interval behavior:

- gate execution on both checkpoint creation and `config.evaluation.benchmark_interval`
- default behavior should evaluate every checkpoint because the current config default is `1`

Failure behavior:

- benchmark failures should be logged explicitly
- prefer not to discard the checkpoint if evaluation fails
- decide whether to raise or continue based on the current repository pattern; the likely first implementation is to log and re-raise only for clearly invalid local logic, but tolerate operational issues if needed

### 4. Persist benchmark score in checkpoint metadata

Update checkpoint state handling in [`src/train/checkpoints.py`](/Users/akashwudali/hoohacks/mafia/src/train/checkpoints.py) and trainer metric assembly so the TruthfulQA score becomes part of the saved checkpoint metadata.

Planned changes:

- add the benchmark score into `CheckpointState.metrics`
- use clear keys such as `truthfulqa_score`, `truthfulqa_mc1_score`, and `truthfulqa_mc2_score`
- optionally include the evaluated subset using either:
  - another metric-like field if kept numeric only, or
  - a dedicated metadata field if non-numeric information is needed later

This ensures a checkpoint can be inspected offline without reading run-level logs.

### 5. Keep configuration explicit

Use the existing evaluation config block in [`src/train/config.py`](/Users/akashwudali/hoohacks/mafia/src/train/config.py) and [`train.yaml`](/Users/akashwudali/hoohacks/mafia/train.yaml), rather than hardcoding benchmark behavior in the trainer.

Planned changes:

- confirm `truthfulqa_subset` is the source of which official `truthful_qa` config to run
- use `benchmark_interval` to control evaluation cadence
- document the behavior in `train.yaml` comments or repository docs if needed

If implementation reveals missing knobs, add them narrowly. Examples:

- `enabled: true`
- `truthfulqa_score_log_filename: truthfulqa_scores.jsonl`

Those should only be added if they materially simplify control flow or testing.

### 6. Add tests for trainer integration and logging

Add or extend tests to cover the new behavior without requiring a real model download.

Planned test coverage:

- evaluation helper test:
  - benchmark dataset loading for the official Hugging Face `truthful_qa` configs
  - result object fields and score calculation
- trainer integration test:
  - when a checkpoint is saved, TruthfulQA evaluation is invoked for the new checkpoint
  - the resulting score is written into checkpoint metrics
- logging test:
  - benchmark events appear in `training.log` or `events.jsonl`
  - `truthfulqa_scores.jsonl` is created and contains one record per evaluated checkpoint
- interval test:
  - benchmark runs only on eligible checkpoints based on `benchmark_interval`

Implementation approach for tests:

- inject or monkeypatch the evaluation callable so trainer tests remain fast and deterministic
- avoid loading actual Hugging Face weights in unit tests

## File-Level Change Plan

- [`src/eval/truthfulqa_eval.py`](/Users/akashwudali/hoohacks/mafia/src/eval/truthfulqa_eval.py)
  - add benchmark subset loading and checkpoint evaluation helpers
- [`src/train/logging.py`](/Users/akashwudali/hoohacks/mafia/src/train/logging.py)
  - add dedicated TruthfulQA score logger helper
- [`src/train/trainer.py`](/Users/akashwudali/hoohacks/mafia/src/train/trainer.py)
  - run benchmark after checkpoint save and record results
- [`src/train/checkpoints.py`](/Users/akashwudali/hoohacks/mafia/src/train/checkpoints.py)
  - ensure benchmark metrics are persisted cleanly
- [`tests/eval/test_truthfulqa_smoke.py`](/Users/akashwudali/hoohacks/mafia/tests/eval/test_truthfulqa_smoke.py)
  - extend evaluation coverage
- [`tests/train/test_trainer.py`](/Users/akashwudali/hoohacks/mafia/tests/train/test_trainer.py)
  - add trainer benchmark integration coverage

## Open Implementation Decisions

These will be resolved during coding with the narrowest change that fits the existing architecture:

- whether the benchmark should load the saved checkpoint from disk or evaluate the in-memory trainable policy directly
  - preferred approach: load from the saved checkpoint path so evaluation verifies the exact persisted artifact
- whether benchmark failures should fail the whole training update
  - preferred approach: save checkpoint first, then log benchmark failure distinctly
- whether to store detailed per-question correctness in checkpoint artifacts
  - likely no for now, unless it materially helps debugging

## Verification Plan

After implementation:

1. run targeted tests for TruthfulQA evaluation and trainer integration
2. confirm checkpoint `state.json` includes the TruthfulQA metric
3. confirm the main training logs contain benchmark events
4. confirm the separate TruthfulQA score log is appended on each evaluated checkpoint

## Expected Deliverable

After the implementation is complete, each eligible checkpoint should produce:

- the normal checkpoint directory with `state.json`
- a saved `model/` artifact
- benchmark event records in `training.log` and `events.jsonl`
- one new line in a dedicated `truthfulqa_scores.jsonl` file capturing the score over time
