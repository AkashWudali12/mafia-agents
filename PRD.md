# PRD: Multi-Agent RL Environment for LLM Agents Playing Mafia

## 1. Overview

This project is a multi-agent reinforcement learning system for training an LLM-based agent to play **Mafia** in a structured, partially observable social-deduction environment. The primary training target is **Qwen3 8B**, and the trainable agent may be assigned **any supported role** in a given episode. Training runs execute on **Modal**. During training, the learned policy competes against an opponent pool composed of multiple external model-backed agents accessed through **OpenRouter**, with reward/scoring interpreted relative to the trainable agent’s current role and faction.

The system includes:

- a Python game engine for Mafia
- support for hidden roles and multi-phase gameplay
- a structured action interface suitable for RL
- natural-language rendering for dialogue
- an opponent-pool self-play and cross-play training framework
- a `train.yaml` file for centralized training configuration
- evaluation, logging, and reproducibility tooling

The purpose of the project is to create a robust, trainable RL environment that captures the strategic, deceptive, and communicative dynamics of Mafia while remaining operationally stable enough for repeated large-scale training.

---

## 2. Goals

### Primary goals

- Train a **Qwen3 8B** policy to play the **full game across all supported roles**, not just a single fixed seat.
- Run **5-player** games with composition **1 mafia, 1 doctor, 1 detective, 2 villagers**.
- Support **doctor**, **detective**, **villager**, and **mafia** roles with the above counts.
- Use **Modal** as the execution platform for scalable training jobs.
- Use an **opponent pool of OpenRouter models** during training to improve robustness and reduce overfitting to a narrow set of behaviors.
- Define training configuration in a reusable `train.yaml` file.
- Build a clean architecture that separates game logic, observation building, policy execution, language rendering, training orchestration, and evaluation.

* Support structured dialogue acts and optional text messages.
* Enable population-based evaluation and checkpoint benchmarking.
* Produce reproducible experiments with deterministic seeds where possible.

### Non-goals for v1

- Full free-form token-level RL over unconstrained chat.
- Human multiplayer interface.
- Production deployment as a public game.
- Support for every Mafia variant and house rule.

---

## 3. Why Mafia

Mafia is a strong RL environment for LLM agents because it combines:

- hidden information
- long-horizon planning
- team coordination
- deception and bluffing
- persuasion and social inference
- role-conditioned objectives

Unlike board games with fully structured state, Mafia requires agents to reason about both explicit game state and socially generated evidence. This makes it especially useful for training models that must act strategically under ambiguity.

---

## 4. Product Vision

Build a research-grade Python framework where a trainable LLM policy can repeatedly play Mafia across **mafia, doctor, detective, and villager** assignments against a diverse pool of scripted and model-based opponents, learn from the resulting trajectories, and improve its strategic performance across the game as a whole rather than for only one seat.

The platform should be modular enough that:

- the game engine can run independently of model choice
- policies can be swapped in or out
- OpenRouter-backed opponents can be added without rewriting the training core
- training configuration can be modified through YAML rather than code changes

---

## 5. Target Users

### Primary users

- the project owner building and training the system
- researchers experimenting with social-deduction RL
- developers working on multi-agent LLM benchmarking

### Secondary users

- collaborators who want to add new roles or baselines
- future users who want to adapt the framework to Werewolf / Among Us–style games

---

## 6. Game Specification

### Base game size

The v1 game is **always 5 players**: **1 mafia, 1 doctor, 1 detective, 2 villagers**. This keeps the environment tractable and matches the training setup, where the trainable agent can occupy any one seat while the remaining seats are filled by opponent-controlled policies.

### Roles in v1

- 1 Mafia
- 1 Detective
- 1 Doctor
- 2 Villagers

This gives the game:

- one adversarial elimination role
- one information-gathering role
- one protection role
- baseline town members

### Win conditions

- **Town wins** when all mafia are eliminated.
- **Mafia wins** when the number of living mafia is greater than or equal to the number of living non-mafia players.

### Game phases

Each game progresses through repeating phases:

1. `night_mafia`
2. `night_doctor`
3. `night_detective`
4. `day_announcement`
5. `day_discussion`
6. `day_voting`
7. `resolution`

### Night rules

#### Mafia

- Chooses one living target to kill.

#### Doctor

- Chooses one living target to protect.
- If the doctor protects the mafia’s target, the kill is prevented.
- Whether self-protect is allowed should be configurable in `train.yaml`.

#### Detective

- Chooses one living target to investigate.
- Receives private information about whether the target is mafia.
- Whether exact role or alignment is returned should be configurable, but v1 should default to **alignment result**.

### Day announcement

- Reveal whether someone died overnight.
- Reveal the eliminated player’s role by default in v1.
- If the doctor saved the target, announce that no one died.

### Discussion

- Each living player gets a fixed number of structured discussion turns.
- A turn may include a dialogue act and optional text.

### Voting

- Each living player votes to eliminate one living player.
- Player with plurality is eliminated.
- Tie-break behavior should be configurable; default to deterministic lowest-player-id among top vote-getters.

---

## 7. RL Environment Design

### Environment type

The environment is a **multi-agent partially observable turn-based environment**. It should be implemented in Python with a clean internal engine.

### Design principle

The core environment should use **structured actions**, not unconstrained free-form language, for game-state transitions.

Language should be treated as:

- optional metadata attached to actions
- visible to other agents through the transcript
- influential only through opponent policies, not direct environment parsing

This keeps the environment stable and trainable.

### Action schema

Each agent action should have fields similar to:

- `action_type`
- `target`
- `intent`
- `message`

#### Allowed action types

- `noop`
- `speak`
- `vote`
- `night_kill`
- `protect`
- `investigate`

#### Discussion intents

- `accuse`
- `defend`
- `claim_villager`
- `claim_detective`
- `claim_doctor`
- `question`
- `coordinate`

The environment should validate actions against legal moves for the current phase and agent.

### Observation design

Each player observation should include:

#### Public state

- current day number
- current phase
- living players
- public transcript
- public vote history
- elimination history
- last night outcome

#### Private state

- own role
- mafia teammate identities if mafia count > 1 in future variants
- detective investigation results
- doctor protection history if relevant

#### Legal actions

- current legal action types
- legal targets
- legal intents for discussion

### Transcript design

Every public statement and vote should be logged in a structured transcript with fields such as:

- day
- phase
- speaker
- action type
- intent
- target
- message

---

## 8. Doctor Role Requirements

The doctor must be fully supported in the core game engine and in observation/reward logic.

### Functional requirements

- Doctor acts during `night_doctor`.
- Doctor chooses one living target.
- The chosen target is protected from the mafia kill for that night.
- If mafia targets a protected player, the kill fails.
- Protection resolution occurs before the day announcement.

### Configurability

The following must be configurable in `train.yaml`:

- whether self-protect is allowed
- whether repeated protection of the same target on consecutive nights is allowed
- whether the doctor is told their save succeeded

### RL considerations

The doctor’s policy must learn from sparse signals, so the system should support shaping rewards such as:

- positive reward for successful save
- small negative reward for protecting a player who would not have been attacked only if such shaping is explicitly enabled

The default shaping should remain conservative.

---

## 9. Reward Design

### Primary reward

The main reward signal should be terminal, team-based, and interpreted relative to the trainable agent’s assigned role for that episode.

- `+1.0` if the agent’s faction wins
- `-1.0` if the agent’s faction loses

The training system must not assume mafia-specific scoring when the trainable agent is playing a town role. Reward computation should always be derived from:

- the trainable agent’s role
- the trainable agent’s faction
- the legal responsibilities and objectives of that role

### Optional shaping rewards

Shaping rewards should be small relative to terminal reward magnitude.

#### For town-aligned roles

- small positive reward for voting out mafia
- small negative reward for voting out town

#### For mafia

- small positive reward for successful elimination of a town player
- small negative reward when mafia is eliminated

#### For detective

- small positive reward for correctly identifying mafia
- optional delayed reward if identified mafia is later eliminated

#### For doctor

- small positive reward for successful protection that prevents a kill

#### Cross-role reward requirements

- Reward logic must be role-aware at rollout time rather than hard-coded to a single seat.
- Shaping terms should only be applied when they are semantically valid for the current role.
- Metrics and logged reward breakdowns should include the role under which each reward was earned.
- If reward normalization is used, it should support per-role normalization so one role’s shaping density does not dominate training.

### Reward design principles

- Keep shaping low magnitude to avoid reward hacking.
- Never reward language quality directly in v1.
- Let language matter only via downstream strategic effects.

---

## 10. Training Setup

### Training target

- **Qwen3 8B** is the only trainable policy in v1.
- **Qwen3 8B may occupy any supported role in v1.** The assigned seat for the trainable policy should be selected at episode start according to configuration. All non-trainable seats are played by models from the opponent pool (or other fixed policies).

### Training platform

- Training runs execute on **Modal**.
- Modal is responsible for job orchestration, environment setup, GPU allocation, checkpoint writing, and logging integration.

### Opponent pool

During training, the trainable acting seat is **Qwen3 8B** regardless of whether that seat is mafia, doctor, detective, or villager for the episode; the other four seats in each 5-player game are filled from the opponent pool (external OpenRouter-backed models or configured fixed policies).

Initial opponent pool:

- Grok 4.1 Fast
- GPT-4.1 Mini
- Gemini 3 Flash
- DeepSeek V3
- Kimi K2.5
- GLM 4.7
- Claude Haiku 4.5
- GPT-5 Mini
- Llama 3.2

### Why use opponent pool training

Using multiple opponent policies should:

- reduce overfitting to one model’s quirks
- expose the trainable model to varied deception and reasoning styles
- make performance more robust at evaluation time

### Training mode

The training system should support:

- rollout generation via self-play/cross-play episodes
- checkpointed policy updates
- replay or trajectory logging
- evaluation against frozen baseline pools
- role-conditioned training where the trainable agent’s assigned role may vary by episode
- configurable role sampling so training can be uniform, weighted, or curriculum-based across roles

### Open question to resolve during implementation

Because OpenRouter models are inference-only external opponents, the training loop must clearly define:

- whether training updates are online after each batch of episodes
- whether opponent outputs are cached
- how API latency and cost constraints affect throughput

The PRD assumes these opponents are **fixed external policies**, not trainable participants.

### Role assignment during training

The training pipeline should explicitly support episodic assignment of the trainable policy to different roles.

- At reset, assign the trainable seat to one legal role instance from the configured game composition.
- The role assignment policy should be configurable in `train.yaml`.
- The observation for the trainable agent must always include its own role so a shared policy can condition behavior appropriately.
- Logged trajectories, metrics, and checkpoints should preserve the role label for each episode.

---

## 11. Model Interaction Architecture

### Policy interface

Every policy, including Qwen3 8B and opponent models, should conform to one common interface:

- input: structured observation
- output: structured action JSON

### Prompted opponent models

OpenRouter-backed opponents should receive:

- a natural-language rendering of the observation
- explicit legal action constraints
- instructions to return strict JSON

### Output validation

All model outputs must be:

- parsed from JSON
- validated against legal actions
- converted to `noop` or penalized if invalid, depending on config

### Text generation

The system should support two modes:

1. structured-only actions with templated messages
2. structured actions plus optional LLM-generated `message`

Default training should use **templated or constrained short text** to reduce variance and cost.

### Technology choices

- **Pydantic AI** is the agent framework for LLM inference in v1—built by the Pydantic team, FastAPI-style ergonomics, and **structured outputs** via Pydantic models. Use it for opponent (and other OpenRouter-backed) policies: **`Agent`** instances with **`OpenRouterModel`** and **`OpenRouterProvider`**, model IDs from `train.yaml`, and **result types** (`BaseModel`) that match the environment’s action schema so invalid JSON is caught at validation time. Retries, tool use, and dependencies are available if needed; the **Mafia phase and turn logic stay in plain Python**—the environment remains the source of truth for legality.
- **OpenRouter** is the **inference API** for those external models. Pydantic AI supports OpenRouter as a first-class provider; wire API keys and model names from config rather than ad hoc HTTP clients for each baseline.
- **Pydantic (v2)** and **`pydantic-settings`** remain appropriate for **non-agent** config and shared types (e.g. loading `train.yaml`, shared `BaseModel` definitions used by both the engine and agents).
- The trainable **Qwen3 8B** policy runs inside the training job (e.g. on Modal) and is **not** required to go through OpenRouter unless you explicitly route it that way for convenience.

---

## 12. Modal Requirements

### Infrastructure requirements

The training system running on Modal should support:

- containerized training jobs
- GPU-backed execution for Qwen3 8B policy training
- environment packaging with reproducible dependencies
- checkpoint persistence to remote storage
- artifact logging
- resuming interrupted runs

### Modal responsibilities

- load `train.yaml`
- provision model and environment resources
- run rollout and update loops
- log metrics and save checkpoints
- optionally parallelize simulation workers

### Non-functional requirements

- reproducibility with fixed seeds where feasible
- clean restart behavior
- easy configuration of run name and experiment metadata

---

## 13. `train.yaml` Requirements

The project must include a `train.yaml` file as the canonical place for training configuration.

### Purpose

`train.yaml` should centralize:

- environment settings
- role settings
- trainable role assignment settings
- opponent pool settings
- model settings
- training hyperparameters
- logging/checkpoint settings
- Modal execution settings

### Proposed top-level sections

#### `project`

- name
- experiment\_name
- seed

#### `environment`

- num\_players (v1: **5**)
- roles (v1: **1 mafia, 1 doctor, 1 detective, 2 villagers**)
- trainable\_seat\_selection (v1: configurable; the trainable policy may occupy any supported role seat)
- trainable\_role\_sampling (e.g. uniform, weighted, curriculum)
- discussion\_rounds
- reveal\_roles\_on\_death
- tie\_break\_rule
- doctor\_can\_self\_protect
- doctor\_can\_repeat\_target
- detective\_returns\_alignment\_only
- max\_days

#### `model`

- trainable\_model\_name
- tokenizer\_name
- max\_context\_tokens
- action\_mode

#### `opponents`

- provider
- model\_names
- sampling\_strategy
- temperature
- max\_tokens
- response\_format
- cache\_responses

#### `training`

- algorithm
- learning\_rate
- batch\_size
- episodes\_per\_batch
- gradient\_accumulation\_steps
- checkpoint\_interval
- eval\_interval
- total\_episodes
- max\_steps\_per\_episode

#### `rewards`

- terminal\_win
- terminal\_loss
- vote\_correct
- vote\_incorrect
- detective\_correct\_investigation
- doctor\_successful\_save
- mafia\_successful\_kill

#### `modal`

- app\_name
- gpu\_type
- cpu
- memory\_mb
- timeout\_seconds
- volume\_name
- checkpoint\_path

#### `logging`

- backend
- save\_transcripts
- save\_action\_traces
- save\_observation\_snapshots
- metrics\_path

### Example skeleton

```yaml
project:
  name: mafia-rl
  experiment_name: qwen3-8b-opponent-pool-v1
  seed: 42

environment:
  num_players: 5
  roles: [mafia, doctor, detective, villager, villager]
  trainable_seat_selection: random_supported_role
  trainable_role_sampling: uniform
  discussion_rounds: 2
  reveal_roles_on_death: true
  tie_break_rule: lowest_id
  doctor_can_self_protect: true
  doctor_can_repeat_target: true
  detective_returns_alignment_only: true
  max_days: 10

model:
  trainable_model_name: qwen3-8b
  tokenizer_name: qwen3-8b
  max_context_tokens: 4096
  action_mode: structured_json

opponents:
  provider: openrouter
  model_names:
    - grok-4.1-fast
    - gpt-4.1-mini
    - gemini-3-flash
    - deepseek-v3
    - kimi-k2.5
    - glm-4.7
    - claude-haiku-4.5
    - gpt-5-mini
    - llama-3.2
  sampling_strategy: uniform
  temperature: 0.7
  max_tokens: 256
  response_format: json
  cache_responses: false

training:
  algorithm: grpo
  learning_rate: 1e-5
  batch_size: 8
  episodes_per_batch: 32
  gradient_accumulation_steps: 4
  checkpoint_interval: 100
  eval_interval: 100
  total_episodes: 10000
  max_steps_per_episode: 200

rewards:
  terminal_win: 1.0
  terminal_loss: -1.0
  vote_correct: 0.1
  vote_incorrect: -0.1
  detective_correct_investigation: 0.15
  doctor_successful_save: 0.2
  mafia_successful_kill: 0.2

modal:
  app_name: mafia-rl-train
  gpu_type: A100
  cpu: 8
  memory_mb: 65536
  timeout_seconds: 86400
  volume_name: mafia-rl-checkpoints
  checkpoint_path: /vol/checkpoints

logging:
  backend: wandb
  save_transcripts: true
  save_action_traces: true
  save_observation_snapshots: false
  metrics_path: /vol/metrics
```

---

## 14. Functional Requirements

### Core engine

- Assign hidden roles at reset.
- Support mafia, doctor, detective, villager roles.
- Enforce legal actions by phase.
- Resolve night actions in correct order.
- Build public transcript and role-aware observations.
- Resolve day voting and eliminations.
- Detect terminal states and assign rewards.

### Training system

- Load training config from `train.yaml`.
- Run episodes repeatedly on Modal.
- Query OpenRouter opponents during rollout.
- Train the Qwen3 8B policy from collected experience across episodes where the trainable agent may hold **any supported role**.
- Save checkpoints and metrics.
- Support periodic evaluation.
- Track metrics and reward breakdowns by role and by faction.

### Evaluation system

- Measure win rate overall.
- Measure win rate by role.
- Measure performance against each opponent model.
- Measure invalid action frequency.
- Measure average game length.
- Measure doctor save rate and detective hit rate.

---

## 15. Non-Functional Requirements

- deterministic behavior where game randomness is seeded
- modular codebase with clear separation of concerns
- robust handling of invalid model outputs
- strong logging for debugging episodes
- support for scaling training jobs on Modal
- maintainable config-driven experimentation

---

## 16. Metrics and Success Criteria

### Primary success metrics

- Qwen3 8B improves win rate over time against the opponent pool.
- Qwen3 8B performs above random and above heuristic baselines.
- The environment runs stably across long Modal jobs.
- Qwen3 8B improves under mixed-role training rather than only on a single role.

### Secondary metrics

- win rate by role
- town vs mafia faction performance
- doctor save success rate
- detective investigation accuracy
- vote accuracy
- invalid action rate
- average number of turns per game

### Minimum success bar for v1

- end-to-end training runs complete successfully on Modal
- Qwen3 8B is trained and evaluated across **mafia, doctor, detective, and villager** assignments in 5-player games (1 mafia / 1 doctor / 1 detective / 2 villagers)
- doctor mechanics function correctly
- OpenRouter opponent pool is integrated and sampled during training
- `train.yaml` drives experiment configuration without hard-coded changes

---

## 17. Risks and Open Issues

### API cost and latency

Using multiple OpenRouter opponent models may create cost and throughput bottlenecks.

### Non-stationarity

Even with fixed external opponents, shifting trainable-policy behavior may create unstable dynamics.

### Output formatting failures

LLMs may return invalid JSON or illegal moves, requiring strong validation.

### Sparse rewards

Terminal-only reward is clean but slow; shaping must be balanced carefully.

### Role imbalance

5-player Mafia with doctor and detective may require tuning for fairness; because the trainable agent now plays across roles, the system must avoid over-optimizing toward the most frequently sampled or most densely rewarded roles.

### Data efficiency

Qwen3 8B may require a large number of rollouts to improve meaningfully, especially if dialogue is included.

---

## 18. Suggested Technical Architecture

### Modules

- `env/`: game engine, rules, observation builder
- `agents/`: trainable policy, heuristic bots, **Pydantic AI** + **OpenRouter** wrappers for opponent models
- `training/`: rollout generation, optimization loop, checkpointing
- `config/`: `train.yaml`
- `infra/`: Modal entrypoints, job config, storage integration
- `evaluation/`: benchmark suite and metrics

### Recommended file structure

```text
mafia_rl/
  env/
    core.py
    rules.py
    types.py
    encoding.py
    renderer.py
  agents/
    qwen_policy.py
    heuristic_bot.py
    openrouter_agent.py
  training/
    rollout.py
    trainer.py
    rewards.py
  evaluation/
    evaluate.py
    metrics.py
  infra/
    modal_app.py
  config/
    train.yaml
  tests/
    test_rules.py
    test_doctor.py
    test_rewards.py
```

---

## 19. Milestones

### Milestone 1: Core environment

- implement hidden-role game engine
- add mafia, doctor, detective, villager
- implement transcripts, observations, rewards
- add tests for night resolution and doctor saves

### Milestone 2: Agent interfaces

- define structured action schema
- add heuristic and random agents
- add OpenRouter-backed opponent wrapper

### Milestone 3: Training pipeline

- create `train.yaml`
- implement rollout collection
- integrate Qwen3 8B training loop
- deploy first Modal training run

### Milestone 4: Evaluation

- benchmark against opponent pool
- track role-specific metrics
- compare checkpoints over time

---

## 20. Final Recommendation

The best v1 architecture is:

- **structured RL environment first**
- **doctor included from day one**
- **Qwen3 8B as the only trainable policy, rotating across supported roles**
- **OpenRouter opponent pool as fixed external policies**
- **Modal for scalable rollout and training execution**
- **`train.yaml` as the single source of truth for experiment settings**

This design is ambitious but still grounded. It preserves the social and deceptive aspects of Mafia while avoiding the chaos of unconstrained token-level RL from the start.

