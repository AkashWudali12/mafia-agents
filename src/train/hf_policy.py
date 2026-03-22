from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from contracts import Action, ActionType, DiscussionIntent, Observation, noop_action
from game import GameState
from policies.parsing import normalize_action_for_observation
from policies.schemas import ModelActionPayload
from project_env import load_project_env
from train.parser import ParsedActionResult, adapt_action_output, malformed_action_result
from train.renderer import render_observation_prompt
from train.logging import log_debug_event, log_error_event


DEFAULT_TRAINABLE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class HuggingFaceActionTrace(FrozenModel):
    prompt: str
    raw_output: str
    logprob: float | None
    submitted_action: Action | None
    normalized_action: Action
    is_valid: bool
    validation_errors: tuple[Any, ...] = ()
    parse_error: str | None = None


@dataclass(frozen=True)
class HuggingFaceRuntimeBundle:
    torch: Any
    model: Any
    tokenizer: Any
    device: Any


class HuggingFaceTrainablePolicy:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_TRAINABLE_MODEL,
        tokenizer_name: str | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        transcript_window: int = 8,
        runtime: HuggingFaceRuntimeBundle | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name or model_name
        self._device_preference = device
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(".cache/huggingface")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._transcript_window = transcript_window
        self._logger = logger
        self._runtime = runtime or load_huggingface_runtime(
            model_name=model_name,
            tokenizer_name=self._tokenizer_name,
            device=device,
            cache_dir=self._cache_dir,
        )
        if self._runtime.tokenizer.pad_token_id is None and self._runtime.tokenizer.eos_token_id is not None:
            self._runtime.tokenizer.pad_token = self._runtime.tokenizer.eos_token
        log_debug_event(
            self._logger,
            "hf_policy_initialized",
            model_name=self._model_name,
            tokenizer_name=self._tokenizer_name,
            device=str(self._runtime.device),
            cache_dir=str(self._cache_dir),
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

    @property
    def model(self) -> Any:
        return self._runtime.model

    @property
    def tokenizer(self) -> Any:
        return self._runtime.tokenizer

    @property
    def device(self) -> Any:
        return self._runtime.device

    def act(self, observation: Observation) -> Action:
        prompt = render_observation_prompt(observation, transcript_window=self._transcript_window)
        raw_output = self._generate_raw_output(prompt)
        action = _parse_action_without_state(
            actor=observation.actor,
            raw_output=raw_output,
            observation=observation,
            logger=self._logger,
        )
        log_debug_event(
            self._logger,
            "hf_policy_action_generated",
            actor=observation.actor,
            raw_output=raw_output,
            normalized_action=action,
        )
        return action

    def act_with_metadata(self, *, observation: Observation, state: GameState) -> HuggingFaceActionTrace:
        prompt = render_observation_prompt(observation, transcript_window=self._transcript_window)
        raw_output = self._generate_raw_output(prompt)
        parsed = _parse_action_with_state(
            actor=observation.actor,
            raw_output=raw_output,
            observation=observation,
            state=state,
            logger=self._logger,
        )
        if parsed.validation_errors:
            log_error_event(
                self._logger,
                "hf_policy_invalid_action",
                actor=observation.actor,
                submitted_action=parsed.submitted_action,
                normalized_action=parsed.normalized_action,
                validation_errors=parsed.validation_errors,
                raw_output=raw_output,
            )
        logprob = None
        if parsed.parse_error is None:
            logprob = compute_completion_logprob(
                runtime=self._runtime,
                prompt=prompt,
                completion=_extract_first_json_object(raw_output) or raw_output,
            )
        log_debug_event(
            self._logger,
            "hf_policy_action_trace",
            actor=observation.actor,
            parse_error=parsed.parse_error,
            submitted_action=parsed.submitted_action,
            normalized_action=parsed.normalized_action,
            is_valid=parsed.is_valid,
            validation_errors=parsed.validation_errors,
            logprob=logprob,
        )
        return HuggingFaceActionTrace(
            prompt=prompt,
            raw_output=raw_output,
            logprob=logprob,
            submitted_action=parsed.submitted_action,
            normalized_action=parsed.normalized_action,
            is_valid=parsed.is_valid,
            validation_errors=parsed.validation_errors,
            parse_error=parsed.parse_error,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._runtime.model.save_pretrained(destination)
        self._runtime.tokenizer.save_pretrained(destination)
        config_path = destination / "training_model_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model_name": self._model_name,
                    "tokenizer_name": self._tokenizer_name,
                    "device_preference": self._device_preference,
                    "cache_dir": str(self._cache_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        log_debug_event(
            self._logger,
            "hf_policy_saved",
            output_dir=str(destination),
            model_name=self._model_name,
        )

    def _generate_raw_output(self, prompt: str) -> str:
        runtime = self._runtime
        torch = runtime.torch
        encoded = _build_generation_inputs(runtime=runtime, prompt=prompt)
        encoded = {name: tensor.to(runtime.device) for name, tensor in encoded.items()}
        generate_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "pad_token_id": runtime.tokenizer.pad_token_id,
            "do_sample": self._temperature > 0.0,
        }
        if self._temperature > 0.0:
            generate_kwargs["temperature"] = self._temperature
        log_debug_event(
            self._logger,
            "hf_generation_start",
            prompt_chars=len(prompt),
            input_tokens=int(encoded["input_ids"].shape[1]),
            device=str(runtime.device),
        )
        with torch.no_grad():
            output = runtime.model.generate(**encoded, **generate_kwargs)
        input_length = encoded["input_ids"].shape[1]
        completion_ids = output[0][input_length:]
        raw_output = runtime.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        log_debug_event(
            self._logger,
            "hf_generation_complete",
            prompt_chars=len(prompt),
            output_chars=len(raw_output),
            device=str(runtime.device),
        )
        return raw_output


class HuggingFaceGroupOptimizer:
    def __init__(
        self,
        *,
        policy: HuggingFaceTrainablePolicy,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
    ) -> None:
        runtime = policy._runtime
        self._policy = policy
        self._torch = runtime.torch
        self._optimizer = runtime.torch.optim.AdamW(runtime.model.parameters(), lr=learning_rate)
        self._max_grad_norm = max_grad_norm
        self._logger = policy._logger

    def update(self, grouped_batch: Any) -> dict[str, float]:
        torch = self._torch
        losses = []
        trainable_step_count = 0
        for episode in grouped_batch.episodes:
            for step in episode.steps:
                if not step.is_trainable_actor:
                    continue
                if step.group_normalized_score is None or step.model_prompt is None or step.raw_model_output is None:
                    continue
                completion = _extract_first_json_object(step.raw_model_output) or step.raw_model_output
                if not completion.strip():
                    continue
                score = float(step.group_normalized_score)
                logprob_tensor = compute_completion_logprob_tensor(
                    runtime=self._policy._runtime,
                    prompt=step.model_prompt,
                    completion=completion,
                )
                losses.append(-score * logprob_tensor)
                trainable_step_count += 1

        if not losses:
            log_debug_event(
                self._logger,
                "hf_optimizer_skipped",
                reason="no_trainable_losses",
                num_group_episodes=len(grouped_batch.episodes),
            )
            return {
                "loss": 0.0,
                "num_group_episodes": float(len(grouped_batch.episodes)),
                "num_trainable_steps": 0.0,
            }

        loss = torch.stack(losses).mean()
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        log_debug_event(
            self._logger,
            "hf_optimizer_step",
            loss=float(loss.detach().cpu().item()),
            num_group_episodes=len(grouped_batch.episodes),
            num_trainable_steps=trainable_step_count,
        )
        return {
            "loss": float(loss.detach().cpu().item()),
            "num_group_episodes": float(len(grouped_batch.episodes)),
            "num_trainable_steps": float(trainable_step_count),
        }


def load_huggingface_runtime(
    *,
    model_name: str,
    tokenizer_name: str | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> HuggingFaceRuntimeBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
        raise RuntimeError(
            "Hugging Face training requires optional dependencies. Install torch and transformers before running local training."
        ) from exc

    resolved_device = _resolve_device(torch, device)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else Path(".cache/huggingface")
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    load_project_env()
    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        token=token,
        cache_dir=str(resolved_cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        cache_dir=str(resolved_cache_dir),
    )
    model.to(resolved_device)
    model.train()
    return HuggingFaceRuntimeBundle(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
    )


def compute_completion_logprob(
    *,
    runtime: HuggingFaceRuntimeBundle,
    prompt: str,
    completion: str,
) -> float:
    tensor = compute_completion_logprob_tensor(runtime=runtime, prompt=prompt, completion=completion)
    return float(tensor.detach().cpu().item())


def compute_completion_logprob_tensor(
    *,
    runtime: HuggingFaceRuntimeBundle,
    prompt: str,
    completion: str,
) -> Any:
    tokenizer = runtime.tokenizer
    torch = runtime.torch
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime.device)
    combined = tokenizer(prompt + completion, return_tensors="pt")
    input_ids = combined["input_ids"].to(runtime.device)
    attention_mask = combined["attention_mask"].to(runtime.device)
    labels = input_ids.clone()
    prompt_length = prompt_ids.shape[1]
    labels[:, :prompt_length] = -100
    outputs = runtime.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    supervised_positions = (labels != -100).sum().clamp(min=1)
    return -outputs.loss * supervised_positions


def _parse_action_with_state(
    *,
    actor: int,
    raw_output: str,
    observation: Observation,
    state: GameState,
    logger: logging.Logger | None = None,
) -> ParsedActionResult:
    payload_text = _extract_first_json_object(raw_output)
    if payload_text is None:
        recovered = _recover_malformed_action_with_state(
            actor=actor,
            observation=observation,
            state=state,
            raw_output=raw_output,
            parse_error="no JSON object found in model output",
            logger=logger,
        )
        if recovered is not None:
            return recovered
        log_error_event(
            logger,
            "hf_policy_parse_failed",
            actor=actor,
            parse_error="no JSON object found in model output",
            raw_output=raw_output,
        )
        return malformed_action_result(
            actor=actor,
            raw_output=raw_output,
            parse_error="no JSON object found in model output",
        )
    try:
        payload = ModelActionPayload.model_validate_json(payload_text)
    except Exception as exc:
        recovered = _recover_malformed_action_with_state(
            actor=actor,
            observation=observation,
            state=state,
            raw_output=raw_output,
            parse_error=str(exc),
            logger=logger,
        )
        if recovered is not None:
            return recovered
        log_error_event(
            logger,
            "hf_policy_parse_failed",
            actor=actor,
            parse_error=str(exc),
            raw_output=raw_output,
            payload_text=payload_text,
        )
        return malformed_action_result(
            actor=actor,
            raw_output=raw_output,
            parse_error=str(exc),
        )
    return adapt_action_output(
        actor=actor,
        output=payload,
        observation=observation,
        state=state,
        raw_output=raw_output,
    )


def _parse_action_without_state(
    *,
    actor: int,
    raw_output: str,
    observation: Observation,
    logger: logging.Logger | None = None,
) -> Action:
    payload_text = _extract_first_json_object(raw_output)
    if payload_text is None:
        recovered = _recover_malformed_action_without_state(
            actor=actor,
            observation=observation,
            raw_output=raw_output,
            parse_error="no JSON object found in model output",
            logger=logger,
        )
        if recovered is not None:
            return recovered
        log_error_event(
            logger,
            "hf_policy_parse_failed",
            actor=actor,
            parse_error="no JSON object found in model output",
            raw_output=raw_output,
        )
        return noop_action(actor)
    try:
        payload = ModelActionPayload.model_validate_json(payload_text)
    except Exception as exc:
        recovered = _recover_malformed_action_without_state(
            actor=actor,
            observation=observation,
            raw_output=raw_output,
            parse_error=str(exc),
            logger=logger,
        )
        if recovered is not None:
            return recovered
        log_error_event(
            logger,
            "hf_policy_parse_failed",
            actor=actor,
            parse_error=str(exc),
            raw_output=raw_output,
            payload_text=payload_text,
        )
        return noop_action(actor)
    action = Action(
        actor=actor,
        action_type=payload.action_type,
        target=payload.target,
        intent=payload.intent,
        message=payload.message,
    )
    return normalize_action_for_observation(action, observation)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _build_generation_inputs(*, runtime: HuggingFaceRuntimeBundle, prompt: str) -> dict[str, Any]:
    tokenizer = runtime.tokenizer
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Mafia game policy. Reply with exactly one compact JSON object and no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            encoded = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            encoded = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if not isinstance(encoded, dict):
                encoded = {
                    "input_ids": encoded,
                    "attention_mask": runtime.torch.ones_like(encoded),
                }
        if isinstance(encoded, dict):
            return encoded
    return tokenizer(prompt, return_tensors="pt")


def _recover_malformed_action_with_state(
    *,
    actor: int,
    observation: Observation,
    state: GameState,
    raw_output: str,
    parse_error: str,
    logger: logging.Logger | None = None,
) -> ParsedActionResult | None:
    recovered_action = _recover_action_from_observation(actor=actor, observation=observation)
    if recovered_action is None:
        return None
    log_debug_event(
        logger,
        "hf_policy_parse_recovered",
        actor=actor,
        parse_error=parse_error,
        recovered_action=recovered_action,
        raw_output=raw_output,
    )
    return adapt_action_output(
        actor=actor,
        output=ModelActionPayload(
            action_type=recovered_action.action_type,
            target=recovered_action.target,
            intent=recovered_action.intent,
            message=recovered_action.message,
        ),
        observation=observation,
        state=state,
        raw_output=raw_output,
    )


def _recover_malformed_action_without_state(
    *,
    actor: int,
    observation: Observation,
    raw_output: str,
    parse_error: str,
    logger: logging.Logger | None = None,
) -> Action | None:
    recovered_action = _recover_action_from_observation(actor=actor, observation=observation)
    if recovered_action is None:
        return None
    log_debug_event(
        logger,
        "hf_policy_parse_recovered",
        actor=actor,
        parse_error=parse_error,
        recovered_action=recovered_action,
        raw_output=raw_output,
    )
    return normalize_action_for_observation(recovered_action, observation)


def _recover_action_from_observation(*, actor: int, observation: Observation) -> Action | None:
    if len(observation.legal_actions) != 1:
        return None
    legal_action = observation.legal_actions[0]
    if legal_action.action_type == ActionType.NOOP:
        return noop_action(actor)
    if legal_action.action_type == ActionType.SPEAK:
        default_intent = legal_action.legal_intents[0] if legal_action.legal_intents else DiscussionIntent.QUESTION
        return Action(
            actor=actor,
            action_type=ActionType.SPEAK,
            target=None,
            intent=default_intent,
            message="I want the table to compare reads before we lock in a vote.",
        )
    if legal_action.legal_targets:
        return Action(
            actor=actor,
            action_type=legal_action.action_type,
            target=legal_action.legal_targets[0],
        )
    return None


def _resolve_device(torch: Any, requested: str | None) -> Any:
    if requested:
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
