from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from train.hf_policy import compute_completion_logprob, load_huggingface_runtime


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class TruthfulQAExample(FrozenModel):
    question: str
    truthful_answer: str
    accepted_answers: tuple[str, ...] = ()


class TruthfulQAMultipleChoiceExample(FrozenModel):
    question: str
    mc1_choices: tuple[str, ...]
    mc1_labels: tuple[int, ...]
    mc2_choices: tuple[str, ...]
    mc2_labels: tuple[int, ...]


class TruthfulQASmokeResult(FrozenModel):
    checkpoint_id: str | None = None
    subset: str = "smoke"
    total_examples: int
    exact_match_score: float
    per_question_correct: dict[str, bool] = Field(default_factory=dict)


class TruthfulQABenchmarkResult(FrozenModel):
    checkpoint_id: str | None = None
    subset: str = "multiple_choice"
    total_examples: int
    score: float
    mc1_score: float
    mc2_score: float
    per_question_mc1_correct: dict[str, bool] = Field(default_factory=dict)
    per_question_mc2_score: dict[str, float] = Field(default_factory=dict)


def run_truthfulqa_smoke_evaluation(
    *,
    responder: Callable[[str], str],
    examples: Sequence[TruthfulQAExample],
    checkpoint_id: str | None = None,
) -> TruthfulQASmokeResult:
    per_question: dict[str, bool] = {}
    for example in examples:
        response = responder(example.question)
        per_question[example.question] = _is_correct(response, example)
    correct = sum(1 for matched in per_question.values() if matched)
    return TruthfulQASmokeResult(
        checkpoint_id=checkpoint_id,
        subset="smoke",
        total_examples=len(examples),
        exact_match_score=correct / len(examples) if examples else 0.0,
        per_question_correct=per_question,
    )


def load_truthfulqa_examples(subset: str = "generation") -> tuple[TruthfulQAExample, ...]:
    dataset = _load_truthfulqa_dataset(subset=subset)
    examples: list[TruthfulQAExample] = []
    for row in dataset:
        best_answer = row.get("best_answer")
        question = row.get("question")
        if not isinstance(question, str) or not isinstance(best_answer, str):
            continue
        examples.append(
            TruthfulQAExample(
                question=question,
                truthful_answer=best_answer,
            )
        )
    if not examples:
        raise ValueError(f"official TruthfulQA subset '{subset}' did not yield any usable examples")
    return tuple(examples)


def load_truthfulqa_multiple_choice_examples(
    subset: str = "multiple_choice",
) -> tuple[TruthfulQAMultipleChoiceExample, ...]:
    dataset = _load_truthfulqa_dataset(subset=subset)
    examples: list[TruthfulQAMultipleChoiceExample] = []
    for row in dataset:
        question = row.get("question")
        mc1_targets = row.get("mc1_targets")
        mc2_targets = row.get("mc2_targets")
        if not isinstance(question, str):
            continue
        mc1_choices, mc1_labels = _parse_target_block(mc1_targets)
        mc2_choices, mc2_labels = _parse_target_block(mc2_targets)
        if not mc1_choices or not mc2_choices:
            continue
        examples.append(
            TruthfulQAMultipleChoiceExample(
                question=question,
                mc1_choices=mc1_choices,
                mc1_labels=mc1_labels,
                mc2_choices=mc2_choices,
                mc2_labels=mc2_labels,
            )
        )
    if not examples:
        raise ValueError(f"official TruthfulQA subset '{subset}' did not yield any usable examples")
    return tuple(examples)


def run_truthfulqa_checkpoint_evaluation(
    *,
    checkpoint_id: str,
    checkpoint_model_dir: str | Path,
    subset: str = "multiple_choice",
    device: str | None = None,
    cache_dir: str | Path | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> TruthfulQABenchmarkResult:
    del max_new_tokens
    del temperature
    examples = load_truthfulqa_multiple_choice_examples(subset)
    runtime = load_huggingface_runtime(
        model_name=str(checkpoint_model_dir),
        tokenizer_name=str(checkpoint_model_dir),
        device=device,
        cache_dir=cache_dir,
    )
    runtime.model.eval()

    per_question_mc1: dict[str, bool] = {}
    per_question_mc2: dict[str, float] = {}

    for example in examples:
        prompt = _build_truthfulqa_prompt(example.question)
        mc1_logprobs = [
            compute_completion_logprob(
                runtime=runtime,
                prompt=prompt,
                completion=_format_choice_completion(choice),
            )
            for choice in example.mc1_choices
        ]
        mc2_logprobs = [
            compute_completion_logprob(
                runtime=runtime,
                prompt=prompt,
                completion=_format_choice_completion(choice),
            )
            for choice in example.mc2_choices
        ]
        predicted_mc1_index = max(range(len(mc1_logprobs)), key=mc1_logprobs.__getitem__)
        per_question_mc1[example.question] = bool(example.mc1_labels[predicted_mc1_index])
        per_question_mc2[example.question] = _compute_mc2_score(mc2_logprobs, example.mc2_labels)

    mc1_score = sum(per_question_mc1.values()) / len(examples) if examples else 0.0
    mc2_score = sum(per_question_mc2.values()) / len(examples) if examples else 0.0
    return TruthfulQABenchmarkResult(
        checkpoint_id=checkpoint_id,
        subset=subset,
        total_examples=len(examples),
        score=mc2_score,
        mc1_score=mc1_score,
        mc2_score=mc2_score,
        per_question_mc1_correct=per_question_mc1,
        per_question_mc2_score=per_question_mc2,
    )


def _load_truthfulqa_dataset(*, subset: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only when optional deps are absent
        raise RuntimeError(
            "TruthfulQA benchmark requires the Hugging Face datasets package. Install project dependencies before running benchmark evaluation."
        ) from exc
    return load_dataset("truthful_qa", subset, split="validation")


def _parse_target_block(target_block: object) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not isinstance(target_block, dict):
        return (), ()
    choices = target_block.get("choices")
    labels = target_block.get("labels")
    if not isinstance(choices, list) or not isinstance(labels, list) or len(choices) != len(labels):
        return (), ()
    parsed_choices: list[str] = []
    parsed_labels: list[int] = []
    for choice, label in zip(choices, labels, strict=True):
        if not isinstance(choice, str):
            return (), ()
        parsed_choices.append(choice)
        parsed_labels.append(int(label))
    return tuple(parsed_choices), tuple(parsed_labels)


def _compute_mc2_score(logprobs: Sequence[float], labels: Sequence[int]) -> float:
    if not logprobs:
        return 0.0
    max_logprob = max(logprobs)
    probs = [math.exp(value - max_logprob) for value in logprobs]
    denominator = sum(probs)
    if denominator == 0.0:
        return 0.0
    numerator = sum(prob for prob, label in zip(probs, labels, strict=True) if label == 1)
    return numerator / denominator


def _format_choice_completion(choice: str) -> str:
    return f" {choice.strip()}"


def _is_correct(response: str, example: TruthfulQAExample) -> bool:
    normalized = _normalize(response)
    accepted = {_normalize(example.truthful_answer), *(_normalize(answer) for answer in example.accepted_answers)}
    return normalized in accepted


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _build_truthfulqa_prompt(question: str) -> str:
    return f"Q: {question}\nA:"
