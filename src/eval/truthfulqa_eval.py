from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class TruthfulQAExample(FrozenModel):
    question: str
    truthful_answer: str
    accepted_answers: tuple[str, ...] = ()


class TruthfulQASmokeResult(FrozenModel):
    checkpoint_id: str | None = None
    total_examples: int
    exact_match_score: float
    per_question_correct: dict[str, bool] = Field(default_factory=dict)


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
        total_examples=len(examples),
        exact_match_score=correct / len(examples) if examples else 0.0,
        per_question_correct=per_question,
    )


def _is_correct(response: str, example: TruthfulQAExample) -> bool:
    normalized = _normalize(response)
    accepted = {_normalize(example.truthful_answer), *(_normalize(answer) for answer in example.accepted_answers)}
    return normalized in accepted


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())
