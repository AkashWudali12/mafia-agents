import sys
from types import SimpleNamespace

from eval.truthfulqa_eval import (
    TruthfulQAExample,
    load_truthfulqa_examples,
    load_truthfulqa_multiple_choice_examples,
    run_truthfulqa_smoke_evaluation,
    _compute_mc2_score,
)


def test_truthfulqa_smoke_evaluation_scores_exact_matches() -> None:
    examples = (
        TruthfulQAExample(question="What planet do humans live on?", truthful_answer="Earth"),
        TruthfulQAExample(question="How many days are in a week?", truthful_answer="7", accepted_answers=("seven",)),
    )

    answers = {
        "What planet do humans live on?": "Earth",
        "How many days are in a week?": "seven",
    }
    result = run_truthfulqa_smoke_evaluation(
        responder=lambda question: answers[question],
        examples=examples,
        checkpoint_id="ckpt-1",
    )

    assert result.checkpoint_id == "ckpt-1"
    assert result.subset == "smoke"
    assert result.total_examples == 2
    assert result.exact_match_score == 1.0
    assert all(result.per_question_correct.values())


def test_load_truthfulqa_examples_uses_official_hf_dataset() -> None:
    captured = {}

    def fake_load_dataset(name: str, subset: str, split: str):
        captured["name"] = name
        captured["subset"] = subset
        captured["split"] = split
        return [
            {"question": "Q1", "best_answer": "A1"},
            {"question": "Q2", "best_answer": "A2"},
        ]

    original = sys.modules.get("datasets")
    sys.modules["datasets"] = SimpleNamespace(load_dataset=fake_load_dataset)
    try:
        examples = load_truthfulqa_examples("generation")
    finally:
        if original is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = original

    assert len(examples) == 2
    assert examples[0].question == "Q1"
    assert examples[0].truthful_answer == "A1"
    assert captured == {
        "name": "truthful_qa",
        "subset": "generation",
        "split": "validation",
    }


def test_run_truthfulqa_benchmark_records_subset_name() -> None:
    captured = {}

    def fake_load_dataset(name: str, subset: str, split: str):
        captured["name"] = name
        captured["subset"] = subset
        captured["split"] = split
        return [
            {
                "question": "Q1",
                "mc1_targets": {"choices": ["true", "false"], "labels": [1, 0]},
                "mc2_targets": {"choices": ["true", "false", "maybe"], "labels": [1, 0, 1]},
            }
        ]

    original = sys.modules.get("datasets")
    sys.modules["datasets"] = SimpleNamespace(load_dataset=fake_load_dataset)
    try:
        examples = load_truthfulqa_multiple_choice_examples("multiple_choice")
    finally:
        if original is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = original

    assert len(examples) == 1
    assert examples[0].mc1_choices == ("true", "false")
    assert examples[0].mc1_labels == (1, 0)
    assert examples[0].mc2_labels == (1, 0, 1)
    assert captured == {
        "name": "truthful_qa",
        "subset": "multiple_choice",
        "split": "validation",
    }


def test_compute_mc2_score_returns_probability_mass_on_true_answers() -> None:
    score = _compute_mc2_score([0.0, -10.0, 0.0], [1, 0, 1])

    assert 0.99 < score <= 1.0
