from eval.truthfulqa_eval import TruthfulQAExample, run_truthfulqa_smoke_evaluation


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
    assert result.total_examples == 2
    assert result.exact_match_score == 1.0
    assert all(result.per_question_correct.values())
