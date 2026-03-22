import os

from test_openrouter_truthfulqa_live import (
    _build_mc1_prompt,
    _openrouter_chat_completion,
    _parse_choice_index,
)

from datasets import load_dataset


def test_openrouter_truthfulqa_manual_run() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running this manual TruthfulQA integration test.")

    model = os.getenv("OPENROUTER_TRUTHFULQA_MODEL", "openai/gpt-4.1-mini")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation[:3]")

    correct = 0
    total = 0
    for row in dataset:
        prompt = _build_mc1_prompt(
            question=row["question"],
            choices=row["mc1_targets"]["choices"],
        )
        raw_response = _openrouter_chat_completion(prompt=prompt, model=model)
        predicted_index = _parse_choice_index(raw_response)
        if predicted_index is None:
            raise AssertionError(f"model did not return a parseable choice index: {raw_response!r}")
        correct_index = next(
            index
            for index, label in enumerate(row["mc1_targets"]["labels"])
            if int(label) == 1
        )
        total += 1
        if predicted_index == correct_index:
            correct += 1

    assert total == 3
    print(f"TruthfulQA MC1 live score for {model}: {correct}/{total} = {correct / total:.3f}")
