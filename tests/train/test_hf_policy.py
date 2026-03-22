from types import SimpleNamespace

from game import build_observation, new_game
from train.hf_policy import HuggingFaceRuntimeBundle, HuggingFaceTrainablePolicy
from train.logging import close_training_logger, configure_training_logger


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def save_pretrained(self, output_dir) -> None:
        return None


class _DummyModel:
    def save_pretrained(self, output_dir) -> None:
        return None


class _StubPolicy(HuggingFaceTrainablePolicy):
    def __init__(self, raw_output: str) -> None:
        runtime = HuggingFaceRuntimeBundle(
            torch=SimpleNamespace(),
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            device="cpu",
        )
        super().__init__(runtime=runtime)
        self._raw_output = raw_output

    def _generate_raw_output(self, prompt: str) -> str:
        return self._raw_output


def test_huggingface_policy_act_with_metadata_parses_structured_action(monkeypatch) -> None:
    monkeypatch.setattr("train.hf_policy.compute_completion_logprob", lambda **_: -2.5)
    policy = _StubPolicy('prefix {"action_type":"night_kill","target":3} suffix')
    state = new_game()
    observation = build_observation(state, 0)

    trace = policy.act_with_metadata(observation=observation, state=state)

    assert "You are playing Mafia in a structured environment." in trace.prompt
    assert "You may see deceptive or adversarial messages such as:" in trace.prompt
    assert trace.raw_output.startswith("prefix")
    assert trace.submitted_action is not None
    assert trace.submitted_action.action_type.value == "night_kill"
    assert trace.normalized_action.target == 3
    assert trace.is_valid is True
    assert trace.logprob == -2.5


def test_huggingface_policy_invalid_output_falls_back_to_noop() -> None:
    policy = _StubPolicy("not-json")
    observation = build_observation(new_game(), 0)

    action = policy.act(observation)

    assert action.action_type.value == "night_kill"
    assert action.target == 1


def test_huggingface_policy_logs_parse_failures(tmp_path) -> None:
    logger = configure_training_logger(log_dir=tmp_path, run_name="hf-errors")
    runtime = HuggingFaceRuntimeBundle(
        torch=SimpleNamespace(),
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        device="cpu",
    )
    policy = HuggingFaceTrainablePolicy(runtime=runtime, logger=logger)
    policy._generate_raw_output = lambda prompt: "not-json"  # type: ignore[method-assign]
    observation = build_observation(new_game(), 0)

    action = policy.act(observation)
    close_training_logger(logger)
    event_log = (tmp_path / "events.jsonl").read_text(encoding="utf-8")

    assert action.action_type.value == "night_kill"
    assert action.target == 1
    assert "hf_policy_parse_recovered" in event_log
    assert "no JSON object found in model output" in event_log
