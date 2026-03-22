from contracts import ActionType, DiscussionIntent, Phase, ValidationErrorCode
from game import build_observation, new_game
from train import ActionOutput, adapt_action_output, malformed_action_result


def test_adapt_action_output_accepts_valid_typed_action() -> None:
    state = new_game()
    observation = build_observation(state, 0)

    result = adapt_action_output(
        actor=0,
        output=ActionOutput(action_type=ActionType.NIGHT_KILL, target=3),
        observation=observation,
        state=state,
    )

    assert result.parse_error is None
    assert result.submitted_action is not None
    assert result.submitted_action.action_type == ActionType.NIGHT_KILL
    assert result.submitted_action.target == 3
    assert result.is_valid is True
    assert result.normalized_action == result.submitted_action


def test_malformed_action_result_marks_schema_failure() -> None:
    result = malformed_action_result(
        actor=0,
        raw_output="not-json",
        parse_error="schema failure",
    )

    assert result.submitted_action is None
    assert result.is_valid is False
    assert result.parse_error == "schema failure"
    assert result.validation_errors == (ValidationErrorCode.MALFORMED_ACTION,)
    assert result.normalized_action.action_type == ActionType.NOOP


def test_adapt_action_output_normalizes_illegal_but_well_typed_action() -> None:
    state = new_game()
    observation = build_observation(state, 0)

    result = adapt_action_output(
        actor=0,
        output=ActionOutput(action_type=ActionType.NIGHT_KILL, target=0),
        observation=observation,
        state=state,
    )

    assert result.submitted_action is not None
    assert result.submitted_action.target == 0
    assert result.is_valid is False
    assert ValidationErrorCode.INVALID_ACTION_TYPE in result.validation_errors or ValidationErrorCode.SELF_TARGET_FORBIDDEN in result.validation_errors
    assert result.normalized_action.action_type == ActionType.NOOP


def test_adapt_action_output_fills_missing_speak_message() -> None:
    state = new_game()
    state = state.model_copy(update={"phase": Phase.DAY_DISCUSSION})
    observation = build_observation(state, 0)
    observation = observation.model_copy(
        update={
            "legal_actions": (
                observation.legal_actions[0].model_copy(
                    update={
                        "action_type": ActionType.SPEAK,
                        "legal_targets": (1, 2, 3, 4),
                        "legal_intents": (DiscussionIntent.ACCUSE,),
                        "allow_message": True,
                    }
                ),
            )
        }
    )

    result = adapt_action_output(
        actor=0,
        output=ActionOutput(action_type=ActionType.SPEAK, target=1, intent=DiscussionIntent.ACCUSE),
        observation=observation,
        state=state,
    )

    assert result.submitted_action is not None
    assert result.submitted_action.message is None
    assert result.normalized_action.action_type == ActionType.SPEAK
    assert result.normalized_action.message == f"{observation.public_state.player_labels[1]} is my strongest suspicion right now."
