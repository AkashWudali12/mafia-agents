from contracts import ActionType, DiscussionIntent, EliminationRecord, LegalActionSpec, Observation, Phase, PrivateObservationState, PublicObservationState, Role, TranscriptEvent, VoteRecord
from policies.rendering import render_model_prompt, render_observation


def test_render_observation_includes_role_phase_and_living_players() -> None:
    observation = Observation(
        actor=2,
        public_state=PublicObservationState(day=3, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 4)),
        private_state=PrivateObservationState(own_role=Role.DETECTIVE),
    )

    rendered = render_observation(observation)

    assert "actor: 2" in rendered
    assert "role: detective" in rendered
    assert "day: 3" in rendered
    assert "phase: day_discussion" in rendered
    assert "living_players: 0, 1, 2, 4" in rendered


def test_render_observation_includes_machine_readable_legal_actions() -> None:
    observation = Observation(
        actor=1,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
        legal_actions=(
            LegalActionSpec(
                action_type=ActionType.SPEAK,
                legal_targets=(0, 2, 3, 4),
                legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
                allow_message=True,
            ),
            LegalActionSpec(action_type=ActionType.VOTE, legal_targets=(0, 2, 3, 4)),
        ),
    )

    rendered = render_observation(observation)

    assert "legal_actions:" in rendered
    assert "- type=speak targets=0,2,3,4 intents=accuse,defend allow_message=true" in rendered
    assert "- type=vote targets=0,2,3,4" in rendered


def test_render_observation_includes_vote_and_elimination_summaries() -> None:
    observation = Observation(
        actor=1,
        public_state=PublicObservationState(
            day=2,
            phase=Phase.DAY_ANNOUNCEMENT,
            living_players=(0, 1, 2, 4),
            vote_history=(VoteRecord(day=1, voter=3, target=0),),
            elimination_history=(EliminationRecord(day=1, player=3, role=Role.VILLAGER, reason="vote"),),
        ),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
    )

    rendered = render_observation(observation)

    assert "vote_history:" in rendered
    assert "- day=1 voter=3 target=0" in rendered
    assert "elimination_history:" in rendered
    assert "- day=1 player=3 role=villager reason=vote" in rendered


def test_render_observation_includes_transcript_events() -> None:
    observation = Observation(
        actor=4,
        public_state=PublicObservationState(
            day=2,
            phase=Phase.DAY_DISCUSSION,
            living_players=(0, 1, 2, 4),
            transcript=(
                TranscriptEvent(
                    day=2,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=1,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.ACCUSE,
                    target=0,
                    message="0 looks suspicious",
                ),
            ),
        ),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
    )

    rendered = render_observation(observation)

    assert "transcript:" in rendered
    assert (
        "- day=2 phase=day_discussion speaker=1 type=speak intent=accuse "
        "target=0 message=0 looks suspicious"
    ) in rendered


def test_render_observation_truncates_to_recent_transcript_events() -> None:
    observation = Observation(
        actor=4,
        public_state=PublicObservationState(
            day=2,
            phase=Phase.DAY_DISCUSSION,
            living_players=(0, 1, 2, 4),
            transcript=(
                TranscriptEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=0,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.DEFEND,
                    target=1,
                    message="first",
                ),
                TranscriptEvent(
                    day=2,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=1,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.ACCUSE,
                    target=0,
                    message="second",
                ),
                TranscriptEvent(
                    day=2,
                    phase=Phase.DAY_DISCUSSION,
                    speaker=2,
                    action_type=ActionType.SPEAK,
                    intent=DiscussionIntent.QUESTION,
                    target=4,
                    message="third",
                ),
            ),
        ),
        private_state=PrivateObservationState(own_role=Role.VILLAGER),
    )

    rendered = render_observation(observation, max_transcript_events=2)

    assert "- truncated_events=1" in rendered
    assert "message=first" not in rendered
    assert "message=second" in rendered
    assert "message=third" in rendered


def test_render_model_prompt_includes_json_only_instruction_and_schema() -> None:
    observation = Observation(
        actor=1,
        public_state=PublicObservationState(day=1, phase=Phase.DAY_DISCUSSION, living_players=(0, 1, 2, 3, 4)),
        private_state=PrivateObservationState(own_role=Role.DOCTOR),
        legal_actions=(
            LegalActionSpec(
                action_type=ActionType.SPEAK,
                legal_targets=(0, 2, 3, 4),
                legal_intents=(DiscussionIntent.ACCUSE, DiscussionIntent.DEFEND),
                allow_message=True,
            ),
        ),
    )

    prompt = render_model_prompt(observation, max_transcript_events=2)

    assert "Basic game outline:" in prompt
    assert "Do not reveal hidden chain-of-thought, private deliberation, or internal reasoning" in prompt
    assert "Role reminders:" in prompt
    assert "Doctor: protect only living players." in prompt
    assert "Do not act as if you can revive eliminated players" in prompt
    assert "Return exactly one JSON object and no extra text." in prompt
    assert "If you choose speak, you must include a plain-English message addressed to the other players." in prompt
    assert "\"action_type\"" in prompt
    assert "\"target\"" in prompt
    assert "Observation:" in prompt
    assert "role: doctor" in prompt
