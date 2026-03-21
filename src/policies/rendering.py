from __future__ import annotations

import json

from contracts import LegalActionSpec, Observation

from .schemas import ModelActionPayload


def render_observation(observation: Observation, max_transcript_events: int | None = None) -> str:
    lines = [
        f"actor: {observation.actor}",
        f"role: {observation.private_state.own_role.value}",
        f"day: {observation.public_state.day}",
        f"phase: {observation.public_state.phase.value}",
        "living_players: " + ", ".join(str(player) for player in observation.public_state.living_players),
        "vote_history:",
        *_format_vote_history(observation),
        "elimination_history:",
        *_format_elimination_history(observation),
        "transcript:",
        *_format_transcript(observation, max_transcript_events=max_transcript_events),
        "legal_actions:",
    ]
    if not observation.legal_actions:
        lines.append("- noop")
    else:
        lines.extend(_format_legal_action(action) for action in observation.legal_actions)
    return "\n".join(lines)


def render_model_prompt(observation: Observation, max_transcript_events: int | None = None) -> str:
    schema = json.dumps(ModelActionPayload.model_json_schema(), indent=2, sort_keys=True)
    instructions = "\n".join(
        [
            "You are playing Mafia in a structured environment.",
            "Return exactly one JSON object and no extra text.",
            "Your JSON must follow this schema and only choose from legal_actions.",
            str(schema),
            'If you choose speak, you must include a plain-English message addressed to the other players.',
            'Example speak output: {"action_type":"speak","target":2,"intent":"accuse","message":"Player 2 is dodging the vote discussion. We should pressure them."}',
            "Do not repeat the schema, legal_actions, or transcript in your answer.",
            "If a field is not needed, use null or omit it.",
            "Observation:",
        ]
    )
    return instructions + "\n" + render_observation(
        observation,
        max_transcript_events=max_transcript_events,
    )


def _format_legal_action(action: LegalActionSpec) -> str:
    parts = [f"type={action.action_type.value}"]
    if action.legal_targets:
        parts.append("targets=" + ",".join(str(target) for target in action.legal_targets))
    if action.legal_intents:
        parts.append("intents=" + ",".join(intent.value for intent in action.legal_intents))
    if action.allow_message:
        parts.append("allow_message=true")
    return "- " + " ".join(parts)


def _format_vote_history(observation: Observation) -> list[str]:
    if not observation.public_state.vote_history:
        return ["- none"]
    return [
        f"- day={record.day} voter={record.voter} target={record.target}"
        for record in observation.public_state.vote_history
    ]


def _format_elimination_history(observation: Observation) -> list[str]:
    if not observation.public_state.elimination_history:
        return ["- none"]
    return [
        "- "
        + " ".join(
            [
                f"day={record.day}",
                f"player={record.player}",
                f"role={record.role.value if record.role is not None else 'hidden'}",
                f"reason={record.reason}",
            ]
        )
        for record in observation.public_state.elimination_history
    ]


def _format_transcript(observation: Observation, max_transcript_events: int | None) -> list[str]:
    transcript = observation.public_state.transcript
    if not transcript:
        return ["- none"]
    lines: list[str] = []
    if max_transcript_events is not None and max_transcript_events >= 0 and len(transcript) > max_transcript_events:
        truncated = len(transcript) - max_transcript_events
        lines.append(f"- truncated_events={truncated}")
        transcript = transcript[-max_transcript_events:] if max_transcript_events > 0 else ()
    for event in transcript:
        parts = [
            f"day={event.day}",
            f"phase={event.phase.value}",
            f"speaker={event.speaker}",
            f"type={event.action_type.value}",
        ]
        if event.intent is not None:
            parts.append(f"intent={event.intent.value}")
        if event.target is not None:
            parts.append(f"target={event.target}")
        if event.message is not None:
            parts.append(f"message={event.message}")
        lines.append("- " + " ".join(parts))
    return lines
