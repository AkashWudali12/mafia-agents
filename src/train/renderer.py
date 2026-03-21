from __future__ import annotations

from contracts import Observation, TranscriptEvent


def render_observation_prompt(observation: Observation, *, transcript_window: int = 8) -> str:
    public_state = observation.public_state
    private_state = observation.private_state

    sections = [
        f"You are player {observation.actor}.",
        f"Role: {private_state.own_role}",
        f"Day: {public_state.day}",
        f"Phase: {public_state.phase}",
        f"Living players: {list(public_state.living_players)}",
        "Private information:",
        _render_private_state(observation),
        "Recent transcript:",
        _render_transcript(public_state.transcript, transcript_window=transcript_window),
        "Legal actions:",
        _render_legal_actions(observation),
        "Output exactly one JSON object matching:",
        '{"action_type": "...", "target": ..., "intent": "...", "message": "..."}',
    ]
    return "\n".join(sections)


def _render_private_state(observation: Observation) -> str:
    private_state = observation.private_state
    details = [
        f"- own_role: {private_state.own_role}",
        f"- mafia_teammates: {list(private_state.mafia_teammates)}",
        f"- investigation_results: {[_render_investigation(result) for result in private_state.investigation_results]}",
        f"- last_protection_target: {private_state.last_protection_target}",
    ]
    return "\n".join(details)


def _render_transcript(transcript: tuple[TranscriptEvent, ...], *, transcript_window: int) -> str:
    if not transcript:
        return "- none"
    rendered_events = []
    for event in transcript[-transcript_window:]:
        rendered_events.append(
            f"- day={event.day} phase={event.phase} speaker={event.speaker} "
            f"action_type={event.action_type} intent={event.intent} target={event.target} message={event.message!r}"
        )
    return "\n".join(rendered_events)


def _render_legal_actions(observation: Observation) -> str:
    if not observation.legal_actions:
        return "- none"
    lines: list[str] = []
    for spec in observation.legal_actions:
        lines.append(
            f"- action_type: {spec.action_type}, legal_targets: {list(spec.legal_targets)}, "
            f"legal_intents: {[intent.value for intent in spec.legal_intents]}, allow_message: {spec.allow_message}"
        )
    return "\n".join(lines)


def _render_investigation(result: object) -> str:
    day = getattr(result, "day")
    target = getattr(result, "target")
    alignment = getattr(result, "alignment")
    role = getattr(result, "role")
    return f"day={day}, target={target}, alignment={alignment}, role={role}"
