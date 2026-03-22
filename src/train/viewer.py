from __future__ import annotations

import threading
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import BaseModel, ConfigDict

from contracts import Action, ActionType, EliminationRecord, NightOutcome, Phase, Role, TranscriptEvent
from game.rules import current_speaker, seat_for_role
from game.state import GameState
from train.trajectory import EpisodeMetadata

_VIEWER_AUTO_ADVANCE_PHASES = frozenset({Phase.DAY_ANNOUNCEMENT, Phase.RESOLUTION})


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, use_enum_values=False)


class ViewerPlayerState(FrozenModel):
    seat: int
    name: str
    initials: str
    alive: bool
    is_trainable: bool
    is_current_speaker: bool = False
    is_turn_highlight: bool = False
    vote_target: int | None = None
    revealed_role: str | None = None


class ViewerBubble(FrozenModel):
    actor: int
    text: str
    style: str = "speech"
    target: int | None = None


class ViewerSnapshot(FrozenModel):
    episode_id: str
    update_index: int
    day: int
    phase: str
    headline: str
    subheadline: str
    players: tuple[ViewerPlayerState, ...]
    active_bubble: ViewerBubble | None = None
    narrator: str | None = None
    transcript: tuple[dict[str, object], ...] = ()
    last_night_outcome: dict[str, object] | None = None
    winner: str | None = None


class ViewerEvent(FrozenModel):
    event_id: str
    kind: str
    episode_id: str
    update_index: int
    step_index: int | None = None
    day: int
    phase: str
    actor: int | None = None
    target: int | None = None
    message: str | None = None
    snapshot: ViewerSnapshot


class ViewerEventSink:
    def publish(self, event: ViewerEvent) -> None:
        raise NotImplementedError


class LiveTrainingViewer(ViewerEventSink):
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765, max_cached_events: int = 5000) -> None:
        self._host = host
        self._port = port
        self._max_cached_events = max_cached_events
        self._condition = threading.Condition()
        self._events: list[tuple[int, str]] = []
        self._next_sse_id = 1
        self._closed = False
        self._httpd: _ViewerHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str | None:
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> str:
        if self._httpd is not None:
            if self.url is None:
                raise RuntimeError("viewer server started without URL")
            return self.url
        self._httpd = _ViewerHTTPServer((self._host, self._port), _ViewerRequestHandler, viewer=self)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mafia-viewer", daemon=True)
        self._thread.start()
        if self.url is None:
            raise RuntimeError("viewer server started without URL")
        return self.url

    def stop(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def publish(self, event: ViewerEvent) -> None:
        payload = event.model_dump_json()
        with self._condition:
            if self._closed:
                return
            event_id = self._next_sse_id
            self._next_sse_id += 1
            self._events.append((event_id, payload))
            if len(self._events) > self._max_cached_events:
                self._events = self._events[-self._max_cached_events :]
            self._condition.notify_all()

    def iter_events(self, last_event_id: int = 0) -> Iterable[tuple[int, str] | None]:
        next_event_id = last_event_id + 1
        while True:
            heartbeat = False
            with self._condition:
                while not self._closed and not any(event_id >= next_event_id for event_id, _ in self._events):
                    self._condition.wait(timeout=15.0)
                    if self._closed:
                        break
                    if not any(event_id >= next_event_id for event_id, _ in self._events):
                        heartbeat = True
                        break
                if self._closed:
                    return
                if heartbeat:
                    batch: list[tuple[int, str]] = []
                else:
                    batch = [(event_id, payload) for event_id, payload in self._events if event_id >= next_event_id]
            if heartbeat:
                yield None
                continue
            for event_id, payload in batch:
                next_event_id = event_id + 1
                yield event_id, payload


def build_episode_start_event(
    *,
    state: GameState,
    metadata: EpisodeMetadata,
    update_index: int,
) -> ViewerEvent:
    return _make_event(
        kind="episode_start",
        state=state,
        metadata=metadata,
        update_index=update_index,
        message="A new game begins around the table.",
        narrator="The players settle in and the game begins.",
    )


def build_transition_events(
    *,
    previous_state: GameState,
    next_state: GameState,
    metadata: EpisodeMetadata,
    update_index: int,
    step_index: int,
    action: Action | None = None,
) -> tuple[ViewerEvent, ...]:
    events: list[ViewerEvent] = []

    if action is not None and action.action_type == ActionType.SPEAK and action.message:
        speaker = _player_label(next_state, action.actor)
        events.append(
            _make_event(
                kind="speech",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=action.actor,
                target=action.target,
                message=action.message,
                bubble=None,
                narrator=f"{speaker}: {action.message}",
            )
        )
    elif action is not None and action.action_type == ActionType.VOTE and action.target is not None:
        events.append(
            _make_event(
                kind="vote",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=action.actor,
                target=action.target,
                message=(
                    f"{_player_label(next_state, action.actor)} points the vote at "
                    f"{_player_label(next_state, action.target)}."
                ),
                narrator=(
                    f"{_player_label(next_state, action.actor)} has cast a vote on "
                    f"{_player_label(next_state, action.target)}."
                ),
            )
        )
    elif action is not None and action.action_type == ActionType.NIGHT_KILL and action.target is not None:
        events.append(
            _make_event(
                kind="night_kill_choice",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=action.actor,
                target=action.target,
                message=(
                    f"{_player_label(next_state, action.actor)} marks "
                    f"{_player_label(next_state, action.target)} for the night kill."
                ),
                narrator=f"The mafia chooses {_player_label(next_state, action.target)} as the night's target.",
            )
        )
    elif action is not None and action.action_type == ActionType.PROTECT and action.target is not None:
        events.append(
            _make_event(
                kind="doctor_save_choice",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=action.actor,
                target=action.target,
                message=(
                    f"{_player_label(next_state, action.actor)} chooses to protect "
                    f"{_player_label(next_state, action.target)} tonight."
                ),
                narrator=f"The doctor places protection on {_player_label(next_state, action.target)}.",
            )
        )
    elif action is not None and action.action_type == ActionType.INVESTIGATE and action.target is not None:
        investigation = _new_investigation(previous_state, next_state)
        if investigation is not None:
            result = investigation.role.value if investigation.role is not None else investigation.alignment.value
            message = (
                f"{_player_label(next_state, action.actor)} investigates "
                f"{_player_label(next_state, investigation.target)} "
                f"and learns they are {result}."
            )
        else:
            message = (
                f"{_player_label(next_state, action.actor)} investigates "
                f"{_player_label(next_state, action.target)}."
            )
        events.append(
            _make_event(
                kind="detective_investigation",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=action.actor,
                target=action.target,
                message=message,
                narrator=message,
            )
        )

    if previous_state.last_night_outcome != next_state.last_night_outcome and next_state.last_night_outcome is not None:
        events.append(
            _make_event(
                kind="night_outcome",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                message=_night_outcome_message(next_state, next_state.last_night_outcome),
                narrator=_night_outcome_message(next_state, next_state.last_night_outcome),
            )
        )

    for elimination in _new_eliminations(previous_state.elimination_history, next_state.elimination_history):
        events.append(
            _make_event(
                kind="elimination",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                actor=elimination.player,
                message=_elimination_message(next_state, elimination),
                narrator=_elimination_message(next_state, elimination),
            )
        )

    if previous_state.phase != next_state.phase:
        events.append(
            _make_event(
                kind="phase_change",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                message=_phase_message(next_state),
                narrator=_phase_message(next_state),
            )
        )

    if previous_state.winner != next_state.winner and next_state.winner is not None:
        events.append(
            _make_event(
                kind="episode_complete",
                state=next_state,
                metadata=metadata,
                update_index=update_index,
                step_index=step_index,
                message=f"{next_state.winner.value.title()} win the game.",
                narrator=f"{next_state.winner.value.title()} have taken the table.",
            )
        )

    return tuple(events)


def build_snapshot(
    *,
    state: GameState,
    metadata: EpisodeMetadata,
    update_index: int,
    bubble: ViewerBubble | None = None,
    narrator: str | None = None,
    acting_seat: int | None = None,
) -> ViewerSnapshot:
    vote_map = {vote.voter: vote.target for vote in state.current_votes}
    current_speaker_seat = _current_speaker(state)
    highlight_seat = acting_seat if acting_seat is not None else _expected_actor_seat(state)
    players = tuple(
        ViewerPlayerState(
            seat=seat,
            name=_player_label(state, seat),
            initials=_seat_initials(state, seat),
            alive=state.alive[seat],
            is_trainable=seat == metadata.trainable_seat,
            is_current_speaker=current_speaker_seat == seat,
            is_turn_highlight=highlight_seat is not None and highlight_seat == seat,
            vote_target=vote_map.get(seat),
            revealed_role=_revealed_role(state, seat),
        )
        for seat in range(state.config.num_players)
    )
    return ViewerSnapshot(
        episode_id=metadata.episode_id,
        update_index=update_index,
        day=state.day,
        phase=state.phase.value,
        headline=f"Day {state.day} • {_phase_label(state.phase)}",
        subheadline=_phase_subheadline(state),
        players=players,
        active_bubble=bubble,
        narrator=narrator,
        transcript=tuple(_transcript_rows(state.transcript, state)),
        last_night_outcome=_night_outcome_payload(state.last_night_outcome),
        winner=state.winner.value if state.winner is not None else None,
    )


class _ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, *, viewer: LiveTrainingViewer) -> None:
        super().__init__(server_address, request_handler_class)
        self.viewer = viewer


class _ViewerRequestHandler(BaseHTTPRequestHandler):
    server: _ViewerHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._write_html()
            return
        if self.path.startswith("/events"):
            self._stream_events()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _write_html(self) -> None:
        html = _viewer_html().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _stream_events(self) -> None:
        last_event_id = self.headers.get("Last-Event-ID", "0")
        try:
            next_after = int(last_event_id)
        except ValueError:
            next_after = 0
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for item in self.server.viewer.iter_events(next_after):
                if item is None:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                event_id, payload = item
                self.wfile.write(f"id: {event_id}\nevent: viewer\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def _make_event(
    *,
    kind: str,
    state: GameState,
    metadata: EpisodeMetadata,
    update_index: int,
    step_index: int | None = None,
    actor: int | None = None,
    target: int | None = None,
    message: str | None = None,
    bubble: ViewerBubble | None = None,
    narrator: str | None = None,
) -> ViewerEvent:
    return ViewerEvent(
        event_id=f"{metadata.episode_id}:{step_index if step_index is not None else 'start'}:{kind}",
        kind=kind,
        episode_id=metadata.episode_id,
        update_index=update_index,
        step_index=step_index,
        day=state.day,
        phase=state.phase.value,
        actor=actor,
        target=target,
        message=message,
        snapshot=build_snapshot(
            state=state,
            metadata=metadata,
            update_index=update_index,
            bubble=bubble,
            narrator=narrator,
            acting_seat=actor,
        ),
    )


def _phase_subheadline(state: GameState) -> str:
    if state.phase == Phase.NIGHT_MAFIA:
        return "The room darkens while the mafia decides in secret."
    if state.phase == Phase.NIGHT_DOCTOR:
        return "A doctor quietly chooses who to protect."
    if state.phase == Phase.NIGHT_DETECTIVE:
        return "A detective studies the table for hidden motives."
    if state.phase == Phase.DAY_ANNOUNCEMENT:
        return "The table reacts to what happened overnight."
    if state.phase == Phase.DAY_DISCUSSION:
        return "Players are talking through accusations and defenses."
    if state.phase == Phase.DAY_VOTING:
        return "Hands go up as the table locks in votes."
    if state.phase == Phase.RESOLUTION:
        return "The vote is being resolved."
    return "The game has reached its final outcome."


def _phase_label(phase: Phase) -> str:
    return phase.value.replace("_", " ").title()


def _phase_message(state: GameState) -> str:
    if state.phase == Phase.DAY_DISCUSSION:
        return "Discussion begins."
    if state.phase == Phase.DAY_VOTING:
        return "The table moves into voting."
    if state.phase == Phase.DAY_ANNOUNCEMENT:
        return "Morning arrives and the group takes stock."
    if state.phase == Phase.NIGHT_MAFIA:
        return "Night falls. The mafia acts in the dark."
    if state.phase == Phase.NIGHT_DOCTOR:
        return "The doctor makes a quiet choice."
    if state.phase == Phase.NIGHT_DETECTIVE:
        return "The detective investigates in secret."
    if state.phase == Phase.RESOLUTION:
        return "The votes are counted."
    return "The game is over."


def _night_outcome_message(state: GameState, outcome: NightOutcome) -> str:
    if outcome.victim is None:
        return "Dawn breaks. No one was targeted overnight."
    if outcome.saved:
        return (
            f"Dawn breaks. {_player_label(state, outcome.victim)} was targeted, "
            "but the doctor saved them."
        )
    if outcome.announced_death is not None:
        return (
            f"Dawn breaks. {_player_label(state, outcome.announced_death)} was killed overnight."
        )
    return "Dawn breaks after a violent night."


def _night_outcome_payload(outcome: NightOutcome | None) -> dict[str, object] | None:
    if outcome is None:
        return None
    return {
        "victim": outcome.victim,
        "saved": outcome.saved,
        "announced_death": outcome.announced_death,
    }


def _new_eliminations(
    previous: tuple[EliminationRecord, ...],
    current: tuple[EliminationRecord, ...],
) -> tuple[EliminationRecord, ...]:
    return current[len(previous) :]


def _new_investigation(previous_state: GameState, next_state: GameState):
    if len(next_state.investigation_history) <= len(previous_state.investigation_history):
        return None
    return next_state.investigation_history[-1]


def _elimination_message(state: GameState, elimination: EliminationRecord) -> str:
    base = f"{_player_label(state, elimination.player)} is eliminated from the game."
    if elimination.role is None:
        return base
    return f"{base} Their role is revealed as {elimination.role.value}."


def _revealed_role(state: GameState, seat: int) -> str | None:
    for elimination in reversed(state.elimination_history):
        if elimination.player == seat and elimination.role is not None:
            return elimination.role.value
    if state.is_terminal and state.winner is not None:
        return state.roles[seat].value
    return None


def _transcript_rows(transcript: tuple[TranscriptEvent, ...], state: GameState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in transcript:
        rows.append(
            {
                "speaker": event.speaker,
                "speaker_name": _player_label(state, event.speaker),
                "action_type": event.action_type.value,
                "target": event.target,
                "message": event.message,
                "intent": event.intent.value if event.intent is not None else None,
            }
        )
    return rows


def _expected_actor_seat(state: GameState) -> int | None:
    """Seat expected to act next (aligned with train.rollout.next_actor)."""
    if state.phase in _VIEWER_AUTO_ADVANCE_PHASES or state.is_terminal:
        return None
    if state.phase == Phase.NIGHT_MAFIA:
        seat = seat_for_role(state, Role.MAFIA)
        return seat if seat >= 0 else None
    if state.phase == Phase.NIGHT_DOCTOR:
        seat = seat_for_role(state, Role.DOCTOR)
        return seat if seat >= 0 else None
    if state.phase == Phase.NIGHT_DETECTIVE:
        seat = seat_for_role(state, Role.DETECTIVE)
        return seat if seat >= 0 else None
    if state.phase == Phase.DAY_DISCUSSION:
        return current_speaker(state)
    if state.phase == Phase.DAY_VOTING:
        for actor in state.living_players:
            if actor not in state.current_voters:
                return actor
        return None
    return None


def _current_speaker(state: GameState) -> int | None:
    if state.phase != Phase.DAY_DISCUSSION:
        return None
    living = state.living_players
    if not living:
        return None
    total_turns = len(living) * state.config.discussion_rounds
    if state.discussion_turn_index >= total_turns:
        return None
    return living[state.discussion_turn_index % len(living)]


def _player_label(state: GameState, seat: int) -> str:
    """Public label for a seat index — must match policies.rendering / observation (player_labels[i])."""
    labels = state.player_labels
    if labels and 0 <= seat < len(labels):
        return labels[seat]
    if 0 <= seat < 26:
        return f"Player {chr(ord('A') + seat)}"
    return f"Player {seat + 1}"


def _initials_from_label(label: str) -> str:
    parts = label.split()
    if not parts:
        return "?"
    token = parts[-1]
    return token[-1].upper() if token else label[:1].upper()


def _seat_initials(state: GameState, seat: int) -> str:
    return _initials_from_label(_player_label(state, seat))


def _viewer_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mafia Training Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg-top: #1d120b;
      --bg-bottom: #080605;
      --panel: rgba(19, 15, 12, 0.84);
      --panel-border: rgba(255, 228, 196, 0.12);
      --gold: #efc784;
      --wood-1: #5d331e;
      --wood-2: #2f170d;
      --muted: #d5c2b1;
      --danger: #ff7a6b;
      --town: #77d8c6;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top, rgba(238, 184, 120, 0.18), transparent 32%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      color: #f4ede6;
    }
    .page {
      display: grid;
      grid-template-columns: minmax(320px, 1.6fr) minmax(280px, 0.9fr);
      gap: 24px;
      padding: 24px;
      min-height: 100vh;
    }
    .stage, .sidebar {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      backdrop-filter: blur(16px);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 24px 0;
    }
    .kicker {
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 0.72rem;
      color: var(--gold);
      margin-bottom: 8px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.9rem, 3vw, 2.8rem);
      line-height: 1.05;
      font-weight: 600;
    }
    .subtitle, .status {
      color: var(--muted);
      font-size: 0.96rem;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 18px 24px 10px;
      flex-wrap: wrap;
    }
    .pill {
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.85rem;
      background: rgba(255,255,255,0.04);
    }
    .table-wrap {
      position: relative;
      height: 720px;
      margin: 8px 18px 18px;
      border-radius: 28px;
      background:
        radial-gradient(circle at center, rgba(244, 197, 136, 0.12), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.12));
      overflow: hidden;
    }
    .table-wrap::before {
      content: "";
      position: absolute;
      inset: 7% 12%;
      border-radius: 50%;
      background:
        radial-gradient(circle at 35% 28%, rgba(255,255,255,0.12), transparent 20%),
        radial-gradient(circle at 50% 50%, var(--wood-1), var(--wood-2));
      box-shadow:
        inset 0 0 0 14px rgba(120, 75, 43, 0.45),
        inset 0 12px 40px rgba(255,255,255,0.08),
        0 24px 70px rgba(0,0,0,0.42);
    }
    .table-wrap.night::after {
      content: "";
      position: absolute;
      inset: 0;
      background: rgba(13, 18, 34, 0.34);
      pointer-events: none;
    }
    .seat {
      position: absolute;
      width: 180px;
      transform: translate(-50%, -50%);
      text-align: center;
    }
    .seat.s0 { left: 50%; top: 15%; }
    .seat.s1 { left: 79%; top: 32%; }
    .seat.s2 { left: 69%; top: 78%; }
    .seat.s3 { left: 31%; top: 78%; }
    .seat.s4 { left: 21%; top: 32%; }
    .seat.s5 { left: 50%; top: 88%; }
    .portrait {
      width: 96px;
      height: 96px;
      margin: 0 auto 10px;
      border-radius: 50%;
      position: relative;
      background:
        radial-gradient(circle at 35% 30%, rgba(255,255,255,0.6), transparent 18%),
        linear-gradient(180deg, #8f5f49, #513428);
      box-shadow: 0 10px 24px rgba(0,0,0,0.36);
      border: 3px solid rgba(255,255,255,0.1);
    }
    .portrait::before {
      content: attr(data-initials);
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      font-size: 1.4rem;
      font-weight: 700;
      color: rgba(255,255,255,0.92);
    }
    .seat.dead .portrait {
      filter: grayscale(1) brightness(0.65);
    }
    .seat.trainable .portrait {
      border-color: rgba(119, 216, 198, 0.62);
      box-shadow: 0 0 0 8px rgba(119, 216, 198, 0.08), 0 10px 24px rgba(0,0,0,0.36);
    }
    .seat.turn-highlight .portrait {
      outline: 3px solid rgba(239, 199, 132, 0.95);
      outline-offset: 5px;
      border-color: rgba(255, 228, 196, 0.75);
      box-shadow:
        0 0 0 6px rgba(239, 199, 132, 0.2),
        0 0 32px rgba(239, 199, 132, 0.4),
        0 10px 24px rgba(0,0,0,0.36);
      animation: turn-ring-pulse 2s ease-in-out infinite;
    }
    .seat.turn-highlight.dead .portrait {
      outline-color: rgba(180, 160, 140, 0.75);
      animation: none;
    }
    @keyframes turn-ring-pulse {
      0%, 100% {
        outline-color: rgba(239, 199, 132, 0.75);
        box-shadow:
          0 0 0 5px rgba(239, 199, 132, 0.15),
          0 0 24px rgba(239, 199, 132, 0.35),
          0 10px 24px rgba(0,0,0,0.36);
      }
      50% {
        outline-color: rgba(255, 236, 210, 1);
        box-shadow:
          0 0 0 8px rgba(239, 199, 132, 0.28),
          0 0 36px rgba(239, 199, 132, 0.5),
          0 10px 24px rgba(0,0,0,0.36);
      }
    }
    .name {
      font-size: 1rem;
      font-weight: 600;
    }
    .meta {
      font-size: 0.82rem;
      color: var(--muted);
      min-height: 1.3em;
    }
    .vote-tag, .role-tag {
      display: inline-flex;
      margin-top: 8px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.74rem;
      border: 1px solid rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.06);
    }
    .role-tag { color: var(--gold); }
    .narrator {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: min(560px, calc(100% - 48px));
      padding: 22px 26px;
      text-align: center;
      font-size: 1.08rem;
      line-height: 1.45;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(13, 12, 16, 0.72);
      backdrop-filter: blur(12px);
      color: #f4ede6;
      z-index: 2;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
    }
    .sidebar-section {
      padding: 18px 20px;
      border-top: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-section:first-child {
      border-top: 0;
    }
    .episode-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    button.episode {
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      color: inherit;
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
    }
    button.episode.active {
      border-color: rgba(119, 216, 198, 0.62);
      background: rgba(119, 216, 198, 0.12);
    }
    .transcript {
      display: grid;
      gap: 10px;
      max-height: 360px;
      overflow: auto;
      align-content: start;
    }
    .line {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.05);
    }
    .line strong {
      color: #fff6e8;
    }
    .line.event-phase_change,
    .line.event-night_outcome,
    .line.event-night_kill_choice,
    .line.event-doctor_save_choice,
    .line.event-detective_investigation,
    .line.event-elimination,
    .line.event-episode_complete {
      background: rgba(239, 199, 132, 0.1);
      border-color: rgba(239, 199, 132, 0.18);
    }
    .line-kind {
      display: block;
      margin-bottom: 4px;
      color: var(--gold);
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    input[type="range"] {
      width: 160px;
    }
    @media (max-width: 1080px) {
      .page { grid-template-columns: 1fr; }
      .table-wrap { height: 620px; }
    }
    @media (max-width: 720px) {
      .table-wrap { height: 560px; }
      .seat { width: 132px; }
      .portrait { width: 76px; height: 76px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="stage">
      <div class="header">
        <div>
          <div class="kicker">Live Mafia Training Viewer</div>
          <h1 id="headline">Waiting for the next game</h1>
          <div class="subtitle" id="subheadline">Start `scripts/train_local.py --ui` to stream games into this room.</div>
        </div>
        <div class="status" id="connection-status">Connecting…</div>
      </div>
      <div class="controls">
        <div class="pill" id="run-meta">No episode selected</div>
        <label class="pill">Playback <input id="speed" type="range" min="0.35" max="4" step="0.05" value="1"></label>
        <div class="pill"><span id="speed-value">1.00x</span></div>
      </div>
      <div class="table-wrap" id="table-wrap">
        <div class="narrator" id="narrator">No live events yet.</div>
      </div>
    </section>
    <aside class="sidebar">
      <div class="sidebar-section">
        <div class="kicker">Episodes</div>
        <div class="subtitle">Switch between featured games from the current training run.</div>
        <div class="episode-list" id="episode-list"></div>
      </div>
      <div class="sidebar-section">
        <div class="kicker">Latest Event</div>
        <div id="latest-event" class="subtitle">No events yet.</div>
      </div>
      <div class="sidebar-section">
        <div class="kicker">Transcript</div>
        <div class="transcript" id="transcript"></div>
      </div>
    </aside>
  </div>
  <script>
    const episodes = new Map();
    let activeEpisodeId = null;
    let playbackSpeed = 1;
    let nextRenderAt = 0;

    const headline = document.getElementById("headline");
    const subheadline = document.getElementById("subheadline");
    const connectionStatus = document.getElementById("connection-status");
    const tableWrap = document.getElementById("table-wrap");
    const narrator = document.getElementById("narrator");
    const runMeta = document.getElementById("run-meta");
    const transcript = document.getElementById("transcript");
    const latestEvent = document.getElementById("latest-event");
    const episodeList = document.getElementById("episode-list");
    const speedInput = document.getElementById("speed");
    const speedValue = document.getElementById("speed-value");

    speedInput.addEventListener("input", () => {
      playbackSpeed = Number(speedInput.value);
      speedValue.textContent = playbackSpeed.toFixed(2) + "x";
    });
    speedValue.textContent = playbackSpeed.toFixed(2) + "x";

    function ensureEpisode(event) {
      if (!episodes.has(event.episode_id)) {
        episodes.set(event.episode_id, {
          id: event.episode_id,
          queue: [],
          current: event,
          label: "Episode " + (episodes.size + 1),
          history: [],
          seenEventIds: new Set(),
        });
        if (activeEpisodeId === null) {
          activeEpisodeId = event.episode_id;
        }
        renderEpisodeButtons();
      }
      return episodes.get(event.episode_id);
    }

    function renderEpisodeButtons() {
      episodeList.innerHTML = "";
      for (const episode of episodes.values()) {
        const button = document.createElement("button");
        button.className = "episode" + (episode.id === activeEpisodeId ? " active" : "");
        button.textContent = episode.label;
        button.onclick = () => {
          activeEpisodeId = episode.id;
          renderEpisodeButtons();
          render();
        };
        episodeList.appendChild(button);
      }
    }

    function connect() {
      const stream = new EventSource("/events");
      stream.onopen = () => {
        connectionStatus.textContent = "Connected";
      };
      stream.onerror = () => {
        connectionStatus.textContent = "Reconnecting…";
      };
      stream.addEventListener("viewer", (raw) => {
        const event = JSON.parse(raw.data);
        const episode = ensureEpisode(event);
        if (episode.seenEventIds.has(event.event_id)) {
          return;
        }
        episode.seenEventIds.add(event.event_id);
        appendTranscriptEntry(episode, event);
        episode.queue.push(event);
      });
    }

    function defaultPlayerLabel(seat) {
      if (typeof seat !== "number" || seat < 0) return "";
      if (seat < 26) return "Player " + String.fromCharCode(65 + seat);
      return "Player " + (seat + 1);
    }

    function seatName(snapshot, seat) {
      if (seat === null || seat === undefined) {
        return "";
      }
      const player = snapshot.players.find((candidate) => candidate.seat === seat);
      return player ? player.name : defaultPlayerLabel(seat);
    }

    function labelForKind(kind, event) {
      if (kind === "speech" && event.actor !== null && event.actor !== undefined) {
        return seatName(event.snapshot, event.actor);
      }
      if (kind === "phase_change") return "Phase";
      if (kind === "night_kill_choice") return "Night";
      if (kind === "doctor_save_choice") return "Doctor";
      if (kind === "detective_investigation") return "Detective";
      if (kind === "night_outcome") return "Dawn";
      if (kind === "elimination") return "Elimination";
      if (kind === "speech") return "Speech";
      if (kind === "vote") return "Vote";
      if (kind === "episode_complete") return "Result";
      if (kind === "episode_start") return "Start";
      return "Event";
    }

    function transcriptText(event) {
      if (event.kind === "speech" && event.message) {
        return event.message;
      }
      if (event.message) {
        return event.message;
      }
      if (event.kind === "speech" && event.actor !== null && event.actor !== undefined) {
        return seatName(event.snapshot, event.actor) + " speaks.";
      }
      return event.snapshot.narrator || "No event details.";
    }

    function appendTranscriptEntry(episode, event) {
      episode.history.push({
        id: event.event_id,
        kind: event.kind,
        label: labelForKind(event.kind, event),
        text: transcriptText(event),
      });
    }

    function tick(now) {
      if (now >= nextRenderAt) {
        for (const episode of episodes.values()) {
          if (episode.queue.length > 0) {
            episode.current = episode.queue.shift();
          }
        }
        render();
        const frameDelay = 900 / playbackSpeed;
        nextRenderAt = now + frameDelay;
      }
      requestAnimationFrame(tick);
    }

    function render() {
      if (!activeEpisodeId || !episodes.has(activeEpisodeId)) {
        return;
      }
      const episode = episodes.get(activeEpisodeId);
      const event = episode.current;
      if (!event) {
        return;
      }
      const snapshot = event.snapshot;
      headline.textContent = snapshot.headline;
      subheadline.textContent = snapshot.subheadline;
      runMeta.textContent = episode.label + " • " + event.kind.replaceAll("_", " ");
      latestEvent.textContent = event.message || snapshot.narrator || "No event details.";

      tableWrap.classList.toggle("night", String(snapshot.phase).startsWith("night"));
      tableWrap.querySelectorAll(".seat").forEach((node) => node.remove());

      snapshot.players.forEach((player) => {
        const seat = document.createElement("div");
        seat.className =
          "seat s" +
          player.seat +
          (player.alive ? "" : " dead") +
          (player.is_trainable ? " trainable" : "") +
          (player.is_turn_highlight ? " turn-highlight" : "");

        const portrait = document.createElement("div");
        portrait.className = "portrait";
        portrait.dataset.initials = player.initials;

        const name = document.createElement("div");
        name.className = "name";
        name.textContent = player.name;

        const meta = document.createElement("div");
        meta.className = "meta";
        if (!player.alive) {
          meta.textContent = "Out of the game";
        } else if (player.is_current_speaker) {
          meta.textContent = "Speaking";
        } else if (player.vote_target !== null && player.vote_target !== undefined) {
          const votedFor = snapshot.players.find((candidate) => candidate.seat === player.vote_target);
          meta.textContent = "Voting " + (votedFor ? votedFor.name : ("Seat " + (player.vote_target + 1)));
        } else {
          meta.textContent = "Reading the table";
        }

        seat.appendChild(portrait);
        seat.appendChild(name);
        seat.appendChild(meta);

        if (player.revealed_role) {
          const role = document.createElement("div");
          role.className = "role-tag";
          role.textContent = player.revealed_role;
          seat.appendChild(role);
        }

        tableWrap.appendChild(seat);
      });

      narrator.textContent = snapshot.narrator || event.message || "The room is quiet for a moment.";

      transcript.innerHTML = "";
      episode.history.forEach((line) => {
        const row = document.createElement("div");
        row.className = "line event-" + line.kind;
        const kind = document.createElement("span");
        kind.className = "line-kind";
        kind.textContent = line.label;
        const detail = document.createElement("span");
        detail.textContent = line.text;
        row.appendChild(kind);
        row.appendChild(detail);
        transcript.appendChild(row);
      });
    }

    connect();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""
