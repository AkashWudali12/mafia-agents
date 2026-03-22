from __future__ import annotations

from urllib.request import urlopen

from game import new_game
from train import build_scripted_policy_map, run_episode
from train.viewer import LiveTrainingViewer, ViewerEvent, ViewerEventSink


class _CollectingSink(ViewerEventSink):
    def __init__(self) -> None:
        self.events: list[ViewerEvent] = []

    def publish(self, event: ViewerEvent) -> None:
        self.events.append(event)


def test_run_episode_emits_live_viewer_events() -> None:
    state = new_game(seed=19)
    sink = _CollectingSink()

    rollout = run_episode(
        trainable_seat=0,
        seat_policies=build_scripted_policy_map(state),
        seed=19,
        initial_state=state,
        update_index=2,
        viewer=sink,
    )

    assert sink.events
    assert sink.events[0].kind == "episode_start"
    kinds = {event.kind for event in sink.events}
    assert "speech" in kinds
    assert "vote" in kinds
    assert "phase_change" in kinds
    assert "night_kill_choice" in kinds
    assert "doctor_save_choice" in kinds
    assert "detective_investigation" in kinds
    assert "night_outcome" in kinds
    assert sink.events[-1].kind == "episode_complete"
    assert sink.events[-1].snapshot.winner == rollout.final_state.winner.value
    speech_events = [event for event in sink.events if event.kind == "speech"]
    assert speech_events
    assert all(event.snapshot.active_bubble is not None for event in speech_events)
    assert len(sink.events[-1].snapshot.transcript) == len(rollout.final_state.transcript)
    assert [row["message"] for row in sink.events[-1].snapshot.transcript] == [
        event.message for event in rollout.final_state.transcript
    ]
    night_events = [event for event in sink.events if event.kind in {"night_kill_choice", "doctor_save_choice", "detective_investigation"}]
    assert all(event.message for event in night_events)
    night_outcomes = [event.message for event in sink.events if event.kind == "night_outcome"]
    assert night_outcomes
    assert all(message.startswith("Dawn breaks.") for message in night_outcomes if message is not None)


def test_live_training_viewer_serves_html_page() -> None:
    viewer = LiveTrainingViewer(host="127.0.0.1", port=0, max_cached_events=10)
    url = viewer.start()
    try:
        with urlopen(url, timeout=2) as response:
            page = response.read().decode("utf-8")
    finally:
        viewer.stop()

    assert "Mafia Training Viewer" in page
    assert "Live Mafia Training Viewer" in page
    assert "appendTranscriptEntry" in page
