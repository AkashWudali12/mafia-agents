from __future__ import annotations

from urllib.request import urlopen

from contracts import Alignment, EnvironmentConfig, Role
from game import new_game
from train import build_scripted_policy_map, run_episode
from train.trajectory import EpisodeMetadata
from train.viewer import LiveTrainingViewer, ViewerEvent, ViewerEventSink, build_snapshot


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
    assert all(
        event.snapshot.active_bubble is None
        and event.snapshot.narrator
        and event.message
        and event.message in event.snapshot.narrator
        for event in speech_events
    )
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


def test_viewer_snapshot_names_match_engine_player_labels_when_shuffled() -> None:
    """Viewer must use state.player_labels[i], not seat index as Player A/B/…."""
    config = EnvironmentConfig(shuffle_player_labels_each_game=True, shuffle_roles_each_game=False)
    state = new_game(config=config, seed=42)
    assert len(set(state.player_labels)) == len(state.player_labels)
    meta = EpisodeMetadata(
        episode_id="label-check",
        seed=42,
        trainable_seat=0,
        trainable_role=Role.MAFIA,
        trainable_alignment=Alignment.MAFIA,
        environment_config_hash="test",
    )
    snap = build_snapshot(state=state, metadata=meta, update_index=0)
    for p in snap.players:
        assert p.name == state.player_labels[p.seat]
        assert p.initials == p.name.split()[-1][-1].upper()
