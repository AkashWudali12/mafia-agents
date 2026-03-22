from __future__ import annotations

import pytest

from train.audio import (
    CompositeViewerEventSink,
    ElevenLabsEventSink,
    build_elevenlabs_event_sink,
    expand_voice_ids,
    parse_voice_ids,
)
from train.viewer import ViewerEvent, ViewerEventSink, ViewerSnapshot


class _CollectingSink(ViewerEventSink):
    def __init__(self) -> None:
        self.events: list[ViewerEvent] = []

    def publish(self, event: ViewerEvent) -> None:
        self.events.append(event)


class _FakeSpeechClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    def synthesize(self, *, voice_id: str, text: str) -> bytes:
        self.requests.append((voice_id, text))
        return f"{voice_id}:{text}".encode("utf-8")


class _FakeAudioPlayer:
    def __init__(self) -> None:
        self.audio: list[bytes] = []

    def play(self, audio_bytes: bytes, *, suffix: str = ".mp3") -> None:
        self.audio.append(audio_bytes)


def _event(*, kind: str, actor: int | None = None, message: str | None = None) -> ViewerEvent:
    snapshot = ViewerSnapshot(
        episode_id="episode-1",
        update_index=0,
        day=1,
        phase="day_discussion",
        headline="Day 1",
        subheadline="Discussion",
        players=(),
    )
    return ViewerEvent(
        event_id=f"event-{kind}",
        kind=kind,
        episode_id="episode-1",
        update_index=0,
        day=1,
        phase="day_discussion",
        actor=actor,
        message=message,
        snapshot=snapshot,
    )


def test_composite_viewer_event_sink_publishes_to_all_sinks() -> None:
    first = _CollectingSink()
    second = _CollectingSink()
    event = _event(kind="speech", actor=0, message="I don't trust Player C.")

    sink = CompositeViewerEventSink(first, second)
    sink.publish(event)

    assert first.events == [event]
    assert second.events == [event]


def test_elevenlabs_event_sink_voices_player_speech() -> None:
    speech_client = _FakeSpeechClient()
    audio_player = _FakeAudioPlayer()
    sink = ElevenLabsEventSink(
        api_key="test",
        player_voice_ids=("voice-a", "voice-b"),
        speech_client=speech_client,
        audio_player=audio_player,
    )

    sink.publish(_event(kind="speech", actor=1, message="Player A feels off to me."))

    assert speech_client.requests == [("voice-b", "Player A feels off to me.")]
    assert audio_player.audio == [b"voice-b:Player A feels off to me."]


def test_elevenlabs_event_sink_voices_narration_only_when_narrator_voice_is_set() -> None:
    speech_client = _FakeSpeechClient()
    audio_player = _FakeAudioPlayer()
    sink = ElevenLabsEventSink(
        api_key="test",
        player_voice_ids=("voice-a",),
        narrator_voice_id="narrator-voice",
        speech_client=speech_client,
        audio_player=audio_player,
    )

    sink.publish(_event(kind="phase_change", message="Night falls. The mafia acts in the dark."))

    assert speech_client.requests == [("narrator-voice", "Night falls. The mafia acts in the dark.")]
    assert audio_player.audio == [b"narrator-voice:Night falls. The mafia acts in the dark."]


def test_voice_id_helpers_parse_and_expand_lists() -> None:
    assert parse_voice_ids(" voice-a,voice-b ,, voice-c ") == ("voice-a", "voice-b", "voice-c")
    assert expand_voice_ids(("voice-a", "voice-b"), player_count=5) == (
        "voice-a",
        "voice-b",
        "voice-a",
        "voice-b",
        "voice-a",
    )


def test_build_elevenlabs_event_sink_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        build_elevenlabs_event_sink(
            player_count=2,
            cli_player_voice_ids=("v1", "v2"),
            narrator_voice_id=None,
        )


def test_build_elevenlabs_event_sink_expands_cli_voices(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    sink = build_elevenlabs_event_sink(
        player_count=3,
        cli_player_voice_ids=("a", "b"),
        narrator_voice_id="narr",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    assert sink._player_voice_ids == ("a", "b", "a")
    assert sink._narrator_voice_id == "narr"
