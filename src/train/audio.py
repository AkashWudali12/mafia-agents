from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from train.viewer import ViewerEvent, ViewerEventSink

DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
_NARRATION_EVENT_KINDS = frozenset({"episode_start", "phase_change", "night_outcome", "elimination", "episode_complete"})
_SUPPORTED_AUDIO_PLAYERS = ("afplay", "ffplay", "mpv", "mpg123", "vlc")


def parse_voice_ids(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def expand_voice_ids(voice_ids: Sequence[str], *, player_count: int) -> tuple[str, ...]:
    if player_count < 0:
        raise ValueError("player_count must be non-negative")
    if player_count == 0:
        return ()
    if not voice_ids:
        raise ValueError("at least one ElevenLabs player voice id is required when TTS is enabled")
    return tuple(voice_ids[index % len(voice_ids)] for index in range(player_count))


def build_elevenlabs_event_sink(
    *,
    player_count: int,
    model_id: str | None = None,
    output_format: str | None = None,
    narrator_voice_id: str | None = None,
    cli_player_voice_ids: Sequence[str] | None = None,
) -> ElevenLabsEventSink:
    """Create a sink that calls the **ElevenLabs HTTP API** for synthesis (no local TTS model).

    ``LocalAudioPlayer`` only writes the returned bytes to a temp file and invokes a system player
    (e.g. ``afplay``). Requires ``ELEVENLABS_API_KEY`` and at least one player voice id from CLI
    or ``ELEVENLABS_PLAYER_VOICE_IDS``.

    ``narrator_voice_id`` should already be resolved (CLI + env); ``None`` means no narrator voice.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is required for ElevenLabs TTS "
            "(speech is generated on ElevenLabs servers, not locally)."
        )
    cli = tuple(cli_player_voice_ids or ())
    if cli:
        player_voice_ids = expand_voice_ids(cli, player_count=player_count)
    else:
        env_voice_ids = parse_voice_ids(os.getenv("ELEVENLABS_PLAYER_VOICE_IDS"))
        player_voice_ids = expand_voice_ids(env_voice_ids, player_count=player_count)
    narr = narrator_voice_id or None
    mid = model_id or os.getenv("ELEVENLABS_MODEL_ID") or DEFAULT_ELEVENLABS_MODEL_ID
    fmt = output_format or os.getenv("ELEVENLABS_OUTPUT_FORMAT") or DEFAULT_ELEVENLABS_OUTPUT_FORMAT
    return ElevenLabsEventSink(
        api_key=api_key,
        player_voice_ids=player_voice_ids,
        narrator_voice_id=narr,
        model_id=mid,
        output_format=fmt,
    )


class CompositeViewerEventSink(ViewerEventSink):
    def __init__(self, *sinks: ViewerEventSink) -> None:
        self._sinks = tuple(sink for sink in sinks if sink is not None)

    def publish(self, event: ViewerEvent) -> None:
        for sink in self._sinks:
            sink.publish(event)


class ElevenLabsSpeechClient:
    """Thin wrapper around the official ElevenLabs SDK (remote API — not a local ML model)."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = DEFAULT_ELEVENLABS_MODEL_ID,
        output_format: str = DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    ) -> None:
        from elevenlabs.client import ElevenLabs

        self._client = ElevenLabs(api_key=api_key)
        self._model_id = model_id
        self._output_format = output_format

    def synthesize(self, *, voice_id: str, text: str) -> bytes:
        chunks = self._client.text_to_speech.convert(
            voice_id,
            text=text,
            model_id=self._model_id,
            output_format=self._output_format,
        )
        return b"".join(chunks)


class LocalAudioPlayer:
    """Plays MP3 (or other) bytes via a **local system binary** — does not perform speech synthesis."""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable or os.getenv("ELEVENLABS_AUDIO_PLAYER") or discover_audio_player()
        if not self._executable:
            supported = ", ".join(_SUPPORTED_AUDIO_PLAYERS)
            raise RuntimeError(
                "no supported local audio player found; install one of "
                f"{supported} or set ELEVENLABS_AUDIO_PLAYER"
            )

    def play(self, audio_bytes: bytes, *, suffix: str = ".mp3") -> None:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            path = Path(handle.name)

        try:
            subprocess.run(
                self._command_for(path),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            path.unlink(missing_ok=True)

    def _command_for(self, path: Path) -> list[str]:
        if self._executable == "afplay":
            return ["afplay", str(path)]
        if self._executable == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        if self._executable == "mpv":
            return ["mpv", "--really-quiet", str(path)]
        if self._executable == "mpg123":
            return ["mpg123", "-q", str(path)]
        if self._executable == "vlc":
            return ["vlc", "--intf", "dummy", "--play-and-exit", str(path)]
        return [self._executable, str(path)]


def discover_audio_player() -> str | None:
    for executable in _SUPPORTED_AUDIO_PLAYERS:
        if shutil.which(executable):
            return executable
    return None


class ElevenLabsEventSink(ViewerEventSink):
    def __init__(
        self,
        *,
        api_key: str,
        player_voice_ids: Sequence[str],
        narrator_voice_id: str | None = None,
        model_id: str = DEFAULT_ELEVENLABS_MODEL_ID,
        output_format: str = DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
        speech_client: ElevenLabsSpeechClient | None = None,
        audio_player: LocalAudioPlayer | None = None,
    ) -> None:
        self._player_voice_ids = tuple(player_voice_ids)
        self._narrator_voice_id = narrator_voice_id
        self._speech_client = speech_client or ElevenLabsSpeechClient(
            api_key=api_key,
            model_id=model_id,
            output_format=output_format,
        )
        self._audio_player = audio_player or LocalAudioPlayer()

    def publish(self, event: ViewerEvent) -> None:
        text, voice_id = self._resolve_request(event)
        if not text or not voice_id:
            return

        try:
            audio_bytes = self._speech_client.synthesize(voice_id=voice_id, text=text)
            if audio_bytes:
                self._audio_player.play(audio_bytes)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            print(
                f"ElevenLabs TTS error ({event.kind}): {exc}",
                file=sys.stderr,
            )

    def _resolve_request(self, event: ViewerEvent) -> tuple[str | None, str | None]:
        if event.kind == "speech" and event.actor is not None and event.message:
            if event.actor >= len(self._player_voice_ids):
                return None, None
            return event.message, self._player_voice_ids[event.actor]
        if self._narrator_voice_id and event.kind in _NARRATION_EVENT_KINDS and event.message:
            return event.message, self._narrator_voice_id
        return None, None
