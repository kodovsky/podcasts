"""Get a timestamped transcript: cache -> published transcript -> local Whisper -> Deepgram."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from ..config import Config
from ..db import DB
from ..http import download
from ..models import Episode, Transcript
from .engines import (
    EngineUnavailable,
    auto_engine,
    transcribe_deepgram,
    transcribe_faster_whisper,
    transcribe_mlx,
)
from .published import dumps, fetch_published

log = logging.getLogger(__name__)

__all__ = ["get_transcript", "EngineUnavailable"]


def _audio_path(ep: Episode, cfg: Config) -> Path:
    if ep.local_audio:
        return Path(ep.local_audio)
    if not ep.audio_url:
        raise RuntimeError(f"Episode {ep.title!r} has no audio URL")
    ext = Path(urlparse(ep.audio_url).path).suffix.lower() or ".mp3"
    if len(ext) > 6:
        ext = ".mp3"
    return download(ep.audio_url, cfg.audio_dir / f"{ep.id}{ext}")


def _run_engine(engine: str, ep: Episode, cfg: Config) -> Transcript:
    tc = cfg.transcription
    if engine == "deepgram":
        # Remote audio: Deepgram fetches it directly, no local download needed.
        if ep.audio_url and not ep.local_audio:
            return transcribe_deepgram(None, tc, audio_url=ep.audio_url)
        return transcribe_deepgram(_audio_path(ep, cfg), tc)
    audio = _audio_path(ep, cfg)
    if engine == "mlx-whisper":
        return transcribe_mlx(audio, tc)
    if engine == "faster-whisper":
        return transcribe_faster_whisper(audio, tc)
    raise ValueError(f"Unknown engine {engine}")


def get_transcript(ep: Episode, cfg: Config, db: DB, engine: str | None = None) -> Transcript:
    cache = cfg.transcripts_dir / f"{ep.id}.json"
    if cache.exists():
        log.info("Using cached transcript %s", cache)
        return Transcript.from_dict(json.loads(cache.read_text()))

    tc = cfg.transcription
    transcript: Transcript | None = None
    if tc.prefer_published and ep.transcripts and engine is None:
        try:
            transcript = fetch_published(ep.transcripts, tc.allow_untimed_published)
        except Exception as e:
            log.warning("Published transcript failed (%s); transcribing instead", e)

    if transcript is None:
        primary = engine or (auto_engine() if tc.engine == "auto" else tc.engine)
        try:
            transcript = _run_engine(primary, ep, cfg)
        except Exception as e:
            fb = tc.fallback
            if engine or not fb or fb == primary:
                raise
            log.warning("%s failed (%s: %s); falling back to %s", primary, type(e).__name__, e, fb)
            transcript = _run_engine(fb, ep, cfg)

    if not transcript.segments:
        raise RuntimeError("Transcription produced no text")
    if transcript.cost_usd:
        db.log_cost(
            episode_id=ep.id,
            stage="transcribe",
            provider=transcript.source,
            model=transcript.model,
            cost_usd=transcript.cost_usd,
            audio_seconds=transcript.duration,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(dumps(transcript))
    return transcript
