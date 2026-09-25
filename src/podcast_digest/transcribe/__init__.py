"""Get a timestamped transcript: cache -> published transcript -> local Whisper -> Deepgram."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from ..config import Config
from ..db import DB
from ..http import download
from ..models import Episode, Transcript, storage_name
from .engines import (
    EngineUnavailable,
    auto_engine,
    transcribe_deepgram,
    transcribe_faster_whisper,
    transcribe_mlx,
)
from .published import dumps, fetch_published

log = logging.getLogger(__name__)

__all__ = ["get_transcript", "transcript_path", "EngineUnavailable"]


def _locate(root: Path, ep: Episode, suffix: str) -> Path:
    """Where an episode's data file lives: <root>/<Podcast>/<date> <title> - <id><suffix>.

    An existing file for this episode id (e.g. from an older flat layout, or saved under a
    different title) is moved to the current name so the folder stays tidy.
    """
    folder, stem = storage_name(ep)
    target = root / folder / f"{stem}{suffix}"
    if not target.exists() and root.exists():
        pattern = f"*{ep.id}{suffix}" if suffix != ".*" else f"*{ep.id}.*"
        for found in root.rglob(pattern):
            if found.is_file() and not found.name.endswith(".part"):
                if suffix == ".*":
                    target = target.with_name(f"{stem}{found.suffix}")
                if found != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    found.rename(target)
                    log.info("Moved %s -> %s", found.name, target.relative_to(root))
                return target
    return target


def _audio_path(ep: Episode, cfg: Config) -> Path:
    if ep.local_audio:
        return Path(ep.local_audio)
    if not ep.audio_url:
        raise RuntimeError(f"Episode {ep.title!r} has no audio URL")
    existing = _locate(cfg.audio_dir, ep, ".*")
    if existing.exists():
        return existing
    ext = Path(urlparse(ep.audio_url).path).suffix.lower() or ".mp3"
    if len(ext) > 6:
        ext = ".mp3"
    return download(ep.audio_url, existing.with_suffix(ext))


def _cleanup_audio(ep: Episode, cfg: Config) -> None:
    if cfg.transcription.keep_audio or ep.local_audio:
        return
    audio = _locate(cfg.audio_dir, ep, ".*")
    if audio.exists():
        audio.unlink()
        log.info("Deleted audio %s (keep_audio: false)", audio.name)


def _use_project_model_cache(cfg: Config) -> None:
    """Point Hugging Face downloads (Whisper weights) at data/models instead of ~/.cache.

    Must run before mlx_whisper / faster_whisper import huggingface_hub, which reads HF_HOME
    at import time; the engines import them lazily, so doing it here is early enough.
    """
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cfg.models_dir)


def _run_engine(engine: str, ep: Episode, cfg: Config) -> Transcript:
    tc = cfg.transcription
    _use_project_model_cache(cfg)
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


def transcript_path(ep: Episode, cfg: Config) -> Path:
    return _locate(cfg.transcripts_dir, ep, ".json")


def get_transcript(ep: Episode, cfg: Config, db: DB, engine: str | None = None) -> Transcript:
    cache = transcript_path(ep, cfg)
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
    _cleanup_audio(ep, cfg)
    return transcript
