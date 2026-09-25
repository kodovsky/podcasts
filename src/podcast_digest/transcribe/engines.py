"""Speech-to-text engines. Heavy imports are lazy so only the chosen engine is required."""

from __future__ import annotations

import logging
import mimetypes
import os
import platform
import time
from pathlib import Path

import httpx

from ..config import TranscriptionConfig
from ..models import Segment, Transcript

log = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    pass


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def auto_engine() -> str:
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and _importable("mlx_whisper")
    ):
        return "mlx-whisper"
    if _importable("faster_whisper"):
        return "faster-whisper"
    if os.environ.get("DEEPGRAM_API_KEY"):
        return "deepgram"
    raise EngineUnavailable(
        "No transcription engine available. Install one: pip install -e '.[mac]' "
        "(Apple Silicon), pip install -e '.[local]', or set DEEPGRAM_API_KEY."
    )


def transcribe_mlx(audio: Path, cfg: TranscriptionConfig) -> Transcript:
    if not _importable("mlx_whisper"):
        raise EngineUnavailable("mlx-whisper not installed: pip install -e '.[mac]'")
    import mlx_whisper

    log.info("Transcribing with mlx-whisper (%s) on the Apple GPU…", cfg.mlx_model)
    t0 = time.time()
    # verbose=False shows a progress bar (None is fully silent, True prints every line).
    result = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=cfg.mlx_model, language=cfg.language, verbose=False
    )
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip())
        for s in result.get("segments", [])
        if s.get("text", "").strip()
    ]
    log.info("mlx-whisper done in %.0fs (%d segments)", time.time() - t0, len(segments))
    return Transcript(
        segments, source="mlx-whisper", model=cfg.mlx_model, language=result.get("language")
    )


def transcribe_faster_whisper(audio: Path, cfg: TranscriptionConfig) -> Transcript:
    if not _importable("faster_whisper"):
        raise EngineUnavailable("faster-whisper not installed: pip install -e '.[local]'")
    from faster_whisper import WhisperModel

    log.info("Transcribing with faster-whisper (%s)…", cfg.faster_whisper_model)
    t0 = time.time()
    model = WhisperModel(
        cfg.faster_whisper_model,
        device=cfg.faster_whisper_device,
        compute_type=cfg.faster_whisper_compute_type,
    )
    seg_iter, info = model.transcribe(str(audio), language=cfg.language, vad_filter=True)
    segments: list[Segment] = []
    next_report = 0.0
    for s in seg_iter:
        if s.text.strip():
            segments.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
        if info.duration and s.end >= next_report:
            log.info(
                "  %.0f%% (%.0f/%.0f min)",
                100 * s.end / info.duration,
                s.end / 60,
                info.duration / 60,
            )
            next_report += 600
    log.info("faster-whisper done in %.0fs (%d segments)", time.time() - t0, len(segments))
    return Transcript(
        segments, source="faster-whisper", model=cfg.faster_whisper_model, language=info.language
    )


def transcribe_deepgram(
    audio: Path | None, cfg: TranscriptionConfig, audio_url: str | None = None
) -> Transcript:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise EngineUnavailable("DEEPGRAM_API_KEY is not set")
    params = {
        "model": cfg.deepgram_model,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
        "diarize": "true",
    }
    if cfg.language:
        params["language"] = cfg.language
    else:
        params["detect_language"] = "true"
    headers = {"Authorization": f"Token {key}"}
    timeout = httpx.Timeout(60.0, read=1800.0, write=1800.0)
    log.info("Transcribing with Deepgram (%s)…", cfg.deepgram_model)
    with httpx.Client(timeout=timeout) as c:
        if audio_url:  # let Deepgram fetch it; avoids uploading hundreds of MB
            resp = c.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                json={"url": audio_url},
            )
        else:
            assert audio is not None
            ctype = mimetypes.guess_type(audio.name)[0] or "audio/mpeg"
            with audio.open("rb") as f:
                resp = c.post(
                    "https://api.deepgram.com/v1/listen",
                    params=params,
                    headers={**headers, "Content-Type": ctype},
                    content=f,
                )
    resp.raise_for_status()
    data = resp.json()
    segments = [
        Segment(
            start=float(u["start"]),
            end=float(u["end"]),
            text=u["transcript"].strip(),
            speaker=f"Speaker {u['speaker']}" if u.get("speaker") is not None else None,
        )
        for u in data.get("results", {}).get("utterances", [])
        if u.get("transcript", "").strip()
    ]
    duration = float(data.get("metadata", {}).get("duration", 0.0))
    cost = duration / 60 * cfg.deepgram_usd_per_minute
    channels = data.get("results", {}).get("channels") or [{}]
    language = channels[0].get("detected_language") or cfg.language
    log.info("Deepgram done: %.0f min audio, ~$%.3f", duration / 60, cost)
    return Transcript(
        segments, source="deepgram", model=cfg.deepgram_model, language=language, cost_usd=cost
    )
