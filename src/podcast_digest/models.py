"""Core data types shared across the pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class TranscriptLink:
    """A <podcast:transcript> entry from an RSS item."""

    url: str
    type: str  # MIME type, e.g. text/vtt, application/json, application/x-subrip
    language: str | None = None


@dataclass
class Episode:
    title: str
    podcast: str
    audio_url: str | None = None
    guid: str | None = None
    feed_url: str | None = None
    link: str | None = None
    published: datetime | None = None
    duration_seconds: float | None = None
    description: str | None = None
    image: str | None = None
    transcripts: list[TranscriptLink] = field(default_factory=list)
    local_audio: str | None = None  # path to a local file, if given directly

    @property
    def id(self) -> str:
        key = self.guid or self.audio_url or self.local_audio or f"{self.podcast}|{self.title}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    segments: list[Segment]
    source: str  # mlx-whisper | faster-whisper | deepgram | published
    model: str | None = None
    language: str | None = None
    timed: bool = True
    cost_usd: float = 0.0

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "language": self.language,
            "timed": self.timed,
            "cost_usd": self.cost_usd,
            "segments": [s.__dict__ for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            segments=[Segment(**s) for s in d["segments"]],
            source=d["source"],
            model=d.get("model"),
            language=d.get("language"),
            timed=d.get("timed", True),
            cost_usd=d.get("cost_usd", 0.0),
        )


def fmt_ts(seconds: float) -> str:
    """12.3 -> '00:12', 3723 -> '1:02:03'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def safe_filename(name: str, max_len: int = 120) -> str:
    """Strip characters that break file systems or Obsidian links."""
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].rstrip(" .") or "Untitled"


def episode_date(ep: Episode) -> str:
    """Publish date, or today for local files that have none."""
    return (ep.published or datetime.now(UTC)).date().isoformat()


def storage_name(ep: Episode) -> tuple[str, str]:
    """(podcast folder, file stem) for data files: 'Show', '2026-09-17 Title - <id>'.

    The id suffix keeps names unique and lets lookups find a file even if the title changes.
    """
    return safe_filename(ep.podcast), f"{episode_date(ep)} {safe_filename(ep.title, 80)} - {ep.id}"
