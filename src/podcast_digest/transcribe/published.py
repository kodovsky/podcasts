"""Parse transcripts published in podcast feeds (Podcasting 2.0 <podcast:transcript>)."""

from __future__ import annotations

import html
import json
import logging
import re

from ..http import get
from ..models import Segment, Transcript, TranscriptLink

log = logging.getLogger(__name__)

_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(rf"{_TS}\s*-->\s*{_TS}")

# Preference order: timed formats first.
TIMED_TYPES = ["application/json", "text/vtt", "application/x-subrip", "application/srt"]
UNTIMED_TYPES = ["text/html", "text/plain"]


def _secs(h: str | None, m: str, s: str, frac: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def parse_cues(text: str) -> list[Segment]:
    """Parse SRT or WebVTT cue text."""
    segments: list[Segment] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = _CUE.search(line)
            if not m:
                continue
            start = _secs(*m.groups()[:4])
            end = _secs(*m.groups()[4:])
            body = " ".join(lines[i + 1 :]).strip()
            speaker = None
            v = re.match(r"<v(?:\.[^ >]*)?\s+([^>]+)>", body)
            if v:
                speaker = v.group(1).strip()
            body = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
            if body:
                segments.append(Segment(start=start, end=end, text=body, speaker=speaker))
            break
    return segments


def parse_json_transcript(data: dict) -> list[Segment]:
    """Podcast namespace JSON: {"segments": [{"startTime", "endTime", "body", "speaker"}]}."""
    segments = []
    for s in data.get("segments", []):
        body = (s.get("body") or "").strip()
        if body:
            segments.append(
                Segment(
                    start=float(s.get("startTime", 0)),
                    end=float(s.get("endTime", s.get("startTime", 0))),
                    text=body,
                    speaker=s.get("speaker"),
                )
            )
    return merge_short_segments(segments)


def merge_short_segments(segments: list[Segment], max_seconds: float = 20.0) -> list[Segment]:
    """Word-level transcripts -> sentence-ish segments (same speaker, <= max_seconds)."""
    merged: list[Segment] = []
    for s in segments:
        last = merged[-1] if merged else None
        if (
            last
            and last.speaker == s.speaker
            and s.start - last.start < max_seconds
            and not last.text.endswith((".", "?", "!"))
        ):
            last.text = f"{last.text} {s.text}"
            last.end = s.end
        else:
            merged.append(Segment(s.start, s.end, s.text, s.speaker))
    return merged


def parse_untimed(text: str, is_html: bool) -> list[Segment]:
    if is_html:
        text = re.sub(r"(?i)<br\s*/?>|</p>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return [Segment(start=0.0, end=0.0, text=p) for p in paras]


def pick_transcript(links: list[TranscriptLink], allow_untimed: bool) -> TranscriptLink | None:
    order = TIMED_TYPES + (UNTIMED_TYPES if allow_untimed else [])
    by_type = {link.type.split(";")[0].strip().lower(): link for link in links}
    for t in order:
        if t in by_type:
            return by_type[t]
    return None


def fetch_published(links: list[TranscriptLink], allow_untimed: bool = False) -> Transcript | None:
    link = pick_transcript(links, allow_untimed)
    if not link:
        if links:
            log.info("Feed has transcripts (%s) but none with timestamps", [x.type for x in links])
        return None
    log.info("Using published transcript %s (%s)", link.url, link.type)
    resp = get(link.url)
    kind = link.type.split(";")[0].strip().lower()
    if kind == "application/json":
        segments = parse_json_transcript(resp.json())
    elif kind in TIMED_TYPES:
        segments = parse_cues(resp.text)
    else:
        segments = parse_untimed(resp.text, is_html=kind == "text/html")
    if not segments:
        log.warning("Published transcript was empty")
        return None
    return Transcript(
        segments=segments,
        source="published",
        model=kind,
        language=link.language,
        timed=kind in TIMED_TYPES,
    )


def dumps(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=1)
