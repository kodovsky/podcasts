import json

from podcast_digest.models import Segment, Transcript, TranscriptLink, fmt_ts
from podcast_digest.transcribe import published
from podcast_digest.transcribe.published import (
    merge_short_segments,
    parse_cues,
    parse_json_transcript,
    pick_transcript,
)

from .conftest import FIXTURES


def test_vtt():
    segs = parse_cues((FIXTURES / "sample.vtt").read_text())
    assert [(s.start, s.speaker, s.text) for s in segs] == [
        (1.0, "Tim Ferriss", "Welcome to the show."),
        (65.25, "Jane Doe", "Pick one niche & own it."),
        (3600.0, None, "Late line."),
    ]


def test_srt():
    segs = parse_cues((FIXTURES / "sample.srt").read_text())
    assert [(s.start, s.end, s.text) for s in segs] == [
        (1.0, 4.0, "Hello there."),
        (5.5, 7.0, "Second line continues here."),
    ]


def test_json_word_level_is_merged():
    data = {
        "segments": [
            {"startTime": 0.0, "endTime": 0.4, "body": "Hello", "speaker": "A"},
            {"startTime": 0.4, "endTime": 0.8, "body": "world.", "speaker": "A"},
            {"startTime": 1.0, "endTime": 1.5, "body": "Hi", "speaker": "B"},
        ]
    }
    segs = parse_json_transcript(data)
    assert [(s.speaker, s.text) for s in segs] == [("A", "Hello world."), ("B", "Hi")]


def test_merge_respects_window():
    segs = [Segment(i * 5.0, i * 5.0 + 5, "word") for i in range(10)]
    assert len(merge_short_segments(segs, max_seconds=20)) == 3


def test_pick_prefers_timed():
    links = [TranscriptLink("h", "text/html"), TranscriptLink("v", "text/vtt")]
    assert pick_transcript(links, allow_untimed=False).url == "v"
    assert pick_transcript(links[:1], allow_untimed=False) is None
    assert pick_transcript(links[:1], allow_untimed=True).url == "h"


def test_fetch_published(monkeypatch):
    class R:
        text = (FIXTURES / "sample.vtt").read_text()

    monkeypatch.setattr(published, "get", lambda url: R())
    t = published.fetch_published([TranscriptLink("u", "text/vtt; charset=utf-8", "en")])
    assert t.source == "published" and t.timed and len(t.segments) == 3


def test_transcript_roundtrip():
    t = Transcript([Segment(0, 1, "a", "X")], source="mlx-whisper", model="m", cost_usd=0.1)
    assert Transcript.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_fmt_ts():
    assert fmt_ts(12.9) == "00:12"
    assert fmt_ts(3723) == "1:02:03"
