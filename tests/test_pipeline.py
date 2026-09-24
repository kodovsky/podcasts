import json
from datetime import UTC, datetime

import yaml

from podcast_digest import pipeline
from podcast_digest.digest import build_digest, iso_week_bounds
from podcast_digest.models import Episode, Segment, Transcript
from podcast_digest.notes import safe_filename
from podcast_digest.summarize import Summarizer

from .test_summarize import make_summary


def ep():
    return Episode(
        title="#701: Jane Doe — How to Build a Consulting Practice",
        podcast="The Test Show",
        guid="guid-701",
        audio_url="https://cdn/701.mp3",
        link="https://example.com/701",
        published=datetime(2026, 9, 18, tzinfo=UTC),
        duration_seconds=5400,
    )


def seed_transcript(cfg, e):
    t = Transcript(
        [Segment(0, 30, "Welcome.", "Tim"), Segment(65, 70, "Own it.", "Jane")],
        source="mlx-whisper",
        model="whisper-large-v3-turbo",
    )
    cfg.transcripts_dir.mkdir(parents=True)
    (cfg.transcripts_dir / f"{e.id}.json").write_text(json.dumps(t.to_dict()))


def test_end_to_end_note_and_digest(cfg, db, fake_client):
    e = ep()
    seed_transcript(cfg, e)  # cached transcript -> no audio/whisper needed
    client = fake_client([make_summary()])
    path = pipeline.process_episode(e, cfg, db, summarizer=Summarizer(cfg, db, client=client))

    assert (
        path
        == cfg.output_dir
        / "The Test Show"
        / "2026-09-18 701 Jane Doe — How to Build a Consulting Practice.md"
    )
    text = path.read_text()
    fm = yaml.safe_load(text.split("---\n")[1])
    assert fm["podcast"] == "The Test Show"
    assert fm["published"] == "2026-09-18"
    assert fm["duration"] == "1:30:00"
    assert fm["cost_usd"] == 0.04
    assert "podcast/the-test-show" in fm["tags"] and "topic/security" in fm["tags"]
    assert "## Summary" in text and text.count("\n- Point") == 5
    assert "- [ ] `02:00` Write a teardown" in text
    assert "## Relevant to: Building a solo AI agent security consulting practice" in text
    assert "## Relevant to: Health" not in text  # empty groups are omitted
    assert "[[The E-Myth]] by Michael Gerber" in text
    assert "[[2026-09-18 701 Jane Doe — How to Build a Consulting Practice (transcript)]]" in text
    transcript_note = path.parent / "Transcripts" / (path.stem + " (transcript).md")
    assert "[01:05] Jane: Own it." in transcript_note.read_text()

    # Idempotent: second run does nothing (no Claude call).
    assert (
        pipeline.process_episode(e, cfg, db, summarizer=Summarizer(cfg, db, client=client)) == path
    )
    assert len(client.messages.calls) == 1

    digest = build_digest(cfg, db, use_llm=False)
    assert digest is not None
    assert "[[" + path.stem + "]]" in digest.read_text()


def test_failure_is_recorded(cfg, db, fake_client):
    e = ep()
    seed_transcript(cfg, e)

    class Boom(Exception):
        pass

    class Bad:
        class messages:  # noqa: N801
            @staticmethod
            def parse(**kw):
                raise Boom("api down")

    try:
        pipeline.process_episode(e, cfg, db, summarizer=Summarizer(cfg, db, client=Bad()))
    except Boom:
        pass
    row = db.get(e.id)
    assert row["status"] == "failed" and "api down" in row["error"]


def test_new_episodes_first_run_takes_latest(cfg, db, monkeypatch):
    from podcast_digest.config import FeedConfig
    from podcast_digest.feeds import parse_feed

    from .conftest import FIXTURES

    monkeypatch.setattr(
        pipeline, "fetch_feed", lambda url: parse_feed((FIXTURES / "feed.xml").read_bytes(), url)
    )
    feed = FeedConfig(name="Test", url="https://feed")
    first = pipeline.new_episodes(feed, db)
    assert [e.guid for e in first] == ["guid-701"]
    db.upsert(first[0])
    assert pipeline.new_episodes(feed, db) == []  # older one was marked seen


def test_iso_week():
    label, start, end = iso_week_bounds("2026-W39")
    assert label == "2026-W39" and start.date().isoformat() == "2026-09-21"
    assert (end - start).days == 7


def test_safe_filename():
    assert safe_filename('a/b: c? "d" [x] #1') == "a b c d x 1"
