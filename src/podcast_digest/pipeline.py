"""End-to-end processing of one episode, and of new episodes across configured feeds."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config, FeedConfig
from .db import DB
from .feeds import fetch_feed
from .models import Episode
from .notes import write_episode_notes
from .summarize import Summarizer
from .transcribe import get_transcript

log = logging.getLogger(__name__)


def process_episode(
    ep: Episode,
    cfg: Config,
    db: DB,
    *,
    force: bool = False,
    engine: str | None = None,
    summarizer: Summarizer | None = None,
) -> Path | None:
    if db.is_done(ep.id) and not force:
        row = db.get(ep.id)
        log.info("Already processed: %s -> %s (use --force to redo)", ep.title, row["note_path"])
        return Path(row["note_path"]) if row["note_path"] else None

    db.upsert(ep)
    log.info("▶ %s — %s", ep.podcast, ep.title)
    try:
        transcript = get_transcript(ep, cfg, db, engine=engine)
        db.set_status(ep.id, "transcribed", transcript_source=transcript.source)
        log.info(
            "Transcript: %s, %d segments, %.0f min",
            transcript.source,
            len(transcript.segments),
            transcript.duration / 60,
        )

        summarizer = summarizer or Summarizer(cfg, db)
        summary = summarizer.summarize(ep, transcript)
        total_cost = db.episode_cost(ep.id)
        path = write_episode_notes(
            ep, summary, transcript, cfg, model=cfg.summarization.model, cost_usd=total_cost
        )
        db.set_status(ep.id, "done", note_path=str(path), summary_json=summary.model_dump_json())
        log.info("✔ Wrote %s (episode cost $%.4f)", path, total_cost)
        return path
    except Exception as e:
        db.set_status(ep.id, "failed", error=f"{type(e).__name__}: {e}")
        log.exception("✘ Failed: %s", ep.title)
        raise


def new_episodes(feed: FeedConfig, db: DB) -> list[Episode]:
    """Episodes not seen before. On a feed's first run only the latest one counts as new."""
    _, episodes = fetch_feed(feed.url)
    known = db.known_ids(feed.url)
    first_run = not known
    fresh = []
    for ep in episodes:
        ep.podcast = feed.name or ep.podcast
        if ep.id in known:
            continue
        if feed.since and ep.published and ep.published.date() < feed.since:
            db.mark_seen(ep)
            continue
        fresh.append(ep)
    if first_run and not feed.since:
        for ep in fresh[1:]:
            db.mark_seen(ep)
        fresh = fresh[:1]
    return fresh[: feed.max_new_per_run]


def run_feeds(cfg: Config, db: DB) -> list[Path]:
    written: list[Path] = []
    for feed in cfg.feeds:
        try:
            eps = new_episodes(feed, db)
        except Exception:
            log.exception("Could not read feed %s", feed.name)
            continue
        log.info("%s: %d new episode(s)", feed.name, len(eps))
        for ep in eps:
            try:
                if p := process_episode(ep, cfg, db):
                    written.append(p)
            except Exception:
                continue  # already logged; move on to the next episode
    return written
