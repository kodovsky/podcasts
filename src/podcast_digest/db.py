"""SQLite state: processed episodes and a cost ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import Episode

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    guid TEXT,
    feed_url TEXT,
    podcast TEXT NOT NULL,
    title TEXT NOT NULL,
    published TEXT,
    audio_url TEXT,
    link TEXT,
    status TEXT NOT NULL DEFAULT 'new',  -- new | transcribed | done | failed
    transcript_source TEXT,
    note_path TEXT,
    summary_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_feed ON episodes(feed_url);
CREATE INDEX IF NOT EXISTS idx_episodes_processed ON episodes(processed_at);

CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    stage TEXT NOT NULL,       -- transcribe | summarize | merge | digest
    provider TEXT NOT NULL,    -- anthropic | deepgram
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    audio_seconds REAL DEFAULT 0,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- episodes -------------------------------------------------------

    def get(self, episode_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()

    def is_done(self, episode_id: str) -> bool:
        row = self.get(episode_id)
        return row is not None and row["status"] == "done"

    def known_ids(self, feed_url: str) -> set[str]:
        rows = self.conn.execute("SELECT id FROM episodes WHERE feed_url = ?", (feed_url,))
        return {r["id"] for r in rows}

    def upsert(self, ep: Episode) -> None:
        self.conn.execute(
            """INSERT INTO episodes (id, guid, feed_url, podcast, title, published, audio_url,
                                     link, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 title = excluded.title, podcast = excluded.podcast,
                 audio_url = COALESCE(excluded.audio_url, episodes.audio_url),
                 link = COALESCE(excluded.link, episodes.link)""",
            (
                ep.id,
                ep.guid,
                ep.feed_url,
                ep.podcast,
                ep.title,
                ep.published.isoformat() if ep.published else None,
                ep.audio_url,
                ep.link,
                _now(),
            ),
        )
        self.conn.commit()

    def mark_seen(self, ep: Episode) -> None:
        """Record an episode as skipped (e.g. older than the first-run window)."""
        self.upsert(ep)
        self.conn.execute(
            "UPDATE episodes SET status = 'skipped' WHERE id = ? AND status = 'new'", (ep.id,)
        )
        self.conn.commit()

    def set_status(self, episode_id: str, status: str, **fields) -> None:
        cols = {"status": status, **fields}
        if status == "done":
            cols.setdefault("processed_at", _now())
            cols.setdefault("error", None)
        assignments = ", ".join(f"{k} = ?" for k in cols)
        self.conn.execute(
            f"UPDATE episodes SET {assignments} WHERE id = ?", (*cols.values(), episode_id)
        )
        self.conn.commit()

    def processed_between(self, start: datetime, end: datetime) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """SELECT * FROM episodes WHERE status = 'done'
                   AND processed_at >= ? AND processed_at < ? ORDER BY processed_at""",
                (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
            )
        )

    @staticmethod
    def summary_of(row: sqlite3.Row) -> dict | None:
        return json.loads(row["summary_json"]) if row["summary_json"] else None

    # --- costs ----------------------------------------------------------

    def log_cost(
        self,
        *,
        episode_id: str | None,
        stage: str,
        provider: str,
        model: str | None,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        audio_seconds: float = 0.0,
    ) -> None:
        self.conn.execute(
            """INSERT INTO costs (episode_id, stage, provider, model, input_tokens, output_tokens,
                                  cache_read_tokens, cache_write_tokens, audio_seconds, cost_usd,
                                  created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                stage,
                provider,
                model,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                audio_seconds,
                cost_usd,
                _now(),
            ),
        )
        self.conn.commit()

    def episode_cost(self, episode_id: str) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM costs WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return float(row["c"])

    def cost_report(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """SELECT COALESCE(e.title, '(digest)') AS title, c.stage, c.provider, c.model,
                          SUM(c.input_tokens) AS input_tokens,
                          SUM(c.output_tokens) AS output_tokens,
                          SUM(c.audio_seconds) AS audio_seconds, SUM(c.cost_usd) AS cost_usd
                   FROM costs c LEFT JOIN episodes e ON e.id = c.episode_id
                   GROUP BY c.episode_id, c.stage, c.provider, c.model
                   ORDER BY MIN(c.created_at)"""
            )
        )
