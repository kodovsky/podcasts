"""Weekly digest across all episodes processed in an ISO week."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Config
from .db import DB
from .notes import _frontmatter, _ts
from .summarize import Summarizer

log = logging.getLogger(__name__)


class Theme(BaseModel):
    theme: str
    explanation: str = Field(description="How it showed up across episodes; name the episodes.")


class TopAction(BaseModel):
    action: str
    source_episode: str = Field(description="Exact episode note name it came from.")
    timestamp: str | None


class InterestSynthesis(BaseModel):
    interest: str
    synthesis: str = Field(description="What this week's episodes add up to for this interest.")
    next_steps: list[str]


class WeeklyDigest(BaseModel):
    headline: str = Field(description="One sentence capturing the week.")
    themes: list[Theme] = Field(
        description="Recurring themes across episodes; empty if just one episode."
    )
    top_actions: list[TopAction] = Field(description="The 5-7 highest-leverage actions this week.")
    interests: list[InterestSynthesis] = Field(description="One entry per listener interest.")
    worth_revisiting: list[str] = Field(
        description="Episode note names worth a full listen, with a reason after ' — '."
    )


DIGEST_SYSTEM = """You write a weekly digest for a listener from structured notes of the \
podcast episodes they processed this week. Be concrete and opinionated; skip filler. \
The listener's interests, in priority order:
{interests}"""

DIGEST_PROMPT = """Week {week}. Notes for {n} episode(s), keyed by episode note name:

<episodes>
{episodes}
</episodes>"""


def iso_week_bounds(week: str | None) -> tuple[str, datetime, datetime]:
    """'2026-W39' (or None for the current week) -> (label, monday 00:00 UTC, next monday)."""
    if week:
        year, w = week.upper().split("-W")
        monday = date.fromisocalendar(int(year), int(w), 1)
    else:
        today = datetime.now(UTC).date()
        monday = today - timedelta(days=today.weekday())
    y, w, _ = monday.isocalendar()
    start = datetime(monday.year, monday.month, monday.day, tzinfo=UTC)
    return f"{y}-W{w:02d}", start, start + timedelta(days=7)


def build_digest(cfg: Config, db: DB, week: str | None = None, use_llm: bool = True) -> Path | None:
    label, start, end = iso_week_bounds(week)
    rows = db.processed_between(start, end)
    if not rows:
        log.info("No episodes processed in %s", label)
        return None

    entries = []
    for r in rows:
        note = Path(r["note_path"]).stem if r["note_path"] else r["title"]
        entries.append((note, r, DB.summary_of(r) or {}))

    out = [
        _frontmatter(
            {
                "title": f"Podcast digest {label}",
                "week": label,
                "episodes": len(entries),
                "tags": [*cfg.output.tags, "podcast/digest"],
            }
        ),
        f"# Podcast digest {label}\n",
        f"_{start.date().isoformat()} – {(end - timedelta(days=1)).date().isoformat()}_\n",
    ]

    if use_llm:
        interests = cfg.interests or ["General self-improvement"]
        summarizer = Summarizer(cfg, db)
        d = summarizer.call(
            system=DIGEST_SYSTEM.format(interests="\n".join(f"- {i}" for i in interests)),
            prompt=DIGEST_PROMPT.format(
                week=label,
                n=len(entries),
                episodes=json.dumps({n: s for n, _, s in entries}, ensure_ascii=False, indent=1),
            ),
            output_format=WeeklyDigest,
            stage="digest",
            episode_id=None,
            model=cfg.summarization.digest_model,
        )
        out.append(f"> {d.headline}\n")
        if d.themes:
            out.append("## Themes\n")
            out += [f"- **{t.theme}** — {t.explanation}" for t in d.themes]
        out.append("\n## Top actions\n")
        out += [
            f"- [ ] {a.action} — [[{a.source_episode}]] {_ts(a.timestamp)}".rstrip()
            for a in d.top_actions
        ]
        for i in d.interests:
            out.append(f"\n## {i.interest}\n\n{i.synthesis}\n")
            out += [f"- [ ] {s}" for s in i.next_steps]
        if d.worth_revisiting:
            out.append("\n## Worth a full listen\n")
            out += [f"- {w}" for w in d.worth_revisiting]
        log.info("Digest cost $%.4f", summarizer.usage.cost_usd)

    out.append("\n## Episodes\n")
    for note, r, s in entries:
        out.append(f"### [[{note}]]\n_{r['podcast']}_ — {s.get('one_liner', '')}\n")
        out += [f"- {b}" for b in s.get("summary", [])]
        out.append("")

    path = cfg.output_dir / "Digests" / f"Podcast digest {label}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n")
    return path
