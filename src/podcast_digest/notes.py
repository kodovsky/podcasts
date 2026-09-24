"""Render Obsidian-friendly Markdown notes."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .config import Config
from .models import Episode, Transcript, fmt_ts
from .summarize import EpisodeSummary, to_paragraphs


def safe_filename(name: str, max_len: int = 120) -> str:
    """Strip characters that break file systems or Obsidian links."""
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].rstrip(" .") or "Untitled"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _ts(t: str | None) -> str:
    return f"`{t}` " if t else ""


def _frontmatter(data: dict) -> str:
    data = {k: v for k, v in data.items() if v not in (None, [], "")}
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def note_basename(ep: Episode) -> str:
    date = ep.published.date().isoformat() if ep.published else "undated"
    return safe_filename(f"{date} {ep.title}")


def episode_dir(cfg: Config, ep: Episode) -> Path:
    return cfg.output_dir / safe_filename(ep.podcast)


def render_episode(
    ep: Episode,
    s: EpisodeSummary,
    transcript: Transcript,
    cfg: Config,
    *,
    model: str,
    cost_usd: float,
    transcript_note: str | None,
) -> str:
    wl = cfg.output.wikilink_resources
    tags = [
        *cfg.output.tags,
        f"podcast/{slug(ep.podcast)}",
        *(f"topic/{slug(t)}" for t in s.topics),
    ]
    fm = _frontmatter(
        {
            "title": ep.title,
            "podcast": ep.podcast,
            "published": ep.published.date().isoformat() if ep.published else None,
            "guests": s.guests,
            "duration": fmt_ts(ep.duration_seconds or transcript.duration),
            "url": ep.link,
            "audio": ep.audio_url,
            "transcript_source": f"{transcript.source}"
            + (f" ({transcript.model})" if transcript.model else ""),
            "summary_model": model,
            "cost_usd": round(cost_usd, 4),
            "processed": datetime.now(UTC).date().isoformat(),
            "tags": tags,
        }
    )
    out = [fm, f"# {ep.title}\n", f"> {s.one_liner}\n"]
    if s.guests:
        out.append(f"**Guests:** {', '.join(s.guests)}\n")

    out.append("## Summary\n")
    out += [f"- {b}" for b in s.summary]

    if s.non_obvious_ideas:
        out.append("\n## Non-obvious ideas\n")
        out += [
            f"- {_ts(i.timestamp)}**{i.idea}** — {i.why_it_matters}" for i in s.non_obvious_ideas
        ]

    if s.actionable_advice:
        out.append("\n## Actionable advice\n")
        out += [f"- [ ] {_ts(a.timestamp)}{a.advice} — _{a.context}_" for a in s.actionable_advice]

    for group in s.interest_ideas:
        if not group.ideas:
            continue
        out.append(f"\n## Relevant to: {group.interest}\n")
        out += [f"- {_ts(i.timestamp)}**{i.idea}** — {i.why_it_matters}" for i in group.ideas]

    if s.resources:
        out.append("\n## Books, tools & resources\n")
        for r in sorted(s.resources, key=lambda r: (r.kind, r.name.lower())):
            name = (
                f"[[{safe_filename(r.name)}]]"
                if wl and r.kind in ("book", "tool", "app")
                else r.name
            )
            by = f" by {r.creator}" if r.creator else ""
            out.append(f"- **{r.kind}**: {name}{by} — {r.context} {_ts(r.timestamp)}".rstrip())

    if s.notable_quotes:
        out.append("\n## Quotes\n")
        for q in s.notable_quotes:
            who = f" — {q.speaker}" if q.speaker else ""
            out.append(f"> {q.quote}{who} {_ts(q.timestamp)}".rstrip() + "\n")

    if transcript_note:
        out.append(f"\n---\nFull transcript: [[{transcript_note}]]")
    return "\n".join(out).rstrip() + "\n"


def render_transcript(ep: Episode, transcript: Transcript, episode_note: str) -> str:
    fm = _frontmatter(
        {
            "title": f"Transcript: {ep.title}",
            "podcast": ep.podcast,
            "source": transcript.source,
            "tags": ["podcast/transcript"],
        }
    )
    body = "\n\n".join(p for _, p in to_paragraphs(transcript))
    return f"{fm}# Transcript: {ep.title}\n\nSummary: [[{episode_note}]]\n\n{body}\n"


def write_episode_notes(
    ep: Episode,
    s: EpisodeSummary,
    transcript: Transcript,
    cfg: Config,
    *,
    model: str,
    cost_usd: float,
) -> Path:
    folder = episode_dir(cfg, ep)
    folder.mkdir(parents=True, exist_ok=True)
    base = note_basename(ep)
    transcript_note = None
    if cfg.output.write_transcript_notes:
        tdir = folder / "Transcripts"
        tdir.mkdir(exist_ok=True)
        transcript_note = f"{base} (transcript)"
        (tdir / f"{transcript_note}.md").write_text(render_transcript(ep, transcript, base))
    path = folder / f"{base}.md"
    path.write_text(
        render_episode(
            ep, s, transcript, cfg, model=model, cost_usd=cost_usd, transcript_note=transcript_note
        )
    )
    return path
