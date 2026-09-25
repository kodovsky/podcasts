"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from .config import load_config


def _setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    file_handler = logging.FileHandler(log_dir / "podcast-digest.log")
    file_handler.setFormatter(logging.Formatter(fmt))
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="podcast-digest",
        description="Podcast -> transcript -> Claude summary -> Obsidian notes.",
    )
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # Arguments shared by `episode` and `transcribe`.
    target = argparse.ArgumentParser(add_help=False)
    target.add_argument(
        "target",
        help="Spotify or Apple Podcasts episode link, RSS feed URL, audio URL, or local audio file",
    )
    target.add_argument(
        "--match", help="With an RSS feed URL: pick the episode whose title matches"
    )
    target.add_argument(
        "--engine",
        choices=["mlx-whisper", "faster-whisper", "deepgram"],
        help="Force a transcription engine (skips published transcripts)",
    )
    target.add_argument("--title", help="Episode title (useful for local files)")
    target.add_argument("--podcast", help="Podcast name (useful for local files)")

    ep = sub.add_parser(
        "episode",
        parents=[target],
        help="Transcribe + summarize one episode and write its note",
    )
    ep.add_argument("--force", action="store_true", help="Reprocess even if already done")
    ep.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the link and show what would be processed",
    )

    tr = sub.add_parser(
        "transcribe",
        parents=[target],
        help="Only transcribe (free, local) and preview the result; no Claude call",
    )
    tr.add_argument("--lines", type=int, default=8, help="Paragraphs to preview")

    sub.add_parser("run", help="Process new episodes from all feeds in the config")

    ls = sub.add_parser("list", help="List recent episodes of a configured feed")
    ls.add_argument("feed", help="Feed name (substring) from the config, or an RSS URL")
    ls.add_argument("-n", type=int, default=10)

    dg = sub.add_parser("digest", help="Write the weekly digest")
    dg.add_argument("--week", help="ISO week like 2026-W39 (default: current week)")
    dg.add_argument("--no-llm", action="store_true", help="Just list the week's episodes")

    sub.add_parser("costs", help="Show the cost ledger")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg.log_dir, args.verbose)
    log = logging.getLogger("podcast_digest")

    from .db import DB

    db = DB(cfg.db_path)
    try:
        if args.command in ("episode", "transcribe"):
            from .resolve import ResolveError, resolve

            try:
                episode = resolve(args.target, match=args.match)
            except ResolveError as e:
                log.error("%s", e)
                return 2
            except httpx.HTTPError as e:
                log.error("Network error while resolving %s: %s", args.target, e)
                return 2
            episode.title = args.title or episode.title
            episode.podcast = args.podcast or episode.podcast

        if args.command == "transcribe":
            from .summarize import to_paragraphs
            from .transcribe import get_transcript

            t = get_transcript(episode, cfg, db, engine=args.engine)
            paras = to_paragraphs(t)
            print(f"\n{episode.podcast} — {episode.title}")
            print(f"{t.source} ({t.model}), {len(t.segments)} segments, {t.duration / 60:.0f} min")
            print(f"Saved: {cfg.transcripts_dir / (episode.id + '.json')}\n")
            for _, text in paras[: args.lines]:
                print(text[:300] + ("…" if len(text) > 300 else ""), "\n")
            print(
                "Next: podcast-digest episode <same file/link and --title/--podcast> to summarize"
            )
        elif args.command == "episode":
            from .pipeline import process_episode

            if args.dry_run:
                print(f"{episode.podcast} — {episode.title}")
                print(f"  published: {episode.published}  audio: {episode.audio_url}")
                print(f"  transcripts in feed: {[t.type for t in episode.transcripts] or 'none'}")
                return 0
            path = process_episode(episode, cfg, db, force=args.force, engine=args.engine)
            if path:
                print(path)
        elif args.command == "run":
            from .pipeline import run_feeds

            for p in run_feeds(cfg, db):
                print(p)
        elif args.command == "list":
            from .feeds import fetch_feed

            url = args.feed
            if not url.startswith("http"):
                matches = [f for f in cfg.feeds if args.feed.lower() in f.name.lower()]
                if not matches:
                    log.error("No feed in the config matches %r", args.feed)
                    return 2
                url = matches[0].url
            _, episodes = fetch_feed(url)
            for e in episodes[: args.n]:
                row = db.get(e.id)
                status = row["status"] if row else "-"
                day = e.published.date().isoformat() if e.published else "????-??-??"
                print(f"{day}  [{status:>11}]  {e.title}")
        elif args.command == "digest":
            from .digest import build_digest

            path = build_digest(cfg, db, week=args.week, use_llm=not args.no_llm)
            print(path or "No episodes processed that week.")
        elif args.command == "costs":
            total = 0.0
            for r in db.cost_report():
                total += r["cost_usd"]
                print(
                    f"${r['cost_usd']:8.4f}  {r['stage']:<16} {r['model'] or '':<18} "
                    f"in={r['input_tokens']:>7} out={r['output_tokens']:>6}  {r['title'][:60]}"
                )
            print(f"${total:8.4f}  TOTAL")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
