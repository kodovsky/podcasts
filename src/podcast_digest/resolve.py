"""Turn whatever the user pastes (Spotify/Apple/RSS/audio URL, local file) into an Episode.

Spotify doesn't expose podcast audio, so Spotify (and Apple) links are resolved to the show's
public RSS feed via the iTunes Search API, and the episode is matched there by title.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .feeds import fetch_feed, looks_like_feed, parse_feed
from .http import get
from .models import Episode

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".m4a", ".mp4", ".aac", ".wav", ".ogg", ".opus", ".flac", ".webm"}
ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
ITUNES_SEARCH = "https://itunes.apple.com/search"


class ResolveError(RuntimeError):
    pass


def _norm(s: str) -> str:
    s = html.unescape(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def match_episode(episodes: list[Episode], query: str, threshold: float = 0.6) -> Episode | None:
    scored = [(title_similarity(e.title, query), e) for e in episodes]
    scored.sort(key=lambda t: t[0], reverse=True)
    if scored and scored[0][0] >= threshold:
        log.info("Matched %r (score %.2f)", scored[0][1].title, scored[0][0])
        return scored[0][1]
    return None


def classify(url: str) -> str:
    """Return one of: file, spotify, apple, audio, url."""
    if not re.match(r"^https?://", url):
        return "file"
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("open.spotify.com") and "/episode/" in parsed.path:
        return "spotify"
    if host.endswith("podcasts.apple.com"):
        return "apple"
    if Path(parsed.path).suffix.lower() in AUDIO_EXTS:
        return "audio"
    return "url"


def parse_apple_ids(url: str) -> tuple[str | None, str | None]:
    """podcasts.apple.com/.../id863897795?i=1000712345678 -> ('863897795', '1000712345678')."""
    parsed = urlparse(url)
    m = re.search(r"/id(\d+)", parsed.path)
    show_id = m.group(1) if m else None
    episode_id = parse_qs(parsed.query).get("i", [None])[0]
    return show_id, episode_id


def page_meta(page_html: str) -> dict[str, str]:
    """Extract <meta property/name=... content=...> and <title> from an HTML page."""
    meta: dict[str, str] = {}
    for m in re.finditer(r"<meta\s+[^>]*>", page_html, re.I):
        tag = m.group(0)
        key = re.search(r'(?:property|name)\s*=\s*"([^"]+)"', tag)
        val = re.search(r'content\s*=\s*"([^"]*)"', tag)
        if key and val:
            meta.setdefault(key.group(1).lower(), html.unescape(val.group(1)))
    t = re.search(r"<title[^>]*>(.*?)</title>", page_html, re.I | re.S)
    if t:
        meta.setdefault("title", html.unescape(t.group(1).strip()))
    return meta


def spotify_show_name(meta: dict[str, str]) -> str | None:
    for key in ("og:description", "description", "twitter:description"):
        m = re.search(r"Listen to this episode from (.+?) on Spotify", meta.get(key, ""))
        if m:
            return m.group(1).strip()
    return None


def _itunes_episode(item: dict) -> Episode:
    released = item.get("releaseDate")
    return Episode(
        title=item.get("trackName", "Untitled"),
        podcast=item.get("collectionName", "Unknown podcast"),
        audio_url=item.get("episodeUrl"),
        guid=item.get("episodeGuid"),
        feed_url=item.get("feedUrl"),
        link=item.get("trackViewUrl"),
        published=datetime.fromisoformat(released.replace("Z", "+00:00")) if released else None,
        duration_seconds=(item["trackTimeMillis"] / 1000) if item.get("trackTimeMillis") else None,
        description=item.get("description"),
    )


def _episode_from_feed(feed_url: str, title: str) -> Episode | None:
    try:
        _, episodes = fetch_feed(feed_url)
    except Exception as e:  # feed unreachable -> caller falls back to iTunes metadata
        log.warning("Could not read feed %s: %s", feed_url, e)
        return None
    return match_episode(episodes, title)


def resolve_apple(url: str) -> Episode:
    show_id, episode_id = parse_apple_ids(url)
    if not show_id:
        raise ResolveError(f"Could not find a show id in {url}")
    data = get(ITUNES_LOOKUP, id=show_id, entity="podcastEpisode", limit=200).json()
    results = data.get("results", [])
    show = next((r for r in results if r.get("kind") == "podcast"), None)
    feed_url = show.get("feedUrl") if show else None
    ep_item = next((r for r in results if str(r.get("trackId")) == str(episode_id)), None)

    title = ep_item.get("trackName") if ep_item else None
    if not title:  # older than the 200 most recent: read the title from the page
        title = page_meta(get(url).text).get("og:title")
    if not title:
        raise ResolveError("Could not determine the episode title from the Apple link")

    if feed_url and (ep := _episode_from_feed(feed_url, title)):
        ep.link = ep.link or url
        return ep
    if ep_item:
        ep = _itunes_episode(ep_item)
        ep.feed_url = ep.feed_url or feed_url
        return ep
    raise ResolveError(f"Found the title {title!r} but not the episode audio")


def resolve_spotify(url: str) -> Episode:
    meta = page_meta(get(url).text)
    title = meta.get("og:title") or meta.get("title", "").split("|")[0].strip()
    if not title:
        raise ResolveError("Could not read the episode title from the Spotify page")
    show = spotify_show_name(meta)
    log.info("Spotify episode: %r from %r", title, show)

    # 1) Find the show's RSS feed and match the episode there (gives transcripts, full metadata).
    if show:
        shows = get(ITUNES_SEARCH, term=show, media="podcast", entity="podcast", limit=5).json()
        for s in shows.get("results", []):
            if s.get("feedUrl") and title_similarity(s.get("collectionName", ""), show) >= 0.8:
                if ep := _episode_from_feed(s["feedUrl"], title):
                    ep.link = ep.link or url
                    return ep

    # 2) Fall back to iTunes episode search by title.
    found = get(
        ITUNES_SEARCH, term=title, media="podcast", entity="podcastEpisode", limit=25
    ).json()
    candidates = found.get("results", [])
    if show:
        candidates = [
            c for c in candidates if title_similarity(c.get("collectionName", ""), show) >= 0.8
        ] or candidates
    best = max(
        candidates, key=lambda c: title_similarity(c.get("trackName", ""), title), default=None
    )
    if best and title_similarity(best.get("trackName", ""), title) >= 0.6:
        ep = _itunes_episode(best)
        ep.link = url
        return ep
    raise ResolveError(
        f"Couldn't find {title!r} outside Spotify (it may be a Spotify exclusive). "
        "Try the Apple Podcasts link or the RSS feed URL with --match."
    )


def resolve(target: str, match: str | None = None) -> Episode:
    kind = classify(target)
    log.info("Resolving %s (%s)", target, kind)
    if kind == "file":
        path = Path(target).expanduser()
        if not path.exists():
            raise ResolveError(f"No such file: {path}")
        return Episode(title=path.stem, podcast="Local", local_audio=str(path.resolve()))
    if kind == "spotify":
        return resolve_spotify(target)
    if kind == "apple":
        return resolve_apple(target)
    if kind == "audio":
        name = Path(urlparse(target).path).stem
        return Episode(title=name, podcast="Unknown podcast", audio_url=target, link=target)

    # Generic URL: treat as an RSS feed.
    resp = get(target)
    if not looks_like_feed(resp.content):
        raise ResolveError(
            f"{target} is not a Spotify/Apple link, audio file or RSS feed I can read"
        )
    _, episodes = parse_feed(resp.content, feed_url=target)
    if not episodes:
        raise ResolveError("Feed has no episodes")
    if match:
        ep = match_episode(episodes, match)
        if not ep:
            raise ResolveError(f"No episode in the feed matches {match!r}")
        return ep
    return episodes[0]
